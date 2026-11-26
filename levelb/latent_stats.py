#!/usr/bin/env python3
"""Statistics of the bridge latents and of what the Stage-0 codes keep of them.

    python -m levelb.latent_stats --data $R/imagenet256 --tok $R/adarq_tok --out stats.json

No model is trained or sampled except a linear probe. On the precomputed Qwen latents:

* energy: share held by the global mean, by class means (between-class), and left
  unexplained by the codes at depth 1 / adaptive / depth 4 (uncentred, as Stage 0 reports,
  and after removing the global mean);
* spectrum: eigenvalues of the patch covariance, total and within-class, as effective
  dimension (participation ratio) and the number of components for 50..99% of the variance;
* class information: top-1/top-5 of one linear probe recipe (class from the image's
  mean-pooled latent; trained on train images, scored on the held-out slice) for the true
  latent, each code-only reconstruction, and the residual the codes leave behind.

Validation statistics only (train split + held-out slice); nothing here is a test result.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from .oracle_render import bridge_latents, load_quantizers
from .precompute import read_latents

VARIANTS = ("latent", "depth1", "adaptive", "depth4", "residual")
FRACTIONS = (0.5, 0.8, 0.9, 0.95, 0.99)


def pick_train(train: np.ndarray, labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Up to ``per_class`` train rows of every class (seeded), sorted."""
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels[train])
    order = rng.permutation(len(train))
    rank = np.empty(len(train), dtype=np.int64)
    for c in np.unique(lab):
        rows = order[lab[order] == c]
        rank[rows] = np.arange(len(rows))
    return np.sort(train[rank < per_class])


def spectrum(cov: torch.Tensor) -> dict:
    ev = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
    cum = (ev.cumsum(0) / ev.sum()).cpu().numpy()
    return {
        "trace": float(ev.sum()),
        "participation_ratio": float(ev.sum() ** 2 / ev.pow(2).sum()),
        "components_for": {str(f): int(np.searchsorted(cum, f) + 1) for f in FRACTIONS},
        "top_eigenvalue_share": [float(x) for x in (ev[:10] / ev.sum()).cpu()],
    }


def linear_probe(xtr, ytr, xte, yte, classes: int, epochs: int, seed: int) -> dict:
    """Multinomial logistic regression on standardised features; one recipe for every
    variant, so the numbers are comparable with each other (not tuned)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    w = torch.zeros(xtr.shape[1], classes, device=xtr.device, requires_grad=True)
    b = torch.zeros(classes, device=xtr.device, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), generator=g).to(xtr.device)
        for s in range(0, len(perm), 2048):
            i = perm[s : s + 2048]
            loss = torch.nn.functional.cross_entropy(xtr[i] @ w + b, ytr[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        top = (xte @ w + b).topk(5, dim=-1).indices
        hit = top == yte[:, None]
    return {"top1": float(hit[:, 0].float().mean()), "top5": float(hit.any(-1).float().mean())}


@torch.no_grad()
def collect(lat, labels, rows, qa, qf, dev, batch: int, classes: int, moments: bool):
    """Mean-pooled features per variant [n, d]; with ``moments``, also the patch second
    moments needed for the energy and spectrum statistics (float64 sums)."""
    d = lat.shape[-1]
    feats = {v: torch.empty(len(rows), d) for v in VARIANTS}
    m = None
    if moments:
        m = {
            "n": 0,
            "sum": torch.zeros(d, dtype=torch.float64, device=dev),
            "outer": torch.zeros(d, d, dtype=torch.float64, device=dev),
            "class_sum": torch.zeros(classes, d, dtype=torch.float64, device=dev),
            "class_n": torch.zeros(classes, dtype=torch.float64, device=dev),
            "err": {v: 0.0 for v in VARIANTS[1:4]},
            "depth_sum": 0,
        }
    for s in range(0, len(rows), batch):
        idx = rows[s : s + batch]
        z = read_latents(lat[idx]).float().to(dev)
        recon = {"latent": z}
        for v in VARIANTS[1:4]:
            recon[v], depth, _ = bridge_latents(z, qa, qf, v)
            if moments:
                m["err"][v] += float((z - recon[v]).double().pow(2).sum())
                if v == "adaptive":
                    m["depth_sum"] += int(depth.sum())
        recon["residual"] = z - recon["adaptive"]
        for v in VARIANTS:
            feats[v][s : s + len(idx)] = recon[v].mean(1).cpu()
        if moments:
            flat = z.reshape(-1, d).double()
            y = torch.from_numpy(np.asarray(labels[idx], dtype=np.int64)).to(dev)
            m["n"] += len(flat)
            m["sum"] += flat.sum(0)
            m["outer"] += flat.T @ flat
            m["class_sum"].index_add_(0, y, z.double().sum(1))
            m["class_n"].index_add_(0, y, torch.full_like(y, z.shape[1], dtype=torch.float64))
    return feats, m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=50, help="train images per class")
    ap.add_argument("--probe-epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = pathlib.Path(args.data)
    dev = torch.device(args.device)
    lat = np.load(root / "latents_train.npy", mmap_mode="r")
    labels = np.load(root / "labels_train.npy", mmap_mode="r")
    split = np.load(root / "split_seed0.npz")
    held = np.sort(split["heldout"])
    train = pick_train(np.sort(split["train"]), labels, args.per_class, args.seed)
    classes = int(np.asarray(labels[held]).max()) + 1
    qa, qf = (q.to(dev) for q in load_quantizers(args.tok))

    ftr, m = collect(lat, labels, train, qa, qf, dev, args.batch, classes, moments=True)
    fte, _ = collect(lat, labels, held, qa, qf, dev, args.batch, classes, moments=False)

    n = m["n"]
    mean = m["sum"] / n
    energy = float(m["outer"].diagonal().sum() / n)  # E||z||^2 per patch
    cov = m["outer"] / n - torch.outer(mean, mean)
    seen = m["class_n"] > 0
    cmean = m["class_sum"][seen] / m["class_n"][seen, None]
    wts = m["class_n"][seen] / n
    dc = cmean - mean
    between = (dc * wts[:, None]).T @ dc
    var = float(cov.diagonal().sum())
    res = {
        "train_images": int(len(train)),
        "heldout_images": int(len(held)),
        "patches": int(n),
        "width": int(lat.shape[-1]),
        "adaptive_mean_depth": m["depth_sum"] / n,
        "energy_per_patch": energy,
        "share_global_mean": float(mean.pow(2).sum()) / energy,
        "share_between_class_of_variance": float(between.diagonal().sum()) / var,
        "codes_unexplained": {
            v: {"of_energy": m["err"][v] / n / energy, "of_variance": m["err"][v] / n / var}
            for v in VARIANTS[1:4]
        },
        "spectrum_total": spectrum(cov),
        "spectrum_within_class": spectrum(cov - between),
        "probe": {},
    }
    ytr = torch.from_numpy(np.asarray(labels[train], dtype=np.int64)).to(dev)
    yte = torch.from_numpy(np.asarray(labels[held], dtype=np.int64)).to(dev)
    for v in VARIANTS:
        res["probe"][v] = linear_probe(
            ftr[v].to(dev), ytr, fte[v].to(dev), yte, classes, args.probe_epochs, args.seed
        )
        print(f"probe {v:9s} top-1 {res['probe'][v]['top1']:.4f}", flush=True)
    res["args"] = vars(args)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
