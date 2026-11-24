#!/usr/bin/env python3
"""Requested-class top-1 / top-5 accuracy of generated images with a fixed ImageNet classifier.

    python -m levelb.class_accuracy --samples <samples_N.npz> [--device cuda]
    python -m levelb.class_accuracy --heldout $LEVELB_ROOT/imagenet256 --n 2000   # real ceiling

IS rewards confident predictions of *some* class; this measures whether an image shows the
class it was generated for. Classifier: torchvision ResNet-50, weights IMAGENET1K_V2, with
its own preprocessing (resize 232, centre crop 224, ImageNet normalisation), fixed for every
arm. The ceiling is the same classifier on real held-out training crops of the same classes.
Writes <samples stem>.cls.metrics.json (or the --out path); with --per-class,
<samples stem>.clsclass.metrics.json, adding per requested class: top-1 hits, mean
probability of the requested class and mean prediction entropy.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

WEIGHTS = "IMAGENET1K_V2"


def topk_hits(logits: torch.Tensor, labels: torch.Tensor, ks=(1, 5)) -> dict[int, int]:
    """Number of rows whose label is among the top-k logits, for each k."""
    top = logits.topk(max(ks), dim=-1).indices
    hit = top == labels[:, None]
    return {k: int(hit[:, :k].any(-1).sum()) for k in ks}


@torch.no_grad()
def classify(
    images: np.ndarray, labels: np.ndarray, device: str, batch: int = 100, per_class: bool = False
) -> dict:
    from torchvision.models import ResNet50_Weights, resnet50

    weights = getattr(ResNet50_Weights, WEIGHTS)
    model = resnet50(weights=weights).eval().to(device)
    prep = weights.transforms()
    hits, n = {1: 0, 5: 0}, 0
    top1_img, p_req, ent = [], [], []
    for s in range(0, len(images), batch):
        x = torch.from_numpy(np.ascontiguousarray(images[s : s + batch])).permute(0, 3, 1, 2)
        y = torch.from_numpy(np.asarray(labels[s : s + batch], dtype=np.int64)).to(device)
        logits = model(prep(x).to(device))
        out = topk_hits(logits, y)
        hits = {k: hits[k] + out[k] for k in hits}
        n += len(y)
        if per_class:
            lp = logits.float().log_softmax(-1)
            top1_img.append((logits.argmax(-1) == y).cpu().numpy())
            p_req.append(lp.gather(1, y[:, None])[:, 0].exp().cpu().numpy())
            ent.append((-(lp.exp() * lp).sum(-1)).cpu().numpy())
    res = {
        "top1": hits[1] / n,
        "top5": hits[5] / n,
        "n": n,
        "classifier": f"torchvision resnet50 {WEIGHTS}",
    }
    if per_class:
        res |= per_class_summary(
            np.asarray(labels), np.concatenate(top1_img), np.concatenate(p_req), np.concatenate(ent)
        )
    return res


def per_class_summary(
    labels: np.ndarray, top1: np.ndarray, p_req: np.ndarray, ent: np.ndarray
) -> dict:
    """Per requested class: images, top-1 hits, mean requested-class probability, mean
    entropy (all as lists over the classes present, in label order)."""
    labels = np.asarray(labels, dtype=np.int64)
    cls = np.unique(labels)
    idx = np.searchsorted(cls, labels)
    cnt = np.bincount(idx, minlength=len(cls))
    return {
        "p_req_mean": float(p_req.mean()),
        "entropy_mean": float(ent.mean()),
        "per_class": {
            "label": cls.tolist(),
            "n": cnt.tolist(),
            "hits": np.bincount(idx, weights=top1.astype(np.float64), minlength=len(cls))
            .astype(int)
            .tolist(),
            "p_req": (np.bincount(idx, weights=p_req, minlength=len(cls)) / cnt).tolist(),
            "entropy": (np.bincount(idx, weights=ent, minlength=len(cls)) / cnt).tolist(),
        },
    }


def heldout_rows(root: pathlib.Path, n: int) -> np.ndarray:
    """Row indices (sorted) of a class-balanced validation set: the first n/1000 held-out
    images of every class. Shared with levelb.oracle_render, so both see the same images."""
    held = np.sort(np.load(root / "split_seed0.npz")["heldout"])
    lab = np.asarray(np.load(root / "labels_train.npy", mmap_mode="r")[held])
    per = n // 1000
    if per < 1 or n % 1000:
        raise SystemExit("--n must be a positive multiple of 1000 (class-balanced)")
    pick = []
    for c in range(1000):
        rows = held[lab == c]
        if len(rows) < per:
            raise SystemExit(f"class {c}: only {len(rows)} held-out images, need {per}")
        pick.extend(rows[:per].tolist())
    return np.sort(np.asarray(pick))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", default=None, help="npz with arr_0 uint8 NHWC and labels")
    ap.add_argument("--heldout", default=None, help="imagenet256 dir: real held-out ceiling")
    ap.add_argument("--n", type=int, default=2000, help="--heldout: images, class-balanced")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--per-class",
        action="store_true",
        help="--samples: also per requested class, to <stem>.clsclass.metrics.json",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.per_class and not args.samples:
        raise SystemExit("--per-class needs --samples")
    if (args.samples is None) == (args.heldout is None):
        raise SystemExit("give exactly one of --samples or --heldout")
    if args.samples:
        path = pathlib.Path(args.samples)
        with np.load(path) as z:
            images, labels = z["arr_0"], np.asarray(z["labels"])
        suffix = ".clsclass.metrics.json" if args.per_class else ".cls.metrics.json"
        out = pathlib.Path(args.out) if args.out else path.with_name(path.stem + suffix)
        source = str(path)
    else:
        root = pathlib.Path(args.heldout)
        pick = heldout_rows(root, args.n)
        per = args.n // 1000
        crops = np.load(root / "crops_train.npy", mmap_mode="r")
        images = np.asarray(crops[pick])
        labels = np.asarray(np.load(root / "labels_train.npy", mmap_mode="r")[pick])
        out = pathlib.Path(args.out) if args.out else root / f"heldout_{args.n}.cls.metrics.json"
        source = f"held-out training crops, {per} per class"
    res = classify(images, labels, args.device, per_class=args.per_class) | {"source": source}
    out.write_text(json.dumps(res, indent=2))
    print(f"top-1 {res['top1']:.4f}  top-5 {res['top5']:.4f}  ({res['n']} images) -> {out}")


if __name__ == "__main__":
    main()
