#!/usr/bin/env python3
"""AdaRQ-Flow Stage 0: adaptive residual quantizer on the precomputed Qwen latents.

    python -m levelb.adarq_tokenizer --data $LEVELB_ROOT/imagenet256 --out $LEVELB_ROOT/adarq_tok

1. Fit per-depth EMA codebooks (K codes each, reserved zero codeword) at FULL depth on
   patches of the TRAIN split. Greedy RVQ codes are prefix-nested, so the halting rule
   only truncates them; fitting at full depth and choosing the threshold afterwards is
   therefore consistent (no circularity between threshold and codebooks).
2. Choose the halting threshold tau on the VALIDATION slice so the mean depth equals the
   pre-registered target (EXPERIMENTS.md, AdaRQ-Flow arm) — a rate choice, not a quality
   one, made before any test scoring.
3. Encode every train row: codes [N, 64, D_max] (int16; K = <halt>), and record the
   residual RMS used to normalise the flow head's target.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from adarq_flow.config import TokenizerConfig
from adarq_flow.tokenizer import AdaptiveResidualQuantizer

from .precompute import read_latents

TOKENS = 64


def make_quantizer(width: int, k: int, depth: int, tau: float = 0.5,
                   adaptive: bool = False) -> AdaptiveResidualQuantizer:
    cfg = TokenizerConfig(clip_dim=width, num_patches=TOKENS, codebook_size=k,
                          max_depth=depth, shared_codebook=False, include_zero_code=True,
                          dead_code_threshold=0.5, ema_decay=0.99, adaptive_depth=adaptive,
                          halt_residual_threshold=tau)
    return AdaptiveResidualQuantizer(cfg)


def _patches(lat: np.ndarray, idx: np.ndarray, device) -> torch.Tensor:
    return read_latents(lat[np.sort(idx)]).float().reshape(-1, lat.shape[-1]).to(device)


def fit(lat: np.ndarray, train: np.ndarray, width: int, k: int, depth: int, steps: int,
        images_per_step: int, seed: int, device) -> AdaptiveResidualQuantizer:
    q = make_quantizer(width, k, depth).to(device)
    q.train()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for s in range(steps):
        x = _patches(lat, rng.choice(train, images_per_step, replace=False), device)
        out = q(x, update_codebook=True)
        if s % 500 == 0 or s == steps - 1:
            print(f"  fit {s}/{steps} recon {out.losses.item()['recon']:.4f}", flush=True)
    return q.eval()


@torch.no_grad()
def residual_ratios(q: AdaptiveResidualQuantizer, x: torch.Tensor) -> torch.Tensor:
    """||r^(k)|| / ||z|| after each full-depth level k = 1..D: [M, D]."""
    out = q(x, update_codebook=False)
    zn = x.norm(dim=-1).clamp_min(1e-8)
    r = x.clone()
    ratios = []
    for lvl in range(q.max_depth):
        c = out.codes[:, lvl]
        r = r - q._book(lvl).embed[c.clamp_max(q.codebook_size - 1)] * (
            c != q.codebook_size).unsqueeze(-1)
        ratios.append(r.norm(dim=-1) / zn)
    return torch.stack(ratios, dim=-1)


def depth_at(ratios: torch.Tensor, tau: float) -> torch.Tensor:
    """The quantizer's rule: stop after the first level whose ratio < tau (cap D)."""
    below = ratios < tau
    first = torch.where(below.any(-1), below.float().argmax(-1) + 1,
                        torch.full_like(below[:, 0], ratios.shape[-1], dtype=torch.long))
    return first


def choose_tau(ratios: torch.Tensor, target: float, iters: int = 60) -> float:
    lo, hi = 1e-6, 1.0 - 1e-6           # depth(tau) is non-increasing in tau
    d_lo, d_hi = depth_at(ratios, lo).float().mean(), depth_at(ratios, hi).float().mean()
    if not (d_hi <= target <= d_lo):
        raise SystemExit(f"target mean depth {target} outside reachable [{d_hi:.2f}, {d_lo:.2f}]")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if depth_at(ratios, mid).float().mean() > target:
            lo = mid
        else:
            hi = mid
    return hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--codebook-size", type=int, default=1024)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--target-mean-depth", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--images-per-step", type=int, default=64)
    ap.add_argument("--val-images", type=int, default=4000)
    ap.add_argument("--encode-batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root, out = pathlib.Path(args.data), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    meta = json.loads((root / "meta.json").read_text())
    lat = np.load(root / "latents_train.npy", mmap_mode="r")
    split = np.load(root / "split_seed0.npz")
    train, held = split["train"], split["heldout"]
    width = int(meta["latent_width"])

    q = fit(lat, train, width, args.codebook_size, args.max_depth, args.steps,
            args.images_per_step, args.seed, dev)
    xv = _patches(lat, held[:args.val_images], dev)
    ratios = residual_ratios(q, xv)
    tau = choose_tau(ratios, args.target_mean_depth)
    d = depth_at(ratios, tau).float()
    final = residual_ratios(q, xv)[torch.arange(len(xv)), (d.long() - 1)]
    print(f"  tau {tau:.5f}: validation mean depth {d.mean():.3f} "
          f"(histogram {[int((d == k).sum()) for k in range(1, args.max_depth + 1)]}), "
          f"mean relative residual {final.mean():.3f}", flush=True)

    qa = make_quantizer(width, args.codebook_size, args.max_depth, tau=tau, adaptive=True)
    qa.load_state_dict(q.state_dict())
    qa = qa.to(dev).eval()
    res_sq, n_res = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(held[:args.val_images]), args.encode_batch):
            o = qa(_patches(lat, held[s:s + args.encode_batch], dev), update_codebook=False)
            res_sq += float(o.res.pow(2).sum())
            n_res += o.res.numel()
    res_rms = (res_sq / n_res) ** 0.5

    codes = np.lib.format.open_memmap(out / "codes_train.npy", mode="w+", dtype=np.int16,
                                      shape=(lat.shape[0], TOKENS, args.max_depth))
    with torch.no_grad():
        for s in range(0, lat.shape[0], args.encode_batch):
            e = min(s + args.encode_batch, lat.shape[0])
            x = read_latents(lat[s:e]).float().reshape(-1, width).to(dev)
            codes[s:e] = qa(x, update_codebook=False).codes.reshape(e - s, TOKENS, -1).cpu()
            if s % (100 * args.encode_batch) == 0:
                print(f"  encoded {e}/{lat.shape[0]}", flush=True)
    codes.flush()
    torch.save({"state_dict": qa.state_dict(), "tau": tau, "res_rms": res_rms,
                "codebook_size": args.codebook_size, "max_depth": args.max_depth,
                "width": width}, out / "tokenizer.pt")
    (out / "tokenizer.json").write_text(json.dumps(
        {**vars(args), "tau": tau, "val_mean_depth": float(d.mean()), "res_rms": res_rms},
        indent=2))
    print(f"[tokenizer] tau {tau:.5f}, residual RMS {res_rms:.4f} -> {out}")


if __name__ == "__main__":
    main()
