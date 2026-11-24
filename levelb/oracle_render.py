#!/usr/bin/env python3
"""Oracle renders: real held-out images' latents -> the same latent ControlNet + FLUX.

    python -m levelb.oracle_render --mode adaptive --tok $R/adarq_tok \
        --controlnet $R/controlnet/final_controlnet.pt --data $R/imagenet256 \
        --n 2000 --seed 100 --out $R/samples/oracle_adaptive_seed100

No MLLM and no trained head: this locates where class information is lost (EXPERIMENTS.md,
Step 0). The bridge input of each real image is, by ``--mode``:

    latent     its true Qwen latent z                      (ceiling of the renderer)
    adaptive   codes only, Stage 0's halting rule (mean depth 2): zhat, residual dropped
    depthK     codes only, the first K levels at every patch (K = 1 .. D_max)

Images: n/1000 per class from the held-out slice, the same rows as
``class_accuracy --heldout``, so real pixels, true-latent renders and code-only renders
are of the same images. Rendering is ``levelb.sample``'s (same prompts, steps, ControlNet
scale, guidance, per-batch seeds). Validation only; output in ``levelb.sample``'s format.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from .adarq_tokenizer import make_quantizer
from .class_accuracy import heldout_rows
from .flux_text import from_bits
from .precompute import read_latents
from .sample import load_pipeline
from .upstream import require_official_flux


def load_quantizers(tok_dir: str | pathlib.Path):
    """Stage 0's quantizer twice: with its halting rule, and at full depth."""
    blob = torch.load(
        pathlib.Path(tok_dir) / "tokenizer.pt", map_location="cpu", weights_only=False
    )
    make = lambda adaptive: make_quantizer(  # noqa: E731
        blob["width"], blob["codebook_size"], blob["max_depth"], tau=blob["tau"], adaptive=adaptive
    )
    qa, qf = make(True), make(False)
    qa.load_state_dict(blob["state_dict"])
    qf.load_state_dict(blob["state_dict"])
    return qa.eval(), qf.eval()


@torch.no_grad()
def bridge_latents(z: torch.Tensor, qa, qf, mode: str):
    """z [B, N, d] fp32 -> (bridge input [B, N, d], depth per patch [B, N] or None, the
    adaptive codes [B, N, D] or None)."""
    if mode == "latent":
        return z, None, None
    b, n, d = z.shape
    flat = z.reshape(-1, d)
    if mode == "adaptive":
        o = qa(flat, update_codebook=False)
        return o.zhat.reshape(b, n, d), o.depths.reshape(b, n), o.codes.reshape(b, n, -1)
    if not (mode.startswith("depth") and mode[5:].isdigit()):
        raise ValueError(f"unknown mode {mode!r}")
    k = int(mode[5:])
    if not 1 <= k <= qf.max_depth:
        raise ValueError(f"{mode}: depth must be 1..{qf.max_depth}")
    codes = qf(flat, update_codebook=False).codes
    zhat = torch.zeros_like(flat)
    for lvl in range(k):
        zhat = zhat + qf._book(lvl).embed[codes[:, lvl]]
    return zhat.reshape(b, n, d), torch.full((b, n), k, device=z.device), None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, help="latent, adaptive, or depthK")
    ap.add_argument("--tok", default=None, help="Stage-0 dir (all modes but latent)")
    ap.add_argument("--controlnet", required=True, help="final_controlnet.pt of Stage 2")
    ap.add_argument("--data", required=True, help="imagenet256 dir")
    ap.add_argument("--flux", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--n", type=int, default=2000, help="images, class-balanced")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cn-scale", type=float, default=0.7)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    require_official_flux(args.flux)
    if args.mode != "latent" and not args.tok:
        raise SystemExit(f"--mode {args.mode} needs --tok")

    dev = torch.device(args.device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    root = pathlib.Path(args.data)
    meta = json.loads((root / "meta.json").read_text())
    width = int(meta["latent_width"])
    t5 = from_bits(np.load(root / "flux_t5.npy"))
    pooled = from_bits(np.load(root / "flux_pooled.npy"))
    rows = heldout_rows(root, args.n)
    labels = np.asarray(np.load(root / "labels_train.npy", mmap_mode="r")[rows]).astype(np.int64)
    lat = np.load(root / "latents_train.npy", mmap_mode="r")
    lo, hi = args.shard * args.n // args.nshards, (args.shard + 1) * args.n // args.nshards
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    part = out / f"shard{args.shard:02d}of{args.nshards:02d}.npz"
    if part.exists():
        print(f"{part} exists; nothing to do")
        return

    qa = qf = stored = None
    if args.mode != "latent":
        qa, qf = (q.to(dev) for q in load_quantizers(args.tok))
        if width != qa.cfg.clip_dim:
            raise SystemExit(f"tokenizer width {qa.cfg.clip_dim} != latent width {width}")
        stored = np.load(pathlib.Path(args.tok) / "codes_train.npy", mmap_mode="r")
    pipe = load_pipeline(args.flux, args.controlnet, width, dev, dtype)

    imgs = np.zeros((hi - lo, args.res, args.res, 3), dtype=np.uint8)
    err_sq = z_sq = rel_sum = 0.0
    n_patch = agree = n_codes = 0
    depth_hist = np.zeros(5, dtype=np.int64)
    for s in range(lo, hi, args.batch):
        e = min(s + args.batch, hi)
        idx = torch.from_numpy(labels[s:e])
        z = read_latents(lat[rows[s:e]]).float().to(dev)
        bridge, depth, codes = bridge_latents(z, qa, qf, args.mode)
        diff = (z - bridge).norm(dim=-1)
        err_sq += float(diff.pow(2).sum())
        z_sq += float(z.pow(2).sum())
        rel_sum += float((diff / z.norm(dim=-1).clamp_min(1e-8)).sum())
        n_patch += diff.numel()
        if depth is not None:
            depth_hist += np.bincount(depth.flatten().cpu().numpy(), minlength=5)[:5]
        if codes is not None:  # the encoder here must be the one that wrote Stage 0's codes
            ref = torch.from_numpy(np.asarray(stored[rows[s:e]], dtype=np.int64)).to(dev)
            agree += int((codes == ref).sum())
            n_codes += codes.numel()
        g = torch.Generator(device=dev).manual_seed(args.seed * 1_000_003 + s)
        res = pipe(
            prompt_embeds=t5[idx].to(dev, dtype),
            pooled_prompt_embeds=pooled[idx].to(dev, dtype),
            control_image=bridge.to(dtype),
            num_inference_steps=args.steps,
            controlnet_conditioning_scale=args.cn_scale,
            guidance_scale=args.guidance,
            generator=g,
            height=args.res,
            width=args.res,
            output_type="np",
        ).images
        imgs[s - lo : e - lo] = (np.clip(res, 0, 1) * 255).round().astype(np.uint8)
        print(f"  {e - lo}/{hi - lo} images", flush=True)
    code_agreement = agree / n_codes if n_codes else None
    if code_agreement is not None and code_agreement < 0.999:
        raise SystemExit(
            f"only {code_agreement:.4f} of the codes match Stage 0's stored codes: "
            "wrong tokenizer or latents"
        )
    np.savez(part, arr_0=imgs, labels=labels[lo:hi], rows=rows[lo:hi])
    (out / f"shard{args.shard:02d}of{args.nshards:02d}.json").write_text(
        json.dumps(
            {
                **vars(args),
                "arm": f"oracle:{args.mode}",
                "range": [lo, hi],
                "source": "real held-out latents, class-balanced (class_accuracy --heldout rows)",
                "mean_relative_residual": rel_sum / n_patch,
                "energy_unexplained": err_sq / z_sq,
                "mean_depth": float((depth_hist * np.arange(5)).sum() / depth_hist.sum())
                if depth_hist.sum()
                else None,
                "depth_hist": depth_hist[1:].tolist() if depth_hist.sum() else None,
                "stored_code_agreement": code_agreement,
            },
            indent=2,
        )
    )
    print(f"wrote {part}")


if __name__ == "__main__":
    main()
