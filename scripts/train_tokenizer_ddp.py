#!/usr/bin/env python3
"""Multi-GPU DISTRIBUTED-PLUMBING CHECK (not a training run).

Validates the full distributed path on real hardware: NCCL init from the launcher
env, per-rank data sharding, EMA codebook updates with cross-rank all-reduce, and a
post-training check that every rank holds an identical codebook.

This validates NCCL init, per-rank sharding, EMA all-reduce and cross-rank codebook
consistency. By default it operates on SYNTHETIC latents, which makes it a plumbing
check and NOT a scientific result — so synthetic mode must be requested explicitly with
--synthetic-plumbing-check. Pass --latents PATH to run the same checks on real CLIP
latents. It never writes a checkpoint, so its output cannot be mistaken for a trained
tokenizer.

Launch (Cineca Leonardo, 4 GPUs, 1 task/GPU):
    srun python scripts/train_tokenizer_ddp.py --preset base_gpu --steps 500
or with torchrun on a single node:
    torchrun --nproc_per_node=4 scripts/train_tokenizer_ddp.py --steps 500
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Make the script runnable directly (srun/torchrun) without `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from adarq_flow.config import get_preset
from adarq_flow.tokenizer import build_tokenizer
from adarq_flow.utils.distributed import (
    all_reduce_mean,
    barrier,
    cleanup,
    get_world_size,
    is_main_process,
    seed_everything,
    setup_distributed,
)


def _synthetic_latents(n, dim, n_clusters, device, info, base_seed):
    """Shared cluster centers (identical across ranks) + per-rank samples."""
    g = torch.Generator(device=device).manual_seed(base_seed)
    centers = torch.randn(n_clusters, dim, generator=g, device=device) * 3.0
    # Per-rank assignments/noise (different data on each GPU == real sharding).
    gr = torch.Generator(device=device).manual_seed(base_seed + 1000 + info.rank)
    assign = torch.randint(n_clusters, (n,), generator=gr, device=device)
    noise = 0.05 * torch.randn(n, dim, generator=gr, device=device)
    return centers[assign] + noise


def main():
    ap = argparse.ArgumentParser(description="Multi-GPU Stage-0 tokenizer trainer")
    ap.add_argument("--preset", default="base_gpu")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--samples-per-rank", type=int, default=8192)
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--backend", default=None, help="override cfg.dist.backend")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latents", default=None,
                    help="path to a .pt tensor of REAL CLIP patch latents [M, d]")
    ap.add_argument("--synthetic-plumbing-check", action="store_true",
                    help="acknowledge that synthetic latents are used and that the run "
                         "validates distributed plumbing only, producing no result")
    args = ap.parse_args()

    if not args.latents and not args.synthetic_plumbing_check:
        raise SystemExit(
            "refusing to run on synthetic latents implicitly.\n"
            "  --latents PATH                 run the checks on real CLIP latents, or\n"
            "  --synthetic-plumbing-check     acknowledge a plumbing-only run.\n"
            "This guard exists so a synthetic run on the cluster cannot be mistaken for "
            "a Stage-0 training result."
        )

    cfg = get_preset(args.preset)
    backend = args.backend or cfg.dist.backend
    info = setup_distributed(backend=backend, init_timeout_min=cfg.dist.init_timeout_min)
    seed_everything(args.seed, rank=info.rank, seed_per_rank=cfg.dist.seed_per_rank)

    if is_main_process():
        print(f"[adarq-flow] world_size={info.world_size} backend={backend} "
              f"device={info.device} preset={args.preset}", flush=True)

    tok = build_tokenizer(cfg).to(info.device)
    dim = cfg.tokenizer.clip_dim
    if args.latents:
        latents = torch.load(args.latents, map_location=info.device, weights_only=True)
        if latents.dim() == 3:
            latents = latents.reshape(-1, latents.size(-1))
        if latents.size(-1) != dim:
            raise ValueError(f"latents dim {latents.size(-1)} != preset clip_dim {dim}")
        source = f"real latents from {args.latents}"
    else:
        latents = _synthetic_latents(
            args.samples_per_rank, dim, args.clusters, info.device, info, args.seed
        )
        source = "SYNTHETIC latents - PLUMBING CHECK ONLY, not a result"
    if is_main_process():
        print(f"[adarq-flow] data source: {source}", flush=True)

    tok.train()
    gen = torch.Generator(device=info.device).manual_seed(args.seed + info.rank)
    for step in range(args.steps):
        idx = torch.randint(latents.shape[0], (args.batch_size,), generator=gen, device=info.device)
        out = tok(latents[idx], update_codebook=True)  # EMA all-reduces across ranks
        if is_main_process() and (step % 50 == 0 or step == args.steps - 1):
            stats = out.losses.item()
            print(f"  step {step:5d}  recon={stats['recon']:.4f}  "
                  f"rate={stats['rate']:.2f}  "
                  f"usage={float(tok.codebook_usage()):.2f}", flush=True)

    barrier()

    # Verify codebooks are identical across ranks: mean-reduce a copy and compare.
    book = tok.quantizer.codebooks[0].embed
    ref = book.clone()
    all_reduce_mean(ref)  # == book on every rank iff all replicas agree
    max_dev = (book - ref).abs().max().item()
    if is_main_process():
        ok = "OK" if max_dev < 1e-5 else f"DIVERGED (max_dev={max_dev:.2e})"
        print(f"[adarq-flow] cross-rank codebook consistency: {ok}", flush=True)
    if get_world_size() > 1 and max_dev >= 1e-5:
        raise RuntimeError(f"codebooks diverged across ranks (max_dev={max_dev:.2e})")

    cleanup()


if __name__ == "__main__":
    main()
