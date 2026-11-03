#!/usr/bin/env python3
"""Multi-GPU DISTRIBUTED CHECK of the Stage-0 quantizer on REAL CLIP latents.

Validates the distributed path on real hardware: NCCL init from the launcher env,
per-rank sharding, EMA codebook updates with cross-rank all-reduce, and a final check
that every rank holds an identical codebook. It never writes a checkpoint, so its output
cannot be mistaken for a trained tokenizer.

Real latents only: ``--latents`` is required (written by scripts/extract_clip_latents.py,
whose ``.meta.json`` must mark them ``"synthetic": false``). There is no synthetic mode,
and only the quantizer is built — no image encoder, so no development stand-in exists.

    torchrun --nproc_per_node=N scripts/train_tokenizer_ddp.py --latents L.pt --steps 500
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Make the script runnable directly (srun/torchrun) without `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from adarq_flow.config import get_preset
from adarq_flow.tokenizer import AdaptiveResidualQuantizer
from adarq_flow.utils.distributed import (
    all_reduce_mean,
    barrier,
    cleanup,
    get_world_size,
    is_main_process,
    seed_everything,
    setup_distributed,
)


def _load_real_latents(path: str, device) -> torch.Tensor:
    meta_path = pathlib.Path(path + ".meta.json")
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} missing: cannot prove {path} holds real latents")
    meta = json.loads(meta_path.read_text())
    if meta.get("synthetic") is not False:
        raise SystemExit(f"{path} is not marked as real latents (synthetic != false)")
    z = torch.load(path, map_location=device, weights_only=True)
    return z.reshape(-1, z.size(-1)) if z.dim() == 3 else z


def main():
    ap = argparse.ArgumentParser(description="Distributed check of the Stage-0 quantizer")
    ap.add_argument("--latents", required=True,
                    help="REAL CLIP patch latents [M, d] from extract_clip_latents.py")
    ap.add_argument("--preset", default="clip_b32")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--backend", default=None, help="override cfg.dist.backend")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = get_preset(args.preset)
    backend = args.backend or cfg.dist.backend
    info = setup_distributed(backend=backend, init_timeout_min=cfg.dist.init_timeout_min)
    seed_everything(args.seed, rank=info.rank, seed_per_rank=cfg.dist.seed_per_rank)

    latents = _load_real_latents(args.latents, info.device)
    dim = cfg.tokenizer.clip_dim
    if latents.size(-1) != dim:
        raise ValueError(f"latents dim {latents.size(-1)} != preset clip_dim {dim}")
    if is_main_process():
        print(f"[adarq-flow] world_size={info.world_size} backend={backend} "
              f"device={info.device} preset={args.preset} | real latents "
              f"{tuple(latents.shape)} from {args.latents}", flush=True)

    q = AdaptiveResidualQuantizer(cfg.tokenizer).to(info.device)
    q.train()
    gen = torch.Generator(device=info.device).manual_seed(args.seed + info.rank)
    for step in range(args.steps):
        idx = torch.randint(latents.shape[0], (args.batch_size,), generator=gen,
                            device=info.device)
        out = q(latents[idx], update_codebook=True)  # EMA all-reduces across ranks
        if is_main_process() and (step % 50 == 0 or step == args.steps - 1):
            stats = out.losses.item()
            print(f"  step {step:5d}  recon={stats['recon']:.4f}  "
                  f"rate={stats['rate']:.2f}", flush=True)

    barrier()

    # Every rank must hold the same codebooks: mean-reduce a copy and compare.
    max_dev = 0.0
    for book in q.codebooks:
        ref = book.embed.clone()
        all_reduce_mean(ref)          # == book.embed on every rank iff replicas agree
        max_dev = max(max_dev, (book.embed - ref).abs().max().item())
    if is_main_process():
        ok = "OK" if max_dev < 1e-5 else f"DIVERGED (max_dev={max_dev:.2e})"
        print(f"[adarq-flow] cross-rank codebook consistency: {ok}", flush=True)
    if get_world_size() > 1 and max_dev >= 1e-5:
        raise RuntimeError(f"codebooks diverged across ranks (max_dev={max_dev:.2e})")

    cleanup()


if __name__ == "__main__":
    main()
