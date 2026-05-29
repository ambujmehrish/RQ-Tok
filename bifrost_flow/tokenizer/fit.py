"""Stage-0 tokenizer fitting: EMA codebook learning on a tensor of CLIP latents.

The adaptive RVQ codebook learns by EMA (no optimizer / gradient step needed for the
default threshold-halting config), so a Stage-0 "fit" is simply repeated quantization
passes with ``update_codebook=True`` over minibatches of latents. This gives a fully
runnable Stage-0 on CPU; the full data-driven training pipeline arrives in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .tokenizer import RVQCLIPTokenizer


@dataclass
class FitReport:
    steps: int
    loss_history: list[dict[str, float]]
    final_recon: float
    final_usage: float


def fit_tokenizer(
    tokenizer: RVQCLIPTokenizer,
    latents: Tensor,
    steps: int = 200,
    batch_size: int = 64,
    seed: int = 0,
    log_every: int = 50,
) -> FitReport:
    """Fit the tokenizer codebook on a fixed tensor of latents by EMA.

    Args:
        tokenizer: the tokenizer to fit (its codebook is updated in place).
        latents: ``[M, d]`` or ``[B, N, d]`` CLIP patch latents.
        steps: number of EMA minibatch passes.
        batch_size: patches per minibatch.
        seed: RNG seed for minibatch sampling (reproducible).
        log_every: record a loss snapshot every this many steps.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if latents.dim() == 3:
        latents = latents.reshape(-1, latents.size(-1))
    elif latents.dim() != 2:
        raise ValueError(
            f"latents must be [M, d] or [B, N, d], got {tuple(latents.shape)}"
        )
    m = latents.shape[0]
    if m == 0:
        raise ValueError("latents is empty")

    gen = torch.Generator(device=latents.device).manual_seed(seed)
    tokenizer.train()

    history: list[dict[str, float]] = []
    last: dict[str, float] = {}
    for step in range(steps):
        idx = torch.randint(m, (min(batch_size, m),), generator=gen, device=latents.device)
        batch = latents[idx]
        out = tokenizer(batch, update_codebook=True)
        last = out.losses.item()
        if step % log_every == 0 or step == steps - 1:
            snap = {"step": float(step), **last, "usage": float(tokenizer.codebook_usage())}
            history.append(snap)

    return FitReport(
        steps=steps,
        loss_history=history,
        final_recon=last["recon"],
        final_usage=float(tokenizer.codebook_usage()),
    )
