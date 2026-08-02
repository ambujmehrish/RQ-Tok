"""Tokenizer evaluation: reconstruction quality, adaptive depth, codebook usage.

The headline ablation we *can* run end-to-end (CPU + dummy) is **reconstruction vs.
adaptive depth** — the AdaRQ-Flow analog of Bifrost-1's token-count scaling curve
(Fig. 4): how reconstruction error falls as more residual codes are spent per patch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..tokenizer.tokenizer import RVQCLIPTokenizer
from .metrics import codebook_perplexity, mse, psnr


@dataclass
class TokenizerReport:
    recon_prefix_mse: float    # ||z - ẑ||^2  (discrete prefix only)
    recon_full_mse: float      # ||z - (ẑ + res)||^2
    prefix_psnr: float
    mean_depth: float
    max_depth: int
    codebook_usage: float
    codebook_perplexity: float
    depth_curve: list[tuple[int, float]]  # (depth d, MSE using first d codes)


@torch.no_grad()
def reconstruction_vs_depth(
    tokenizer: RVQCLIPTokenizer, latents: Tensor
) -> list[tuple[int, float]]:
    """MSE between ``z`` and the prefix dequantized using only the first ``d`` codes."""
    if latents.dim() == 3:
        latents = latents.reshape(-1, latents.size(-1))
    q = tokenizer.quantizer
    out = q(latents, update_codebook=False)
    D = q.max_depth
    curve: list[tuple[int, float]] = []
    for d in range(1, D + 1):
        codes_d = out.codes.clone()
        codes_d[:, d:] = q.codebook_size           # truncate: levels > d -> <halt>
        zhat_d = q.dequantize(codes_d)
        curve.append((d, float(mse(zhat_d, latents))))
    return curve


@torch.no_grad()
def evaluate_tokenizer(tokenizer: RVQCLIPTokenizer, latents: Tensor) -> TokenizerReport:
    if latents.dim() == 3:
        latents = latents.reshape(-1, latents.size(-1))
    q = tokenizer.quantizer
    out = q(latents, update_codebook=False)

    cluster_size = q._book(0).cluster_size
    return TokenizerReport(
        recon_prefix_mse=float(mse(out.zhat, latents)),
        recon_full_mse=float(mse(out.zhat + out.res, latents)),
        prefix_psnr=float(psnr(out.zhat, latents)),
        mean_depth=float(out.depths.float().mean()),
        max_depth=q.max_depth,
        codebook_usage=float(tokenizer.codebook_usage()),
        codebook_perplexity=codebook_perplexity(cluster_size),
        depth_curve=reconstruction_vs_depth(tokenizer, latents),
    )
