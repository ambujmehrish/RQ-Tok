"""Evaluation metrics.

Two groups:

  * **Self-contained** (no pretrained nets / real data): MSE, PSNR, SSIM, codebook
    perplexity — runnable on CPU and in tests.
  * **External** (FID / sFID / IS / rFID / LPIPS / GenEval / DPG-Bench): require
    pretrained networks (Inception, LPIPS), an object detector, or real eval sets.
    These **raise** with guidance rather than return a fake/fallback number.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


# -- self-contained --------------------------------------------------------------------
def mse(pred: Tensor, target: Tensor) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    return (pred - target).pow(2).mean()


def psnr(pred: Tensor, target: Tensor, data_range: float | None = None) -> Tensor:
    m = mse(pred, target).clamp_min(1e-12)
    if data_range is None:
        data_range = float((target.max() - target.min()).clamp_min(1e-8))
    return 10.0 * torch.log10(torch.tensor(data_range ** 2, device=pred.device) / m)


def _gaussian_window(window_size: int, sigma: float, channels: int,
                     device: torch.device) -> Tensor:
    coords = torch.arange(window_size, device=device) - window_size // 2
    g = torch.exp(-(coords.float() ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = g[:, None] * g[None, :]
    return w2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: Tensor, target: Tensor, data_range: float | None = None,
         window_size: int = 7, sigma: float = 1.5) -> Tensor:
    """Windowed SSIM for image tensors ``[B, C, H, W]`` (mean over the batch)."""
    if pred.dim() != 4 or pred.shape != target.shape:
        raise ValueError("ssim expects matching [B, C, H, W] tensors")
    b, c, h, w = pred.shape
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")
    if window_size > min(h, w):
        # Silently shrinking the window changes the metric definition between arms.
        raise ValueError(
            f"ssim window_size={window_size} exceeds image size {h}x{w}; pass a smaller "
            "odd window explicitly rather than having it adjusted silently")
    ws = window_size
    if data_range is None:
        data_range = float((target.max() - target.min()).clamp_min(1e-8))
    window = _gaussian_window(ws, sigma, c, pred.device)
    pad = ws // 2

    def filt(x: Tensor) -> Tensor:
        return F.conv2d(x, window, padding=pad, groups=c)

    mu1, mu2 = filt(pred), filt(target)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = filt(pred * pred) - mu1_sq
    s2 = filt(target * target) - mu2_sq
    s12 = filt(pred * target) - mu12
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return ssim_map.mean()


def codebook_perplexity(cluster_size: Tensor) -> float:
    """``exp(H(p))`` of the normalized code-usage distribution (max == #codes)."""
    p = cluster_size.float().clamp_min(0)
    total = p.sum()
    if total <= 0:
        raise ValueError("cluster_size sums to zero; no usage recorded")
    p = p / total
    nz = p[p > 0]
    entropy = -(nz * nz.log()).sum()
    return float(math.exp(float(entropy)))


# -- external (require pretrained nets / real data) ------------------------------------
def _needs(name: str, requirement: str):
    raise NotImplementedError(
        f"{name} requires {requirement}. It cannot be computed on CPU/dummy data; "
        "wire it with the real models and eval sets (see DESIGN.md §7)."
    )


def fid(*_a, **_k):
    _needs("FID", "a pretrained InceptionV3 + real reference images")


def sfid(*_a, **_k):
    _needs("sFID", "a pretrained InceptionV3 (spatial features) + real reference images")


def inception_score(*_a, **_k):
    _needs("Inception Score", "a pretrained InceptionV3 classifier")


def rfid(*_a, **_k):
    _needs("rFID", "a pretrained InceptionV3 + reconstructed/real image pairs")


def lpips(*_a, **_k):
    _needs("LPIPS", "a pretrained LPIPS (AlexNet/VGG) network")


def geneval_score(*_a, **_k):
    _needs("GenEval", "the GenEval object detector + prompt suite")


def dpg_bench_score(*_a, **_k):
    _needs("DPG-Bench", "the DPG-Bench prompts + scoring model")
