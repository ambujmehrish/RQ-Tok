"""CLIP patch-latent encoders for the tokenizer.

The tokenizer quantizes **MLLM-native CLIP patch latents**. On real GPU configs
those latents come from the frozen MLLM's CLIP visual encoder (wired in Phase 2).
For CPU tests and end-to-end smoke runs we use a small, frozen, deterministic
``DummyCLIPEncoder`` selected explicitly via ``mllm.backbone == "dummy"``.

This is a *config-selected variant*, not a fallback: a non-dummy backbone raises
``NotImplementedError`` here rather than silently degrading to the dummy encoder.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import TokenizerConfig


class DummyCLIPEncoder(nn.Module):
    """Tiny frozen image -> patch-latent encoder for CPU runs.

    Maps an image ``[B, C, H, W]`` to patch latents ``[B, N, d]`` with a fixed
    (non-trainable) conv stem followed by an adaptive pool to a ``sqrt(N) x sqrt(N)``
    grid. Deterministic given the global RNG seed at construction time; frozen so it
    mimics a pretrained, non-updated CLIP encoder.
    """

    def __init__(self, clip_dim: int, num_patches: int, in_channels: int = 3) -> None:
        super().__init__()
        side = int(round(math.sqrt(num_patches)))
        if side * side != num_patches:
            raise ValueError(
                f"num_patches must be a perfect square for the dummy encoder, "
                f"got {num_patches}"
            )
        self.clip_dim = int(clip_dim)
        self.num_patches = int(num_patches)
        self.in_channels = int(in_channels)
        self.grid = side

        self.stem = nn.Conv2d(in_channels, clip_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((side, side))
        # Frozen: mimics a pretrained encoder that the tokenizer does not update.
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        if images.dim() != 4 or images.size(1) != self.in_channels:
            raise ValueError(
                f"expected images of shape [B, {self.in_channels}, H, W], "
                f"got {tuple(images.shape)}"
            )
        h = self.stem(images)                 # [B, d, H, W]
        h = self.pool(h)                      # [B, d, side, side]
        b, d, _, _ = h.shape
        z = h.reshape(b, d, self.num_patches).transpose(1, 2)  # [B, N, d]
        return z.contiguous()


def build_clip_encoder(backbone: str, tok_cfg: TokenizerConfig) -> nn.Module:
    """Construct the CLIP patch-latent encoder for the given MLLM backbone.

    ``backbone == "dummy"`` -> ``DummyCLIPEncoder``. Any other value raises, because
    the real CLIP encoder is loaded with the MLLM in Phase 2.
    """
    if backbone == "dummy":
        return DummyCLIPEncoder(tok_cfg.clip_dim, tok_cfg.num_patches)

    from ..mllm.qwen import QwenVisionEncoder, is_qwen_vl

    if is_qwen_vl(backbone):
        # The MLLM's OWN visual tower: patch latents in the space the LLM consumes.
        return QwenVisionEncoder(backbone, tok_cfg)
    raise NotImplementedError(
        f"no image encoder for backbone '{backbone}'. Supported: 'dummy' (CPU "
        "development) or a Qwen-VL checkpoint, whose visual tower supplies the "
        "MLLM-native patch latents. Refusing to substitute a stand-in."
    )
