"""Adaptive RVQ-CLIP tokenizer (Component A, top-level module).

Combines a (frozen) CLIP patch-latent encoder with the adaptive-depth residual
quantizer. The quantizer is the trainable Stage-0 object (its codebook learns by
EMA); the encoder is frozen.

Primary interfaces:
    z   = tok.encode(images)        # images -> CLIP patch latents [B, N, d]
    out = tok(z)                    # latents -> QuantizeOutput (codes/depths/zhat/res)
    out = tok.tokenize(images)      # encode + quantize (no codebook update by default)
    zhat = tok.dequantize(codes)    # codes -> dequantized prefix
"""

from __future__ import annotations

from torch import Tensor, nn

from ..config import AdaRQFlowConfig, TokenizerConfig
from .encoder import build_clip_encoder
from .quantizer import AdaptiveResidualQuantizer, QuantizeOutput


class RVQCLIPTokenizer(nn.Module):
    """Frozen CLIP encoder + adaptive residual quantizer."""

    def __init__(self, tok_cfg: TokenizerConfig, backbone: str = "dummy") -> None:
        super().__init__()
        self.cfg = tok_cfg
        self.encoder = build_clip_encoder(backbone, tok_cfg)
        self.quantizer = AdaptiveResidualQuantizer(tok_cfg)

    @property
    def codebook_vocab(self) -> int:
        """Code-classifier vocabulary size: ``K`` codes + 1 ``<halt>`` symbol."""
        return self.cfg.codebook_size + 1

    def encode(self, images: Tensor) -> Tensor:
        """Images ``[B, C, H, W]`` -> CLIP patch latents ``[B, N, d]`` (frozen)."""
        return self.encoder(images)

    def forward(self, z: Tensor, update_codebook: bool | None = None) -> QuantizeOutput:
        """Quantize precomputed latents. See ``AdaptiveResidualQuantizer.forward``."""
        return self.quantizer(z, update_codebook=update_codebook)

    def tokenize(self, images: Tensor, update_codebook: bool = False) -> QuantizeOutput:
        """Encode then quantize. Codebook is not updated unless explicitly requested."""
        z = self.encode(images)
        return self.quantizer(z, update_codebook=update_codebook)

    def dequantize(self, codes: Tensor) -> Tensor:
        """Code indices ``[M, D_max]`` -> dequantized prefix ``zhat`` ``[M, d]``."""
        return self.quantizer.dequantize(codes)

    def codebook_usage(self) -> Tensor:
        return self.quantizer.codebook_usage()


def build_tokenizer(cfg: AdaRQFlowConfig) -> RVQCLIPTokenizer:
    """Construct a tokenizer from a full :class:`AdaRQFlowConfig`.

    The CLIP encoder follows ``cfg.mllm.backbone`` (the tokenizer's latents are the
    MLLM's native CLIP latents); the quantizer follows ``cfg.tokenizer``.
    """
    return RVQCLIPTokenizer(cfg.tokenizer, backbone=cfg.mllm.backbone)
