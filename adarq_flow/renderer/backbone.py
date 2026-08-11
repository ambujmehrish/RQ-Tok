"""Flow-matching renderer backbone (Component C).

A rectified-flow DiT over image-latent tokens with time conditioning. On real
configs this is the **frozen FLUX.1-dev** transformer; here it is a tiny frozen DiT
development stand-in (replaced after the final development phase, per DESIGN.md §9).

The backbone is parameter-frozen but its forward is **grad-transparent**: ControlNet
residuals are injected inside it, so gradients must flow through to the (trainable)
ControlNet. No diffusion anywhere — this predicts a flow velocity.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import RendererConfig
from ..mllm.layers import SelfBlock, TimeEmbedding


class DummyFluxBackbone(nn.Module):
    """Frozen flow-matching DiT: ``(x_t, t[, control]) -> velocity``."""

    def __init__(self, latent_dim: int, model_dim: int, num_blocks: int,
                 num_heads: int, num_tokens: int) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.num_tokens = num_tokens

        self.in_proj = nn.Linear(latent_dim, model_dim)
        self.pos = nn.Embedding(num_tokens, model_dim)
        self.time = TimeEmbedding(model_dim)
        self.blocks = nn.ModuleList(SelfBlock(model_dim, num_heads) for _ in range(num_blocks))
        self.norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_dim)

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x_t: Tensor, t: Tensor,
                control_residuals: list[Tensor] | None = None) -> Tensor:
        if x_t.dim() != 3 or x_t.shape[1:] != (self.num_tokens, self.latent_dim):
            raise ValueError(
                f"expected x_t [B, {self.num_tokens}, {self.latent_dim}], "
                f"got {tuple(x_t.shape)}"
            )
        pos = self.pos(torch.arange(self.num_tokens, device=x_t.device)).unsqueeze(0)
        h = self.in_proj(x_t) + pos + self.time(t).unsqueeze(1)
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if control_residuals is not None and i < len(control_residuals):
                h = h + control_residuals[i]
        return self.out_proj(self.norm(h))


def build_renderer_backbone(cfg: RendererConfig) -> nn.Module:
    if cfg.backbone == "dummy":
        return DummyFluxBackbone(
            cfg.latent_dim, cfg.model_dim, cfg.num_blocks, cfg.num_heads,
            cfg.num_image_tokens,
        )
    from .flux import FluxRendererBackbone, is_flux

    if is_flux(cfg.backbone):
        return FluxRendererBackbone(cfg.backbone, cfg)
    raise NotImplementedError(
        f"no adapter for renderer backbone '{cfg.backbone}'. Supported: 'dummy' (CPU "
        "development) or a FLUX checkpoint. Refusing to substitute a stand-in."
    )
