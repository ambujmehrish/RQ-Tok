"""Flow-matching renderer = frozen backbone + trainable latent ControlNet.

Trained with the rectified-flow velocity loss on image latents, conditioned on the
dequantized + flow-refined latent ``ẑ + res`` (the exposure-bias-aware "dequantized
code space"). Sampling integrates the probability-flow ODE from noise to image latent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import AdaRQFlowConfig, RendererConfig
from ..mllm.flow import flow_matching_loss
from .backbone import build_renderer_backbone
from .controlnet import LatentControlNet


@dataclass
class RendererLoss:
    flow: Tensor

    def item(self) -> dict[str, float]:
        return {"flow": self.flow.detach().item()}


class FlowRenderer(nn.Module):
    """Conditional rectified-flow renderer over image-latent tokens."""

    def __init__(self, cfg: RendererConfig, control_dim: int,
                 num_control_tokens: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.L = cfg.num_image_tokens
        self.backbone = build_renderer_backbone(cfg)       # frozen, grad-transparent
        self.controlnet = LatentControlNet(control_dim, num_control_tokens, cfg)

    def velocity(self, x_t: Tensor, t: Tensor, control: Tensor) -> Tensor:
        residuals = self.controlnet(x_t, t, control)
        return self.backbone(x_t, t, control_residuals=residuals)

    def compute_loss(self, image_latents: Tensor, control: Tensor,
                     generator: torch.Generator | None = None) -> RendererLoss:
        """Rectified-flow velocity loss.

        Args:
            image_latents: target image latents ``[B, L, C]`` (real: FLUX VAE latents).
            control: conditioning latents ``[B, N, d]`` (= ``ẑ + res``).
        """
        if image_latents.shape[1:] != (self.L, self.latent_dim):
            raise ValueError(
                f"image_latents must be [B, {self.L}, {self.latent_dim}], "
                f"got {tuple(image_latents.shape)}"
            )
        flow = flow_matching_loss(
            lambda x_t, t: self.velocity(x_t, t, control), image_latents, generator
        )
        return RendererLoss(flow=flow)

    @torch.no_grad()
    def sample(self, control: Tensor, steps: int | None = None,
               generator: torch.Generator | None = None,
               x0: Tensor | None = None) -> Tensor:
        """Integrate the flow ODE from noise to image latents ``[B, L, C]``."""
        self.eval()
        steps = steps if steps is not None else self.cfg.num_inference_steps
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        b = control.shape[0]
        device = control.device
        if x0 is None:
            x = torch.empty(b, self.L, self.latent_dim, device=device).normal_(
                generator=generator)
        elif x0.shape != (b, self.L, self.latent_dim):
            raise ValueError(f"x0 shape {tuple(x0.shape)} != ({b}, {self.L}, {self.latent_dim})")
        else:
            x = x0
        dt = 1.0 / steps
        for s in range(steps):
            t = torch.full((b,), s * dt, device=device)
            x = x + dt * self.velocity(x, t, control)
        return x


def build_renderer(cfg: AdaRQFlowConfig) -> FlowRenderer:
    """Construct the renderer; conditioning geometry comes from the tokenizer config."""
    return FlowRenderer(
        cfg.renderer,
        control_dim=cfg.tokenizer.clip_dim,
        num_control_tokens=cfg.tokenizer.num_patches,
    )
