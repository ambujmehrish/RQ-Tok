"""End-to-end inference pipeline (DESIGN.md §6).

    text -> frozen MLLM + gen branch
         -> MAR decode adaptive RVQ codes (CFG) -> dequantize ẑ
         -> flow-matching residual head -> res
         -> latent ControlNet(ẑ + res) -> flow ODE solve -> image latents

On real configs the final image latents are decoded to pixels by the FLUX VAE; the
dummy renderer returns the image latents directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import AdaRQFlowConfig
from ..mllm import build_vision_gen_model
from ..renderer import build_renderer
from ..tokenizer import build_tokenizer


@dataclass
class PipelineOutput:
    codes: Tensor          # [B, N, D_max] discrete RVQ codes (K == <halt>)
    depths: Tensor         # [B, N] adaptive per-patch depth
    latents: Tensor        # [B, N, d] bridge latents ẑ + res (renderer conditioning)
    image_latents: Tensor  # [B, L, C] renderer output (real: FLUX VAE latents)


class AdaRQFlowPipeline(nn.Module):
    """Bundles the frozen tokenizer, the vision-generation model, and the renderer."""

    def __init__(self, cfg: AdaRQFlowConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device or torch.device(cfg.train.device)
        self.tokenizer = build_tokenizer(cfg).to(self.device)
        self.model = build_vision_gen_model(cfg).to(self.device)
        self.renderer = build_renderer(cfg).to(self.device)
        self.eval()

    # -- checkpoint loading ------------------------------------------------------------
    def load_stage(self, path: str) -> str:
        """Load a Trainer checkpoint into the matching submodule (by its ``stage``)."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        target = {"tokenizer": self.tokenizer, "branch": self.model,
                  "renderer": self.renderer}.get(payload["stage"])
        if target is None:
            raise ValueError(f"unknown checkpoint stage {payload['stage']!r}")
        target.load_state_dict(payload["model"])
        return payload["stage"]

    # -- generation --------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, text_ids: Tensor, cfg_scale: float | None = None,
                 temperature: float = 1.0, decode_steps: int | None = None,
                 render_steps: int | None = None,
                 generator: torch.Generator | None = None) -> PipelineOutput:
        if text_ids.dim() != 2:
            raise ValueError(f"text_ids must be [B, T], got {tuple(text_ids.shape)}")
        text_ids = text_ids.to(self.device)

        gen = self.model.generate(
            text_ids, self.tokenizer.dequantize, steps=decode_steps,
            cfg_scale=cfg_scale, temperature=temperature, generator=generator)

        image_latents = self.renderer.sample(
            gen.latents, steps=render_steps, generator=generator)

        return PipelineOutput(codes=gen.codes, depths=gen.depths, latents=gen.latents,
                              image_latents=image_latents)


def build_pipeline(cfg: AdaRQFlowConfig,
                   device: torch.device | None = None) -> AdaRQFlowPipeline:
    return AdaRQFlowPipeline(cfg, device=device)
