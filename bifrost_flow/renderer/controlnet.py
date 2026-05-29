"""Flow-matching latent ControlNet (Component C, trainable part).

Adapts the Bifrost-1 / FLUX ControlNet recipe: an input projection on the conditioning
latents, a 2D downsampling conv over the patch grid, then a few trainable DiT blocks
whose **zero-initialized** output projections inject residuals into the frozen renderer
backbone (so training starts as the identity and the frozen renderer is untouched at
init). Conditions on the dequantized + flow-refined latent ``ẑ + res``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import RendererConfig
from ..mllm.layers import BranchBlock, TimeEmbedding


class LatentControlNet(nn.Module):
    def __init__(self, control_dim: int, num_control_tokens: int,
                 cfg: RendererConfig) -> None:
        super().__init__()
        grid = int(round(math.sqrt(num_control_tokens)))
        if grid * grid != num_control_tokens:
            raise ValueError(
                f"num_control_tokens must be a perfect square, got {num_control_tokens}"
            )
        if grid % cfg.downsample_factor != 0:
            raise ValueError(
                f"control grid {grid} not divisible by downsample_factor "
                f"{cfg.downsample_factor}"
            )
        n_blocks = cfg.controlnet_double_blocks + cfg.controlnet_single_blocks
        if n_blocks < 1:
            raise ValueError("ControlNet needs >= 1 block")
        if n_blocks > cfg.num_blocks:
            raise ValueError(
                f"controlled blocks ({n_blocks}) exceed backbone num_blocks "
                f"({cfg.num_blocks})"
            )

        self.grid = grid
        self.model_dim = cfg.model_dim
        self.latent_dim = cfg.latent_dim
        self.num_image_tokens = cfg.num_image_tokens
        self.cond_scale = cfg.controlnet_conditioning_scale

        # Input projection (analog of FLUX ControlNet's 3xd -> d'xd input proj).
        self.input_proj = nn.Linear(control_dim, cfg.model_dim)
        self.downsample = nn.Conv2d(
            cfg.model_dim, cfg.model_dim,
            kernel_size=cfg.downsample_factor, stride=cfg.downsample_factor,
        )
        n_ctrl = (grid // cfg.downsample_factor) ** 2
        self.ctrl_pos = nn.Embedding(n_ctrl, cfg.model_dim)

        self.img_in = nn.Linear(cfg.latent_dim, cfg.model_dim)
        self.img_pos = nn.Embedding(cfg.num_image_tokens, cfg.model_dim)
        self.time = TimeEmbedding(cfg.model_dim)

        self.blocks = nn.ModuleList(BranchBlock(cfg.model_dim, cfg.num_heads)
                                    for _ in range(n_blocks))
        # Zero-init output projections: residuals start at 0 (identity init).
        self.zero_projs = nn.ModuleList(
            nn.Linear(cfg.model_dim, cfg.model_dim) for _ in range(n_blocks)
        )
        for zp in self.zero_projs:
            nn.init.zeros_(zp.weight)
            nn.init.zeros_(zp.bias)

    def forward(self, x_t: Tensor, t: Tensor, control: Tensor) -> list[Tensor]:
        b = x_t.shape[0]
        n = self.grid * self.grid
        if control.shape[1] != n:
            raise ValueError(
                f"control must have {n} tokens, got {control.shape[1]}"
            )
        # Conditioning tokens: project -> grid -> downsample -> flatten.
        c = self.input_proj(control)                          # [B, N, model_dim]
        c = c.transpose(1, 2).reshape(b, self.model_dim, self.grid, self.grid)
        c = self.downsample(c)                                # [B, model_dim, g', g']
        ctrl = c.flatten(2).transpose(1, 2)                   # [B, N', model_dim]
        ctrl = ctrl + self.ctrl_pos(
            torch.arange(ctrl.shape[1], device=x_t.device)).unsqueeze(0)

        img_pos = self.img_pos(
            torch.arange(self.num_image_tokens, device=x_t.device)).unsqueeze(0)
        h = self.img_in(x_t) + img_pos + self.time(t).unsqueeze(1)

        residuals: list[Tensor] = []
        for blk, zp in zip(self.blocks, self.zero_projs):
            h = blk(h, ctrl)                                   # cross-attend to control
            residuals.append(self.cond_scale * zp(h))         # zero-init -> 0 at start
        return residuals
