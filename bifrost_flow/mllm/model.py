"""Component B — MLLM + hybrid vision-generation branch (top-level model).

Ties the frozen backbone, the trainable branch, and the hybrid head together with:

  * :meth:`compute_loss` — MAR-style masked training: a random subset of patches is
    masked; the code head is trained with cross-entropy (codes + ``<halt>``) and the
    flow head with the flow-matching objective, both only on masked patches.
    Classifier-free guidance is trained via per-sample context dropout.
  * :meth:`generate` — MaskGIT-style iterative decoding: codes are unmasked over
    ``decode_steps`` rounds (most-confident first) with CFG on the logits, then the
    flow head fills the continuous residual by ODE sampling.

Fail-loud throughout; no silent fallbacks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import BifrostFlowConfig, MLLMConfig, TokenizerConfig
from .backbone import build_backbone
from .branch import VisionGenBranch
from .flow import flow_matching_loss, flow_sample
from .heads import CodeClassifierHead, FlowResidualHead


@dataclass
class BranchLosses:
    total: Tensor
    code_ce: Tensor
    flow: Tensor

    def item(self) -> dict[str, float]:
        return {"total": self.total.detach().item(), "code_ce": self.code_ce.detach().item(),
                "flow": self.flow.detach().item()}


@dataclass
class GenerateOutput:
    codes: Tensor      # [B, N, D_max] long; entries [0, K), K == <halt>
    depths: Tensor     # [B, N] long
    residual: Tensor   # [B, N, d]
    zhat: Tensor       # [B, N, d] dequantized prefix
    latents: Tensor    # [B, N, d] == zhat + residual (renderer input)


class VisionGenModel(nn.Module):
    def __init__(self, mllm_cfg: MLLMConfig, tok_cfg: TokenizerConfig,
                 backbone: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.mllm_cfg = mllm_cfg
        self.tok_cfg = tok_cfg
        self.K = tok_cfg.codebook_size
        self.D = tok_cfg.max_depth
        self.N = tok_cfg.num_patches
        self.d = tok_cfg.clip_dim
        self.vocab = self.K + 1  # codes + <halt>
        H = mllm_cfg.hidden_dim

        self.backbone = backbone if backbone is not None else build_backbone(mllm_cfg)
        self.latent_in = nn.Linear(self.d, H)
        self.pos = nn.Embedding(self.N, H)
        self.mask_token = nn.Parameter(torch.randn(H) * 0.02)
        self.null_context = nn.Parameter(torch.randn(1, 1, H) * 0.02)  # CFG unconditional
        self.branch = VisionGenBranch(H, mllm_cfg.num_layers, mllm_cfg.num_heads)
        self.code_head = CodeClassifierHead(H, self.D, self.vocab)
        self.flow_head = FlowResidualHead(
            self.d, H, mllm_cfg.flow_head_dim, mllm_cfg.flow_head_layers
        )

    # -- context / embeddings ----------------------------------------------------------
    def encode_context(self, text_ids: Tensor) -> Tensor:
        """Frozen backbone context hidden states ``[B, T, H]``."""
        return self.backbone(text_ids)

    def _patch_embed(self, zhat: Tensor, masked: Tensor) -> Tensor:
        """Img-G input embeddings: revealed patches embed their dequantized prefix,
        masked patches use the mask token; positional embedding added."""
        emb = self.latent_in(zhat)
        emb = torch.where(masked.unsqueeze(-1), self.mask_token.to(emb.dtype), emb)
        pos = self.pos(torch.arange(self.N, device=emb.device)).unsqueeze(0)
        return emb + pos

    def _apply_cfg_dropout(self, context: Tensor, drop_prob: float) -> Tensor:
        if drop_prob <= 0.0:
            return context
        b = context.shape[0]
        drop = torch.rand(b, device=context.device) < drop_prob
        null = self.null_context.expand_as(context)
        return torch.where(drop.view(b, 1, 1), null, context)

    # -- training ----------------------------------------------------------------------
    def compute_loss(self, text_ids: Tensor, codes: Tensor, residual: Tensor,
                     zhat: Tensor, generator: Optional[torch.Generator] = None) -> BranchLosses:
        """MAR masked training step.

        Args:
            text_ids: ``[B, T]`` context tokens.
            codes: ``[B, N, D_max]`` target codes (``K`` == ``<halt>``).
            residual: ``[B, N, d]`` target continuous residual.
            zhat: ``[B, N, d]`` teacher dequantized prefix.
        """
        self._check_targets(codes, residual, zhat)
        b = text_ids.shape[0]
        device = text_ids.device

        context = self.encode_context(text_ids)
        context = self._apply_cfg_dropout(context, self.mllm_cfg.cfg_text_dropout)

        masked = self._sample_mask(b, device, generator)        # [B, N] bool
        emb = self._patch_embed(zhat, masked)
        hidden = self.branch(emb, context)                       # [B, N, H]

        sel = masked.reshape(-1)
        if not bool(sel.any()):                                  # guarantee >=1 target
            sel[0] = True

        logits = self.code_head(hidden).reshape(-1, self.D, self.vocab)[sel]
        tgt = codes.reshape(-1, self.D)[sel]
        code_ce = F.cross_entropy(logits.reshape(-1, self.vocab), tgt.reshape(-1))

        h_sel = hidden.reshape(-1, hidden.shape[-1])[sel]
        zhat_sel = zhat.reshape(-1, self.d)[sel]
        res_sel = residual.reshape(-1, self.d)[sel]
        flow = flow_matching_loss(
            lambda x_t, t: self.flow_head(x_t, t, h_sel, zhat_sel), res_sel, generator
        )

        total = code_ce + flow
        return BranchLosses(total=total, code_ce=code_ce, flow=flow)

    def _sample_mask(self, b: int, device: torch.device,
                     generator: Optional[torch.Generator]) -> Tensor:
        lo, hi = self.mllm_cfg.mask_min, self.mllm_cfg.mask_max
        ratio = torch.empty(b, device=device).uniform_(lo, hi, generator=generator)
        n_mask = (ratio * self.N).ceil().clamp(1, self.N).long()    # >=1 masked
        noise = torch.rand(b, self.N, device=device, generator=generator)
        order = noise.argsort(dim=1)
        ranks = order.argsort(dim=1)
        return ranks < n_mask.unsqueeze(1)

    # -- decoding ----------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, text_ids: Tensor, dequantize_fn: Callable[[Tensor], Tensor],
                 steps: Optional[int] = None, cfg_scale: Optional[float] = None,
                 temperature: float = 1.0,
                 generator: Optional[torch.Generator] = None) -> GenerateOutput:
        """Iterative MaskGIT decode (codes) + flow sampling (residual).

        Args:
            dequantize_fn: maps codes ``[M, D_max]`` -> prefix ``[M, d]`` (the frozen
                tokenizer's ``dequantize``).
        """
        self.eval()
        steps = steps if steps is not None else self.mllm_cfg.decode_steps
        cfg_scale = cfg_scale if cfg_scale is not None else self.mllm_cfg.cfg_scale
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        b, device = text_ids.shape[0], text_ids.device

        cond = self.encode_context(text_ids)
        uncond = self.null_context.expand_as(cond)

        codes = torch.full((b, self.N, self.D), self.K, dtype=torch.long, device=device)
        revealed = torch.zeros(b, self.N, dtype=torch.bool, device=device)

        def deq(c: Tensor) -> Tensor:
            return dequantize_fn(c.reshape(-1, self.D)).reshape(b, self.N, self.d)

        for i in range(steps):
            zhat = deq(codes)
            emb = self._patch_embed(zhat, ~revealed)
            logits = self._cfg_logits(emb, cond, uncond, cfg_scale)   # [B,N,D,vocab]

            cand, conf = self._sample_codes(logits, temperature, generator)
            conf = conf.masked_fill(revealed, float("inf"))           # keep revealed

            n_keep_masked = self._schedule_masked(i, steps)
            n_reveal = self.N - n_keep_masked
            if i == steps - 1:
                n_reveal = self.N
            self._reveal(codes, revealed, cand, conf, n_reveal)

        # All patches revealed: final residual via flow sampling.
        zhat = deq(codes)
        emb = self._patch_embed(zhat, ~revealed)
        hidden = self.branch(emb, cond).reshape(-1, self.mllm_cfg.hidden_dim)
        zhat_flat = zhat.reshape(-1, self.d)
        res = flow_sample(
            lambda x_t, t: self.flow_head(x_t, t, hidden, zhat_flat),
            num=b * self.N, dim=self.d, steps=self.mllm_cfg.decode_steps,
            device=device, generator=generator,
        ).reshape(b, self.N, self.d)

        depths = (codes != self.K).sum(-1)
        return GenerateOutput(codes=codes, depths=depths, residual=res, zhat=zhat,
                              latents=zhat + res)

    def _cfg_logits(self, emb: Tensor, cond: Tensor, uncond: Tensor,
                    cfg_scale: float) -> Tensor:
        lc = self.code_head(self.branch(emb, cond))
        if cfg_scale == 1.0:
            return lc
        lu = self.code_head(self.branch(emb, uncond))
        return lu + cfg_scale * (lc - lu)

    def _sample_codes(self, logits: Tensor, temperature: float,
                      generator: Optional[torch.Generator]) -> tuple[Tensor, Tensor]:
        logp = F.log_softmax(logits, dim=-1)               # [B,N,D,vocab]
        if temperature <= 0.0:
            cand = logits.argmax(dim=-1)                    # greedy
        else:
            probs = F.softmax(logits / temperature, dim=-1).reshape(-1, self.vocab)
            cand = torch.multinomial(probs, 1, generator=generator).reshape(
                logits.shape[:-1])
        chosen = logp.gather(-1, cand.unsqueeze(-1)).squeeze(-1)   # [B,N,D]
        conf = chosen.mean(dim=-1)                                  # [B,N]
        return cand, conf

    def _schedule_masked(self, step: int, steps: int) -> int:
        frac = math.cos(0.5 * math.pi * (step + 1) / steps)        # 1 -> 0
        return int(math.floor(self.N * frac))

    def _reveal(self, codes: Tensor, revealed: Tensor, cand: Tensor,
                conf: Tensor, n_reveal: int) -> None:
        n_reveal = max(0, min(self.N, n_reveal))
        already = int(revealed[0].sum()) if revealed.numel() else 0
        k = n_reveal - already
        if k <= 0:
            return
        pick = conf.masked_fill(revealed, float("-inf")).topk(k, dim=1).indices  # [B,k]
        batch_idx = torch.arange(codes.shape[0], device=codes.device).unsqueeze(1)
        codes[batch_idx, pick] = cand[batch_idx, pick]
        revealed[batch_idx, pick] = True

    # -- validation --------------------------------------------------------------------
    def _check_targets(self, codes: Tensor, residual: Tensor, zhat: Tensor) -> None:
        if codes.shape[1:] != (self.N, self.D):
            raise ValueError(f"codes must be [B, {self.N}, {self.D}], got {tuple(codes.shape)}")
        if residual.shape[1:] != (self.N, self.d) or zhat.shape[1:] != (self.N, self.d):
            raise ValueError(
                f"residual/zhat must be [B, {self.N}, {self.d}], got "
                f"{tuple(residual.shape)} / {tuple(zhat.shape)}"
            )
        if int(codes.max()) > self.K or int(codes.min()) < 0:
            raise ValueError(f"codes out of range [0, {self.K}]")


def build_vision_gen_model(cfg: BifrostFlowConfig,
                           backbone: Optional[nn.Module] = None) -> VisionGenModel:
    """Construct the vision-generation model from a full config."""
    return VisionGenModel(cfg.mllm, cfg.tokenizer, backbone=backbone)
