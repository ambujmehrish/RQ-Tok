"""AdaRQ-Flow arm on upstream's Bifrost-1 MLLM: same backbone + branch, different bridge.

Everything up to the image-token hidden states ``h`` is upstream's own training forward
(``branch_hidden`` reproduces it exactly; a test checks that upstream's MSE loss is
recovered bit for bit). Only what reads ``h`` differs from the Bifrost-1 arm:

code head (discrete, sampled)   level k of patch i:
    logits_k = W_k LN( h_i + P(zhat_i^(k)) + e_k ),   zhat_i^(k) = sum_{j<k} C_j[q_ij]
    over K + 1 symbols (K = <halt>, forbidden at level 1); cross-entropy on masked patches,
    levels 1 .. min(D_i + 1, D_max) (each code, then the first <halt>).
residual head (continuous, sampled by flow matching)   s = residual RMS from Stage 0:
    rho = (z_i - zhat_i) / s,  rho_t = (1 - t) eps + t rho,  t ~ U(0, 1)
    L_flow = || u(rho_t, t, h_i + Q(zhat_i)) - (rho - eps) ||^2
Total L = L_code + lambda_flow L_flow (lambda_flow = 1). The codebooks are Stage 0's,
frozen. Upstream's linear ``vision_head`` is not used by this arm and is frozen.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from . import bifrost_build as bb


class TimeEmbed(nn.Module):
    def __init__(self, dim: int, freq: int = 256) -> None:
        super().__init__()
        self.freq = freq
        self.mlp = nn.Sequential(nn.Linear(freq, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * 1000.0 * f[None]
        return self.mlp(torch.cat([a.cos(), a.sin()], dim=-1))


class ResBlock(nn.Module):
    """AdaLN residual MLP block (as MAR's SimpleMLPAdaLN)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 3 * width))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.ada(c).chunk(3, dim=-1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class FlowHead(nn.Module):
    def __init__(self, dim: int, width: int = 1024, depth: int = 3) -> None:
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.cond = nn.Linear(dim, width)
        self.time = TimeEmbed(width)
        self.blocks = nn.ModuleList(ResBlock(width) for _ in range(depth))
        self.out_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(width, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t, t, cond):
        c = self.cond(cond) + self.time(t)
        x = self.inp(x_t)
        for b in self.blocks:
            x = b(x, c)
        return self.out(self.out_norm(x))


class AdaRQHeads(nn.Module):
    def __init__(self, dim: int, codebooks: torch.Tensor, res_rms: float,
                 flow_width: int = 1024, flow_depth: int = 3) -> None:
        super().__init__()
        depth, k, d = codebooks.shape
        if d != dim:
            raise ValueError(f"codebook width {d} != hidden width {dim}")
        self.K, self.D = k, depth
        self.register_buffer("books", codebooks.clone(), persistent=True)   # frozen Stage 0
        self.res_rms = float(res_rms)
        self.prefix = nn.Linear(dim, dim)
        self.level = nn.Parameter(torch.zeros(depth, dim))
        self.norm = nn.LayerNorm(dim)
        self.code_out = nn.ModuleList(nn.Linear(dim, k + 1) for _ in range(depth))
        self.res_cond = nn.Linear(dim, dim)
        self.flow = FlowHead(dim, flow_width, flow_depth)

    # -- codes ------------------------------------------------------------------------
    def prefixes(self, codes: torch.Tensor) -> torch.Tensor:
        """zhat^(k) for k = 0..D: [..., D + 1, d] (halted levels add nothing)."""
        live = codes != self.K
        parts = [torch.zeros(*codes.shape[:-1], self.books.shape[-1], device=codes.device,
                             dtype=self.books.dtype)]
        for lvl in range(self.D):
            c = codes[..., lvl].clamp_max(self.K - 1)
            parts.append(parts[-1] + self.books[lvl][c] * live[..., lvl, None])
        return torch.stack(parts, dim=-2)

    def code_logits(self, h: torch.Tensor, prefix: torch.Tensor, lvl: int,
                    forbid_halt: bool = True) -> torch.Tensor:
        x = self.norm(h + self.prefix(prefix.to(h.dtype)) + self.level[lvl])
        logits = self.code_out[lvl](x).float()
        if lvl == 0 and forbid_halt:
            logits[..., self.K] = float("-inf")          # level 1 never halts
        return logits

    # -- losses -----------------------------------------------------------------------
    def counts(self, codes: torch.Tensor, masked: torch.Tensor, flow_reps: int = 4,
               ) -> tuple[int, int]:
        """(code targets, flow elements) in a batch: the denominators of its mean losses,
        so a batch split into micro-batches optimises exactly the full-batch objective."""
        depth = (codes[masked] != self.K).sum(-1)
        n_ce = int(torch.clamp(depth + 1, max=self.D).sum())
        n_flow = int(masked.sum()) * flow_reps * self.books.shape[-1]
        return n_ce, n_flow

    def loss(self, h: torch.Tensor, z: torch.Tensor, codes: torch.Tensor,
             masked: torch.Tensor, flow_reps: int = 4,
             totals: tuple[int, int] | None = None) -> dict[str, torch.Tensor]:
        """h, z: [B, N, d]; codes: [B, N, D]; masked: [B, N] bool (positions to learn).
        ``totals`` (from ``counts`` on the whole batch) normalises the sums when this is one
        micro-batch of a larger batch; without it the means are over this call only."""
        h, z, codes = h[masked], z[masked].float(), codes[masked].long()
        pre = self.prefixes(codes)                                   # [M, D+1, d]
        depth = (codes != self.K).sum(-1)                            # [M]
        ce, n_ce = h.new_zeros((), dtype=torch.float32), 0
        for lvl in range(self.D):
            valid = lvl <= depth                    # codes 1..D_i, then the first <halt>
            valid = valid & (lvl < self.D)
            if not valid.any():
                continue
            logit = self.code_logits(h[valid], pre[valid, lvl], lvl)
            ce = ce + F.cross_entropy(logit, codes[valid, lvl], reduction="sum")
            n_ce += int(valid.sum())
        l_code = ce / (totals[0] if totals else max(n_ce, 1))

        zhat = pre[torch.arange(len(codes), device=codes.device), depth]
        rho = (z - zhat) / self.res_rms
        cond = h + self.res_cond(zhat.to(h.dtype))
        cond, rho = cond.repeat(flow_reps, 1), rho.repeat(flow_reps, 1)
        t = torch.rand(len(rho), device=rho.device)
        eps = torch.randn_like(rho)
        x_t = (1 - t[:, None]) * eps + t[:, None] * rho
        v = self.flow(x_t.to(h.dtype), t, cond).float()
        sq = (v - (rho - eps)).pow(2).sum()
        l_flow = sq / (totals[1] if totals else (v.numel()))
        return {"loss": l_code + l_flow, "code_ce": l_code.detach(), "flow": l_flow.detach()}

    # -- sampling ---------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, h_c, h_u=None, cfg_weight: float = 1.0, temperature: float = 1.0,
               flow_steps: int = 32, generator: torch.Generator | None = None):
        """One latent per row of h: codes depth by depth (guided logits), then the residual
        by Euler integration of the guided velocity. Returns (latent [M, d], depth [M])."""
        h_c = h_c.float()                       # heads run in fp32 (backbone may be bf16)
        h_u = None if h_u is None else h_u.float()
        m = h_c.shape[0]
        codes = torch.full((m, self.D), self.K, dtype=torch.long, device=h_c.device)
        alive = torch.ones(m, dtype=torch.bool, device=h_c.device)
        zhat = torch.zeros(m, self.books.shape[-1], device=h_c.device, dtype=self.books.dtype)
        for lvl in range(self.D):
            # guide the RAW logits, then forbid <halt> at level 1: masking first would give
            # -inf - (-inf) = NaN in the guidance arithmetic
            lg = self.code_logits(h_c, zhat, lvl, forbid_halt=False)
            if h_u is not None:
                lu = self.code_logits(h_u, zhat, lvl, forbid_halt=False)
                lg = lu + cfg_weight * (lg - lu)
            if lvl == 0:
                lg[..., self.K] = float("-inf")
            probs = torch.softmax(lg / temperature, dim=-1)
            q = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            q = torch.where(alive, q, torch.full_like(q, self.K))
            alive = alive & (q != self.K)                      # <halt> ends the patch
            codes[:, lvl] = q
            zhat = zhat + self.books[lvl][q.clamp_max(self.K - 1)] * alive[:, None]
        cond_c = h_c + self.res_cond(zhat.to(h_c.dtype))
        cond_u = None if h_u is None else h_u + self.res_cond(zhat.to(h_u.dtype))
        x = torch.randn(m, zhat.shape[-1], device=h_c.device, generator=generator)
        for i in range(flow_steps):
            t = torch.full((m,), i / flow_steps, device=h_c.device)
            v = self.flow(x.to(h_c.dtype), t, cond_c).float()
            if cond_u is not None:
                vu = self.flow(x.to(h_c.dtype), t, cond_u).float()
                v = vu + cfg_weight * (v - vu)
            x = x + v / flow_steps
        latent = zhat.float() + self.res_rms * x
        return latent, (codes != self.K).sum(-1)


def branch_hidden(model, flow: dict, batch_size: int) -> torch.Tensor:
    """Upstream ``MultiModalityCausalLM.forward`` up to the image-token hidden states, for
    precomputed latents (``image_clip_embs``) — the same inputs, embeddings, masks and
    backbone call, returning ``hidden_states[:, -N:]`` instead of applying the linear head.
    """
    from einops import rearrange

    n_tok = bb.NUM_TOKENS
    z = flow["image_clip_embs"].to(model.dtype)
    ar = flow["ar_mask"].unsqueeze(-1)
    z_emb = model.mask_token * ar + z * (1 - ar)
    z_emb = z_emb + model.learnable_pos_emb
    text = model.vision_language_model.model.embed_tokens(flow["input_ids"][:, :-n_tok])
    emb = torch.cat((text, z_emb), dim=1)
    pos = rearrange(flow["position_ids"], "bsz k c -> k bsz c")
    out = model.vision_language_model(inputs_embeds=emb.to(model.dtype), use_cache=None,
                                      attention_mask=flow["attention_mask"].to(model.dtype),
                                      past_key_values=None, return_dict=True,
                                      position_ids=pos,
                                      image_position_mask=flow["image_position_mask"],
                                      t_emb=None)
    return out["hidden_states"][:batch_size, -n_tok:]


def set_trainable_adarq(model, heads: AdaRQHeads) -> dict[str, int]:
    """As Bifrost-1 (branch + mask token + position embedding) but WITHOUT upstream's
    linear vision_head, which this arm does not use; plus the AdaRQ heads."""
    counts = bb.set_trainable(model)
    for p in model.vision_head.parameters():
        p.requires_grad_(False)
    n_head = sum(p.numel() for p in heads.parameters() if p.requires_grad)
    n_vh = sum(p.numel() for p in model.vision_head.parameters())
    return {"trainable": counts["trainable"] - n_vh + n_head,
            "frozen": counts["frozen"] + n_vh, "adarq_heads": n_head,
            "bifrost_linear_head": n_vh}
