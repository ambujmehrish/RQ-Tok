"""Transformer primitives shared by the frozen backbone and the trainable branch.

Attention uses ``F.scaled_dot_product_attention`` with an additive float mask
(0 = attend, -inf = block) so the token-type masking rules from DESIGN.md
Appendix-A are explicit and testable.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Additive ``[S, S]`` mask: position i may attend to j <= i."""
    full = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(full, diagonal=1)


class MHA(nn.Module):
    """Multi-head attention supporting separate query / key-value inputs."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, q_in: Tensor, kv_in: Tensor | None = None,
                attn_mask: Tensor | None = None) -> Tensor:
        if kv_in is None:
            kv_in = q_in
        b, sq, _ = q_in.shape
        sk = kv_in.shape[1]

        def split(x: Tensor, s: int, proj: nn.Linear) -> Tensor:
            return proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        q = split(q_in, sq, self.q)
        k = split(kv_in, sk, self.k)
        v = split(kv_in, sk, self.v)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, sq, -1)
        return self.o(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mult * dim)
        self.fc2 = nn.Linear(mult * dim, dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SelfBlock(nn.Module):
    """Pre-norm self-attention block (used by the frozen backbone)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = MHA(dim, num_heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.n1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.n2(x))
        return x


class BranchBlock(nn.Module):
    """Branch block: image-generation queries attend (bidirectionally) over the
    concatenation of the frozen context and themselves; the context is read-only
    key/value and is never updated by the branch.

    This realizes DESIGN.md's masking rule (Img-G bidirectional + attends to all
    token types; the frozen MLLM's context is not modified) without a hand-built
    block mask: queries are Img-G only, KV is ``[context ; Img-G]``, full attention.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nkv = nn.LayerNorm(dim)
        self.attn = MHA(dim, num_heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, imgg: Tensor, context: Tensor) -> Tensor:
        kv = torch.cat([context, imgg], dim=1)
        imgg = imgg + self.attn(self.nq(imgg), self.nkv(kv))
        imgg = imgg + self.mlp(self.n2(imgg))
        return imgg


class TimeEmbedding(nn.Module):
    """Sinusoidal + MLP embedding of a continuous flow time ``t in [0, 1]``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        ang = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if emb.shape[-1] < self.dim:  # odd dim padding
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)
