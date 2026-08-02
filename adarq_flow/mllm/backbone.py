"""MLLM backbones for the vision-generation branch.

The backbone is **frozen** (understanding preserved) and produces the context
hidden states that the trainable branch attends to. ``DummyMLLMBackbone`` is a
tiny CPU transformer selected via ``mllm.backbone == "dummy"``. The real
Qwen2.5-VL adapter is a config-selected variant; requesting it raises (no silent
fallback) until the GPU integration lands.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import MLLMConfig
from .layers import SelfBlock, causal_mask


class DummyMLLMBackbone(nn.Module):
    """Tiny frozen causal transformer over text token ids -> context hidden states."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int,
                 vocab_size: int = 256, max_tokens: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos = nn.Embedding(max_tokens, hidden_dim)
        self.blocks = nn.ModuleList(SelfBlock(hidden_dim, num_heads) for _ in range(num_layers))
        self.norm = nn.LayerNorm(hidden_dim)

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.dim() != 2:
            raise ValueError(f"expected token_ids [B, T], got {tuple(token_ids.shape)}")
        b, t = token_ids.shape
        if t > self.max_tokens:
            raise ValueError(f"sequence length {t} exceeds max_tokens {self.max_tokens}")
        if int(token_ids.max()) >= self.vocab_size or int(token_ids.min()) < 0:
            raise ValueError(f"token id out of range [0, {self.vocab_size})")
        pos = torch.arange(t, device=token_ids.device)
        x = self.embed(token_ids) + self.pos(pos).unsqueeze(0)
        mask = causal_mask(t, token_ids.device)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        return self.norm(x)


def build_backbone(cfg: MLLMConfig) -> nn.Module:
    """Construct the (frozen) MLLM backbone for ``cfg.backbone``."""
    if cfg.backbone == "dummy":
        return DummyMLLMBackbone(cfg.hidden_dim, cfg.num_layers, cfg.num_heads)
    raise NotImplementedError(
        f"real MLLM backbone '{cfg.backbone}' is not yet wired. The frozen "
        "Qwen2.5-VL adapter (load via transformers, freeze, expose hidden states + "
        "init the branch from its decoder layers) is GPU-only and lands as a focused "
        "follow-up. For CPU runs set mllm.backbone='dummy'."
    )
