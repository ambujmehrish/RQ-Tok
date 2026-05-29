"""The hybrid vision-generation head (DESIGN.md §4.2).

Two heads consume the branch hidden state per Img-G patch:

  * :class:`CodeClassifierHead` — predicts the adaptive RVQ code sequence (one
    softmax of size ``K + 1`` per residual level; class ``K`` is the ``<halt>``
    symbol). Cross-entropy training; enables temperature/top-p sampling and
    classifier-free guidance.
  * :class:`FlowResidualHead` — a flow-matching MLP that generates the continuous
    residual conditioned on the hidden state, the dequantized prefix, the flow
    time, and the noisy sample.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .layers import TimeEmbedding


class CodeClassifierHead(nn.Module):
    """Linear head -> ``[..., D_max, K+1]`` logits (per-level code classifier)."""

    def __init__(self, hidden_dim: int, max_depth: int, vocab: int) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.vocab = vocab  # K + 1 (codes + <halt>)
        self.proj = nn.Linear(hidden_dim, max_depth * vocab)

    def forward(self, h: Tensor) -> Tensor:
        logits = self.proj(h)
        return logits.reshape(*h.shape[:-1], self.max_depth, self.vocab)


class FlowResidualHead(nn.Module):
    """Flow-matching velocity field for the continuous residual.

    ``forward(x_t, t, h, zhat) -> v`` predicts the velocity at noisy residual
    ``x_t`` and time ``t``, conditioned on the branch hidden ``h`` and the
    dequantized prefix ``zhat``.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, width: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.latent_dim = latent_dim
        self.time = TimeEmbedding(width)
        in_dim = latent_dim + width + hidden_dim + latent_dim  # x_t, t_emb, h, zhat
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: Tensor, t: Tensor, h: Tensor, zhat: Tensor) -> Tensor:
        if x_t.shape[-1] != self.latent_dim:
            raise ValueError(f"x_t dim {x_t.shape[-1]} != latent_dim {self.latent_dim}")
        temb = self.time(t)
        inp = torch.cat([x_t, temb, h, zhat], dim=-1)
        return self.net(inp)
