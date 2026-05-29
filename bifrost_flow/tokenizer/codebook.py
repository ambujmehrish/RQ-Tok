"""EMA vector-quantization codebook used by the adaptive residual quantizer.

A single shared codebook of ``K`` vectors in ``R^d``. Learning is by exponential
moving average (EMA) of assigned vectors — no gradient flows into the codebook —
plus two standard anti-collapse mechanisms:

  * **first-batch data init** — on the very first EMA update the code vectors are
    seeded from real input vectors, so codes start on the data manifold;
  * **dead-code reinit** — codes whose usage falls below a threshold are resampled
    from the current batch.

There are deliberately **no silent fallbacks**: shape/dtype/device mismatches and
misuse raise immediately.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Codebook(nn.Module):
    """A single EMA-updated VQ codebook.

    Args:
        codebook_size: number of code vectors ``K``.
        dim: code vector dimensionality ``d``.
        ema_decay: EMA decay for cluster statistics (``0 < decay < 1``).
        eps: Laplace-smoothing / numerical epsilon.
        dead_code_threshold: a code with EMA cluster size below this is considered
            dead and is reinitialized from the current batch.
    """

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: float = 1e-2,
    ) -> None:
        super().__init__()
        if codebook_size < 1:
            raise ValueError(f"codebook_size must be >= 1, got {codebook_size}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")

        self.codebook_size = int(codebook_size)
        self.dim = int(dim)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        self.dead_code_threshold = float(dead_code_threshold)

        # Code vectors. Not an nn.Parameter: the codebook learns via EMA, not grad.
        embed = torch.randn(self.codebook_size, self.dim)
        self.register_buffer("embed", embed)
        # EMA accumulators.
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("embed_avg", embed.clone())
        # Whether the first-batch data init has happened (0/1 flag as a buffer).
        self.register_buffer("initted", torch.zeros((), dtype=torch.bool))

    # -- queries ----------------------------------------------------------------------
    def _check(self, x: Tensor) -> None:
        if x.dim() != 2 or x.size(-1) != self.dim:
            raise ValueError(
                f"expected input of shape [M, {self.dim}], got {tuple(x.shape)}"
            )
        if x.device != self.embed.device:
            raise ValueError(
                f"input device {x.device} != codebook device {self.embed.device}"
            )

    def distances(self, x: Tensor) -> Tensor:
        """Squared L2 distances ``[M, K]`` from each input to each code."""
        self._check(x)
        # ||x||^2 - 2 x·C + ||C||^2 ; clamp tiny negatives from round-off.
        x_sq = x.pow(2).sum(-1, keepdim=True)              # [M, 1]
        c_sq = self.embed.pow(2).sum(-1)                   # [K]
        cross = x @ self.embed.t()                         # [M, K]
        return (x_sq - 2.0 * cross + c_sq).clamp_min(0.0)

    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Nearest-code assignment. Returns ``(indices [M], quantized [M, d])``."""
        idx = self.distances(x).argmin(dim=-1)
        return idx, self.embed[idx]

    def lookup(self, idx: Tensor) -> Tensor:
        """Map code indices to code vectors. Indices must be in ``[0, K)``."""
        if idx.numel() and (idx.min() < 0 or idx.max() >= self.codebook_size):
            raise ValueError(
                f"code index out of range [0, {self.codebook_size}): "
                f"min={int(idx.min())}, max={int(idx.max())}"
            )
        return self.embed[idx]

    # -- EMA learning -------------------------------------------------------------------
    @torch.no_grad()
    def ema_update(self, flat: Tensor, idx: Tensor) -> None:
        """Update code vectors from assigned inputs by EMA.

        Args:
            flat: assigned input vectors ``[M, d]``.
            idx: their code indices ``[M]`` in ``[0, K)``.
        """
        self._check(flat)
        if idx.shape[0] != flat.shape[0]:
            raise ValueError(
                f"idx/flat length mismatch: {idx.shape[0]} vs {flat.shape[0]}"
            )
        if flat.shape[0] == 0:
            raise ValueError("ema_update received an empty batch")

        # First-batch data init: seed codes from real vectors (sampled with repeat
        # if the batch is smaller than K). Deterministic given the global RNG seed.
        if not bool(self.initted):
            m = flat.shape[0]
            perm = torch.randint(m, (self.codebook_size,), device=flat.device)
            seed = flat[perm]
            self.embed.copy_(seed)
            self.embed_avg.copy_(seed)
            self.cluster_size.fill_(1.0)
            self.initted.fill_(True)

        onehot = torch.zeros(
            flat.shape[0], self.codebook_size, device=flat.device, dtype=flat.dtype
        )
        onehot.scatter_(1, idx.unsqueeze(1), 1.0)

        batch_count = onehot.sum(0)                 # [K]
        batch_sum = onehot.t() @ flat               # [K, d]

        d = self.ema_decay
        self.cluster_size.mul_(d).add_(batch_count, alpha=1.0 - d)
        self.embed_avg.mul_(d).add_(batch_sum, alpha=1.0 - d)

        # Laplace smoothing keeps unused codes from collapsing to zero.
        n = self.cluster_size.sum()
        smoothed = (
            (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
        )
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))

        self._reinit_dead_codes(flat)

    @torch.no_grad()
    def _reinit_dead_codes(self, flat: Tensor) -> None:
        dead = self.cluster_size < self.dead_code_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return
        perm = torch.randint(flat.shape[0], (n_dead,), device=flat.device)
        resampled = flat[perm]
        self.embed[dead] = resampled
        self.embed_avg[dead] = resampled
        self.cluster_size[dead] = 1.0

    # -- diagnostics --------------------------------------------------------------------
    def usage(self) -> Tensor:
        """Fraction of codes currently considered live (cluster size above threshold)."""
        return (self.cluster_size >= self.dead_code_threshold).float().mean()
