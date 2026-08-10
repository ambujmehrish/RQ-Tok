"""Adaptive-depth residual vector quantizer (Component A core).

Residual-quantizes patch latents ``z in R^{N x d}`` with a shared (or per-depth)
codebook to a per-patch **adaptive depth**. A residual-norm halting rule lets flat
patches stop early while detailed patches spend more codes. Depth beyond a patch's
halt point is marked with a ``<halt>`` sentinel index ``K`` (so the downstream MLLM
code-classifier predicts a vocabulary of size ``K + 1``).

Design ref: DESIGN.md §4.1. No silent fallbacks — bad shapes/config raise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import TokenizerConfig
from .codebook import Codebook


@dataclass
class TokenizerLosses:
    """Stage-0 tokenizer losses (all scalar tensors).

    ``total`` is the config-weighted sum. ``recon`` is the discrete-prefix
    quantization error ``||z - zhat||^2``; ``commitment`` ties residuals to their
    chosen codes; ``entropy`` is the (minimized) usage-entropy penalty; ``rate`` is
    the mean per-patch depth (penalized to keep budgets honest).
    """

    total: Tensor
    recon: Tensor
    commitment: Tensor
    entropy: Tensor
    rate: Tensor

    def item(self) -> dict[str, float]:
        return {
            "total": self.total.detach().item(),
            "recon": self.recon.detach().item(),
            "commitment": self.commitment.detach().item(),
            "entropy": self.entropy.detach().item(),
            "rate": self.rate.detach().item(),
        }


@dataclass
class QuantizeOutput:
    """Result of quantizing a batch of latents.

    Shapes use ``M = B * N`` flattened patches (or ``M`` if called on flat input).
    """

    codes: Tensor          # [M, D_max] long; entries in [0, K), or K == <halt>
    depths: Tensor         # [M] long; number of real codes per patch (1..D_max)
    zhat: Tensor           # [M, d] dequantized prefix (sum of chosen codes)
    res: Tensor            # [M, d] continuous remainder z - zhat (flow-head target)
    losses: TokenizerLosses


class AdaptiveResidualQuantizer(nn.Module):
    """Adaptive-depth RVQ over a shared or per-depth EMA codebook."""

    HALT = -1  # sentinel meaning "no code"; exposed as index K in `codes`

    def __init__(self, cfg: TokenizerConfig, entropy_temp: float = 1.0) -> None:
        super().__init__()
        if cfg.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {cfg.max_depth}")
        if not 0.0 < cfg.halt_residual_threshold < 1.0:
            raise ValueError(
                "halt_residual_threshold must be in (0, 1), got "
                f"{cfg.halt_residual_threshold}"
            )
        if entropy_temp <= 0.0:
            raise ValueError(f"entropy_temp must be > 0, got {entropy_temp}")

        self.cfg = cfg
        self.dim = cfg.clip_dim
        self.max_depth = cfg.max_depth
        self.codebook_size = cfg.codebook_size
        self.shared = cfg.shared_codebook
        self.adaptive = cfg.adaptive_depth
        self.threshold = cfg.halt_residual_threshold
        self.entropy_temp = float(entropy_temp)

        n_books = 1 if self.shared else self.max_depth
        self.codebooks = nn.ModuleList(
            Codebook(cfg.codebook_size, cfg.clip_dim, ema_decay=cfg.ema_decay,
                     include_zero_code=cfg.include_zero_code,
                     dead_code_threshold=cfg.dead_code_threshold)
            for _ in range(n_books)
        )

    def _book(self, depth: int) -> Codebook:
        book = self.codebooks[0] if self.shared else self.codebooks[depth]
        assert isinstance(book, Codebook)  # ModuleList indexing widens to Module
        return book

    # ----------------------------------------------------------------------------------
    def _as_flat(self, z: Tensor) -> Tensor:
        if z.dim() == 3:
            z = z.reshape(-1, z.size(-1))
        elif z.dim() != 2:
            raise ValueError(f"expected z of shape [B,N,d] or [M,d], got {tuple(z.shape)}")
        if z.size(-1) != self.dim:
            raise ValueError(f"latent dim {z.size(-1)} != codebook dim {self.dim}")
        return z

    def forward(self, z: Tensor, update_codebook: bool | None = None) -> QuantizeOutput:
        """Quantize latents.

        Args:
            z: latents ``[B, N, d]`` or ``[M, d]``.
            update_codebook: run an EMA codebook update. Defaults to ``self.training``.
                Passing ``True`` in ``eval()`` or ``False`` in ``train()`` is honored
                exactly (no hidden behavior).
        """
        flat = self._as_flat(z)
        m = flat.shape[0]
        if m == 0:
            raise ValueError("received an empty latent batch")
        do_update = self.training if update_codebook is None else bool(update_codebook)

        eps = 1e-8
        z_norm = flat.norm(dim=-1).clamp_min(eps)            # [M]
        active = torch.ones(m, dtype=torch.bool, device=flat.device)
        codes = torch.full(
            (m, self.max_depth), self.codebook_size, dtype=torch.long, device=flat.device
        )  # default = <halt> sentinel (== K)
        zhat = torch.zeros_like(flat)
        r = flat.clone()

        # Collected (residual, index) pairs per codebook for EMA + losses.
        n_books = len(self.codebooks)
        collected: list[tuple[list[Tensor], list[Tensor]]] = [([], []) for _ in range(n_books)]

        for k in range(self.max_depth):
            book = self._book(k)
            book_i = 0 if self.shared else k
            idx, q = book.quantize(r)                          # [M], [M, d]
            # Straight-through: gradient (if any upstream) flows to r via q_st.
            q_st = r + (q - r).detach()

            mask = active.unsqueeze(-1)
            zhat = zhat + q_st * mask
            codes[:, k] = torch.where(active, idx, torch.full_like(idx, self.codebook_size))

            if active.any():
                a = active
                collected[book_i][0].append(r[a].detach())
                collected[book_i][1].append(idx[a])

            # Residual recursion (active patches only).
            r = r - q * mask

            if self.adaptive:
                frac = r.norm(dim=-1) / z_norm
                active = active & (frac >= self.threshold)
            else:
                # Non-adaptive: every patch uses all depths.
                pass
            if not active.any():
                break

        depths = (codes != self.codebook_size).sum(dim=-1)     # [M]
        res = flat - zhat

        losses = self._losses(flat, zhat, depths, collected)

        if do_update:
            self._ema_update(collected)

        return QuantizeOutput(codes=codes, depths=depths, zhat=zhat, res=res, losses=losses)

    # ----------------------------------------------------------------------------------
    def _ema_update(self, collected) -> None:
        for book_i, (res_list, idx_list) in enumerate(collected):
            if not res_list:
                continue
            flat = torch.cat(res_list, dim=0)
            idx = torch.cat(idx_list, dim=0)
            self._book(book_i).ema_update(flat, idx)

    def _losses(self, flat, zhat, depths, collected) -> TokenizerLosses:
        device = flat.device
        recon = (flat - zhat).pow(2).sum(-1).mean()

        commit_terms = []
        entropy_terms = []
        for book_i, (res_list, idx_list) in enumerate(collected):
            if not res_list:
                continue
            r = torch.cat(res_list, dim=0)               # [Mk, d]
            idx = torch.cat(idx_list, dim=0)             # [Mk]
            book = self._book(book_i)
            q = book.lookup(idx).detach()
            commit_terms.append((r - q).pow(2).sum(-1).mean())
            entropy_terms.append(self._entropy_loss(book, r))

        zero = torch.zeros((), device=device)
        commitment = torch.stack(commit_terms).mean() if commit_terms else zero
        entropy = torch.stack(entropy_terms).mean() if entropy_terms else zero
        rate = depths.float().mean()

        total = (
            recon
            + self.cfg.commitment_weight * commitment
            + self.cfg.entropy_weight * entropy
            + self.cfg.rate_penalty * rate
        )
        return TokenizerLosses(
            total=total, recon=recon, commitment=commitment, entropy=entropy, rate=rate
        )

    def _entropy_loss(self, book: Codebook, r: Tensor) -> Tensor:
        """Usage-entropy penalty (MaskGIT/VQGAN style).

        Minimizes per-sample assignment entropy (confident codes) while maximizing
        the batch-marginal code entropy (use the whole codebook). Returned value is
        ``sample_entropy - codebook_entropy`` (lower is better).
        """
        logits = -book.distances(r) / self.entropy_temp     # [Mk, K]
        probs = logits.softmax(dim=-1)
        log_probs = logits.log_softmax(dim=-1)
        sample_entropy = -(probs * log_probs).sum(-1).mean()
        avg = probs.mean(dim=0).clamp_min(1e-9)
        codebook_entropy = -(avg * avg.log()).sum()
        return sample_entropy - codebook_entropy

    # -- dequantization -----------------------------------------------------------------
    def dequantize(self, codes: Tensor) -> Tensor:
        """Reconstruct the dequantized prefix ``zhat`` from code indices.

        Args:
            codes: ``[M, D_max]`` long; entries in ``[0, K)`` are real codes, ``K``
                is the ``<halt>`` sentinel (contributes nothing).
        """
        if codes.dim() != 2 or codes.size(-1) != self.max_depth:
            raise ValueError(
                f"expected codes of shape [M, {self.max_depth}], got {tuple(codes.shape)}"
            )
        # RVQ prefixes are nested: <halt> must occupy a contiguous suffix. A code after a
        # halt would be applied to a residual that never passed through the skipped
        # level, silently producing a zhat outside the quantizer's reachable set.
        halted = (codes == self.codebook_size).cummax(dim=-1).values
        if bool((halted & (codes != self.codebook_size)).any()):
            bad = int((halted & (codes != self.codebook_size)).any(-1).sum())
            raise ValueError(
                f"{bad} code row(s) have a real code after a <halt>; <halt> must be a "
                "contiguous suffix. Sampling must be projected onto the valid set."
            )
        m = codes.shape[0]
        zhat = torch.zeros(m, self.dim, device=codes.device, dtype=self._book(0).embed.dtype)
        for k in range(self.max_depth):
            book = self._book(k)
            col = codes[:, k]
            live = col != self.codebook_size
            if live.any():
                zhat[live] = zhat[live] + book.lookup(col[live])
        return zhat

    def codebook_usage(self) -> Tensor:
        return torch.stack(
            [self._book(i).usage() for i in range(len(self.codebooks))]
        ).mean()
