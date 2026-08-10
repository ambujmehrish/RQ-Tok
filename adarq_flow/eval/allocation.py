"""E0 — headroom pilot for the rate-allocation thesis (EXPERIMENTS.md §E0).

The central claim of AdaRQ-Flow is that bits should be allocated *non-uniformly* across
image regions. This module measures the **headroom** for that claim before any allocator
is built, by comparing allocation policies **at identical mean rate**:

    uniform   every patch gets the same depth
    random    per-patch depth drawn at random          <- negative control
    threshold the shipped rule (halt on relative residual norm); NOT rate-matched, so it
              is reported for reference and never used to decide a verdict
    oracle    rate-distortion optimal (Lagrangian sweep) — an upper bound on any allocator

If `oracle` cannot beat `uniform`, non-uniform allocation does not matter and the thesis
is false. If `random` matches `oracle`, then only *rate variance* matters, not
*content-awareness* — which also refutes the claim. Both verdicts are computed here
rather than left to interpretation.

Separability note
-----------------
Distortion is treated as **separable over patches**: total = Σ_p err(p, d_p). This makes
the whole sweep cost one quantization pass and is exact for latent-space distortion. It is
*not* exact for a real renderer, whose output depends on all patches jointly — replace
:func:`prefix_errors` with a downstream measurement to plug that in (at the cost of O(N·D)
renderer evaluations).

Interpreting the numbers
-----------------------
Headroom is **not** measured against zero. Even i.i.d. latents with no semantic structure
show non-zero oracle gain, because per-sample quantization error varies by chance. The
i.i.d. (`homogeneous`) regime is therefore the true null: only oracle gain *in excess of*
that baseline is evidence for content-aware allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..tokenizer.quantizer import AdaptiveResidualQuantizer


@dataclass
class PolicyResult:
    name: str
    mean_depth: float
    distortion: float


@dataclass
class HeadroomReport:
    """Result of the E0 pilot, with the pre-registered verdicts already applied."""

    mean_depth_budget: float
    policies: list[PolicyResult] = field(default_factory=list)
    oracle_gain_vs_uniform: float = 0.0   # relative reduction in distortion
    oracle_gain_vs_random: float = 0.0
    threshold_gain_vs_uniform: float = 0.0
    effect_floor: float = 0.10
    verdict: str = ""
    # True when the requested rate left no allocation freedom and an interior rate was
    # substituted. Surfaced because it means the reported budget is NOT the one asked for.
    budget_was_substituted: bool = False
    requested_mean_depth: float = 0.0

    def by_name(self, name: str) -> PolicyResult:
        for p in self.policies:
            if p.name == name:
                return p
        raise KeyError(name)

    def summary(self) -> str:
        lines = [f"E0 headroom pilot @ mean depth {self.mean_depth_budget:.2f}", ""]
        lines.append(f"{'policy':<12}{'mean depth':>12}{'distortion':>14}{'vs uniform':>12}"
                     f"  rate-matched")
        base = self.by_name("uniform").distortion
        for p in self.policies:
            rel = (base - p.distortion) / base if base > 0 else 0.0
            matched = abs(p.mean_depth - self.mean_depth_budget) < 1e-2
            flag = "yes" if matched else "NO (not comparable)"
            lines.append(f"{p.name:<12}{p.mean_depth:>12.3f}{p.distortion:>14.6f}"
                         f"{rel:>11.1%}  {flag}")
        if self.budget_was_substituted:
            lines += ["", f"  ! requested mean depth {self.requested_mean_depth:.2f} left no "
                          f"allocation freedom (all policies would coincide);",
                      f"    substituted {self.mean_depth_budget:.2f}. The reported budget is "
                      "NOT the one requested."]
        lines += ["", f"oracle vs uniform : {self.oracle_gain_vs_uniform:+.1%}",
                  f"oracle vs random  : {self.oracle_gain_vs_random:+.1%}",
                  f"threshold vs unif : {self.threshold_gain_vs_uniform:+.1%}",
                  "", f"VERDICT: {self.verdict}"]
        return "\n".join(lines)


# -- prefix distortion table -----------------------------------------------------------
@torch.no_grad()
def prefix_errors(quantizer: AdaptiveResidualQuantizer, latents: Tensor) -> Tensor:
    """Per-patch squared error for every prefix depth.

    Returns ``errs`` of shape ``[M, D_max + 1]`` where ``errs[p, d]`` is
    ``||z_p - sum_{k<=d} c_k||^2`` — the distortion if patch ``p`` is given depth ``d``.
    Computed in a single pass by accumulating the residual quantization greedily to full
    depth (halting disabled), which is exactly the nested-prefix structure of RVQ.
    """
    if latents.dim() == 3:
        latents = latents.reshape(-1, latents.size(-1))
    if latents.dim() != 2 or latents.size(-1) != quantizer.dim:
        raise ValueError(
            f"latents must be [M, {quantizer.dim}] or [B, N, d], got {tuple(latents.shape)}"
        )
    m = latents.shape[0]
    d_max = quantizer.max_depth

    errs = torch.empty(m, d_max + 1, device=latents.device)
    errs[:, 0] = latents.pow(2).sum(-1)          # depth 0 == transmit nothing
    zhat = torch.zeros_like(latents)
    r = latents.clone()
    for k in range(d_max):
        book = quantizer._book(k)
        _, q = book.quantize(r)
        zhat = zhat + q
        r = r - q
        errs[:, k + 1] = (latents - zhat).pow(2).sum(-1)
    return errs


def _distortion(errs: Tensor, depths: Tensor) -> float:
    """Mean per-patch distortion for an allocation. Out-of-range depths RAISE rather
    than being clamped: a silently repaired allocation is a different experiment."""
    d_max = errs.shape[1] - 1
    if int(depths.min()) < 0 or int(depths.max()) > d_max:
        raise ValueError(
            f"depths out of range [0, {d_max}]: "
            f"min={int(depths.min())}, max={int(depths.max())}")
    return float(errs.gather(1, depths.unsqueeze(1)).mean())


# -- allocation policies ---------------------------------------------------------------
def _validate_budget(total: int, m: int, d_max: int) -> None:
    if total < m:
        raise ValueError(
            f"budget {total} < {m} patches; every patch needs depth >= 1 "
            "(the tokenizer never emits depth 0)"
        )
    if total > m * d_max:
        raise ValueError(f"budget {total} exceeds capacity {m * d_max} (M x D_max)")


def allocate_uniform(m: int, total: int, d_max: int, device: torch.device) -> Tensor:
    """Every patch the same depth; the remainder is spread over the first patches so the
    budget is met **exactly** (mean-rate matching is the key control in E0)."""
    _validate_budget(total, m, d_max)
    base, rem = divmod(total, m)
    depths = torch.full((m,), base, dtype=torch.long, device=device)
    if rem:
        depths[:rem] += 1
    return depths.clamp(1, d_max)


def allocate_random(m: int, total: int, d_max: int, device: torch.device,
                    generator: torch.Generator | None = None) -> Tensor:
    """Random per-patch depths summing to exactly ``total``.

    The negative control: same rate, same rate *variance* as a non-uniform policy, but
    zero content information. If this matches the oracle, content-awareness is worthless.
    """
    _validate_budget(total, m, d_max)
    depths = torch.ones(m, dtype=torch.long, device=device)
    remaining = total - m
    # Distribute the remaining codes uniformly at random over patches with headroom.
    while remaining > 0:
        headroom = (depths < d_max).nonzero(as_tuple=True)[0]
        if headroom.numel() == 0:
            raise RuntimeError(
                f"cannot place {remaining} more codes: every patch is at D_max. "
                "The budget would be silently under-spent, breaking rate matching."
            )
        take = min(int(remaining), int(headroom.numel()))
        pick = headroom[torch.randperm(headroom.numel(), generator=generator,
                                       device=device)[:take]]
        depths[pick] += 1
        remaining -= take
    return depths


def allocate_oracle(errs: Tensor, total: int, d_max: int,
                    lam_hi: float = 1e12, iters: int = 200) -> Tensor:
    """Rate–distortion optimal allocation by Lagrangian sweep (Shoham–Gersho).

    For a multiplier ``λ`` every patch independently picks
    ``d*(p) = argmin_d err[p, d] + λ·d``; the total rate is non-increasing in ``λ``, so
    bisection finds the ``λ`` matching the budget. Any integrality slack is then closed
    by taking/undoing the individually best/worst increments so the budget is met
    **exactly** — mean-rate matching is the whole point of E0.

    This is preferred over sorting all increments once: RVQ marginal returns are *not*
    guaranteed non-increasing in depth (see :func:`marginal_returns_are_monotone`), and a
    single global sort silently drops increments whose predecessor was not yet taken,
    which underspends the budget and can make the "oracle" worse than uniform.

    It solves the convexified problem, so it is a strong optimum and an upper bound for
    any practical allocator; with non-convex per-patch RD curves it may be marginally
    below the exact integer optimum.
    """
    m = errs.shape[0]
    _validate_budget(total, m, d_max)
    device = errs.device
    depths_grid = torch.arange(1, d_max + 1, device=device, dtype=errs.dtype)
    usable = errs[:, 1:]                                    # depth 1..D_max

    def alloc_for(lam: float) -> Tensor:
        cost = usable + lam * depths_grid.unsqueeze(0)      # [M, D_max]
        return cost.argmin(dim=1) + 1                       # depths in [1, D_max]

    lo, hi = 0.0, lam_hi
    depths = alloc_for(lo)
    if int(depths.sum()) <= total:                          # λ=0 already within budget
        pass
    else:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if int(alloc_for(mid).sum()) > total:
                lo = mid
            else:
                hi = mid
        depths = alloc_for(hi)

    # Close integrality slack exactly.
    gains = errs[:, :-1] - errs[:, 1:]                       # gain of depth k -> k+1
    deficit = total - int(depths.sum())
    while deficit > 0:
        cand = depths < d_max
        if not bool(cand.any()):
            raise RuntimeError(
                f"cannot add {deficit} codes: all patches at D_max (rate not matched)")
        g = torch.where(cand, gains.gather(1, (depths - 1).unsqueeze(1)).squeeze(1),
                        torch.full((m,), -float("inf"), device=device))
        depths[int(g.argmax())] += 1
        deficit -= 1
    while deficit < 0:
        cand = depths > 1
        if not bool(cand.any()):
            raise RuntimeError(
                f"cannot remove {-deficit} codes: all patches at depth 1 (rate not matched)")
        g = torch.where(cand, gains.gather(1, (depths - 2).unsqueeze(1)).squeeze(1),
                        torch.full((m,), float("inf"), device=device))
        depths[int(g.argmin())] -= 1
        deficit += 1
    return depths.clamp(1, d_max)


def allocate_threshold(quantizer: AdaptiveResidualQuantizer, latents: Tensor) -> Tensor:
    """The depths the *shipped* halting rule actually produces (its natural rate)."""
    if latents.dim() == 3:
        latents = latents.reshape(-1, latents.size(-1))
    out = quantizer(latents, update_codebook=False)
    return out.depths


def marginal_returns_are_monotone(errs: Tensor) -> bool:
    """Whether per-patch marginal gain is non-increasing in depth (greedy optimality)."""
    gains = errs[:, :-1] - errs[:, 1:]
    return bool((gains[:, 1:] <= gains[:, :-1] + 1e-6).all())


# -- the pilot -------------------------------------------------------------------------
@torch.no_grad()
def headroom_pilot(quantizer: AdaptiveResidualQuantizer, latents: Tensor,
                   mean_depth: float | None = None, effect_floor: float = 0.10,
                   seed: int = 0, real_data: bool = False) -> HeadroomReport:
    """Run E0 and apply the pre-registered verdicts.

    Args:
        mean_depth: the rate all policies are held to. Defaults to the rate the shipped
            threshold rule naturally chooses, which is the fairest comparison point.
        effect_floor: R1 minimum relative effect (default 10%).
    """
    if latents.dim() == 3:
        latents = latents.reshape(-1, latents.size(-1))
    m = latents.shape[0]
    d_max = quantizer.max_depth
    device = latents.device

    errs = prefix_errors(quantizer, latents)
    thr_depths = allocate_threshold(quantizer, latents)
    if mean_depth is None:
        mean_depth = float(thr_depths.float().mean())
    total = int(round(mean_depth * m))
    total = max(m, min(total, m * d_max))

    # A budget pinned at either extreme leaves NO allocation freedom (every patch is
    # forced to depth 1, or to D_max), so all policies coincide and the pilot is
    # vacuous. Fall back to an interior rate rather than report a meaningless verdict.
    requested = total / m
    degenerate = total in (m, m * d_max)
    if degenerate:
        total = int(round(m * (1.0 + d_max) / 2.0))

    gen = torch.Generator(device=device).manual_seed(seed)
    policies = {
        "uniform": allocate_uniform(m, total, d_max, device),
        "random": allocate_random(m, total, d_max, device, generator=gen),
        "threshold": thr_depths,
        "oracle": allocate_oracle(errs, total, d_max),
    }

    results = [
        PolicyResult(name=n, mean_depth=float(d.float().mean()),
                     distortion=_distortion(errs, d))
        for n, d in policies.items()
    ]
    rep = HeadroomReport(mean_depth_budget=total / m, policies=results,
                         effect_floor=effect_floor,
                         budget_was_substituted=degenerate,
                         requested_mean_depth=requested)

    u = rep.by_name("uniform").distortion
    o = rep.by_name("oracle").distortion
    # Structural invariant: at equal rate the RD-optimal allocation can always fall back
    # to the uniform one, so it can never be worse. If it is, the instrument is broken
    # and no verdict it produces is trustworthy.
    if o > u * (1.0 + 1e-6):
        raise RuntimeError(
            f"oracle distortion ({o:.6f}) exceeds uniform ({u:.6f}) at the same rate — "
            "this is impossible for a correct allocator; the measurement is broken."
        )
    r = rep.by_name("random").distortion
    t = rep.by_name("threshold").distortion
    rep.oracle_gain_vs_uniform = (u - o) / u if u > 0 else 0.0
    rep.oracle_gain_vs_random = (r - o) / r if r > 0 else 0.0
    rep.threshold_gain_vs_uniform = (u - t) / u if u > 0 else 0.0

    # Pre-registered verdicts (EXPERIMENTS.md E0).
    if rep.oracle_gain_vs_uniform < effect_floor:
        if real_data:
            rep.verdict = (
                f"THESIS REFUTED — oracle beats uniform by only "
                f"{rep.oracle_gain_vs_uniform:.1%} (< {effect_floor:.0%} floor) on real "
                "latents. Non-uniform allocation does not matter for this bridge; do not "
                "proceed to C3."
            )
        else:
            rep.verdict = (
                f"BELOW FLOOR on SYNTHETIC data ({rep.oracle_gain_vs_uniform:.1%} < "
                f"{effect_floor:.0%}) — NOT a refutation. Synthetic latents only validate "
                "the instrument; and this measures LATENT-L2 distortion, which "
                "NOVELTY.md 6 argues is the wrong criterion. Re-run with real CLIP "
                "latents and downstream distortion before drawing any conclusion."
            )
    elif rep.oracle_gain_vs_random < effect_floor:
        rep.verdict = (
            f"CONTENT-AWARENESS REFUTED — oracle beats random by only "
            f"{rep.oracle_gain_vs_random:.1%}. The gain comes from rate VARIANCE, not "
            "from content-aware allocation; the adaptive claim does not stand."
        )
    else:
        rep.verdict = (
            f"PROCEED — headroom {rep.oracle_gain_vs_uniform:.1%} over uniform and "
            f"{rep.oracle_gain_vs_random:.1%} over random. The shipped threshold rule "
            f"captures {rep.threshold_gain_vs_uniform:.1%}, i.e. "
            f"{rep.threshold_gain_vs_uniform / max(rep.oracle_gain_vs_uniform, 1e-9):.0%} "
            "of the available headroom."
        )
    return rep
