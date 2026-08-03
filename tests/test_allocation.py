"""E0 allocation-pilot tests.

These guard the invariants whose violation produced a false "thesis refuted" verdict on
the first implementation: the oracle underspent its budget and came out *worse* than
uniform. A measurement instrument that can silently invert its own result is more
dangerous than a missing one.
"""

import pytest

torch = pytest.importorskip("torch")

from adarq_flow.config import TokenizerConfig
from adarq_flow.eval import (
    allocate_oracle,
    allocate_random,
    allocate_uniform,
    headroom_pilot,
    prefix_errors,
)
from adarq_flow.tokenizer import AdaptiveResidualQuantizer


def _quantizer(**kw):
    base = dict(clip_dim=8, num_patches=4, codebook_size=32, max_depth=4)
    base.update(kw)
    return AdaptiveResidualQuantizer(TokenizerConfig(**base))


def _fit(q, latents, steps=60):
    for _ in range(steps):
        q(latents, update_codebook=True)
    return q


# -- prefix error table ----------------------------------------------------------------
def test_prefix_errors_shape():
    torch.manual_seed(0)
    q = _quantizer()
    z = torch.randn(64, 8)
    errs = prefix_errors(q, z)
    assert errs.shape == (64, q.max_depth + 1)
    assert torch.allclose(errs[:, 0], z.pow(2).sum(-1), atol=1e-5)


def test_depth_is_monotone_only_with_zero_code():
    """RVQ is NOT monotone in depth unless a zero codeword exists.

    Without one, an extra code overshoots a residual smaller than the nearest codeword
    and INCREASES error. Adaptive depth would then be partly rewarded for avoiding the
    quantizer's own self-harm rather than for content-aware allocation.
    """
    z = torch.randn(512, 8)

    def harmed(zero_code):
        torch.manual_seed(0)
        q = _quantizer(include_zero_code=zero_code)
        _fit(q, z, steps=120)
        e = prefix_errors(q, z)
        return float((e[:, 1:] > e[:, :-1] + 1e-6).any(-1).float().mean())

    assert harmed(True) == 0.0, "a zero codeword must make depth monotone by construction"
    assert harmed(False) > 0.0, "without a zero code, deeper RVQ can increase error"


# -- budgets are met exactly (the mean-rate control) -----------------------------------
@pytest.mark.parametrize("total", [64, 96, 128, 200, 256])
def test_all_policies_hit_budget_exactly(total):
    torch.manual_seed(0)
    q = _quantizer()
    z = torch.randn(64, 8)
    errs = prefix_errors(q, z)
    m, d_max = 64, q.max_depth
    dev = z.device
    for depths in (allocate_uniform(m, total, d_max, dev),
                   allocate_random(m, total, d_max, dev),
                   allocate_oracle(errs, total, d_max)):
        assert int(depths.sum()) == total
        assert int(depths.min()) >= 1 and int(depths.max()) <= d_max


def test_budget_validation_raises():
    q = _quantizer()
    with pytest.raises(ValueError):
        allocate_uniform(64, 32, q.max_depth, torch.device("cpu"))    # below 1/patch
    with pytest.raises(ValueError):
        allocate_uniform(64, 64 * q.max_depth + 1, q.max_depth, torch.device("cpu"))


# -- the invariant that caught the bug -------------------------------------------------
@pytest.mark.parametrize("total", [96, 128, 160, 200])
def test_oracle_never_worse_than_uniform_or_random(total):
    """At equal rate the RD-optimal allocation can always fall back to uniform."""
    torch.manual_seed(0)
    q = _quantizer()
    z = torch.randn(64, 8) * torch.rand(64, 1) * 4      # heterogeneous scales
    _fit(q, z)
    errs = prefix_errors(q, z)
    m, d_max, dev = 64, q.max_depth, z.device

    def dist(d):
        return float(errs.gather(1, d.unsqueeze(1)).mean())

    o = dist(allocate_oracle(errs, total, d_max))
    assert o <= dist(allocate_uniform(m, total, d_max, dev)) + 1e-6
    assert o <= dist(allocate_random(m, total, d_max, dev)) + 1e-6


def test_pilot_raises_if_oracle_beaten_by_uniform():
    """The pilot must refuse to emit a verdict from an impossible measurement."""
    torch.manual_seed(0)
    q = _quantizer()
    z = torch.randn(64, 8)
    import adarq_flow.eval.allocation as alloc

    real = alloc.allocate_oracle
    # Sabotage the allocator: always spend the minimum.
    alloc.allocate_oracle = lambda errs, total, d_max, **kw: torch.ones(
        errs.shape[0], dtype=torch.long, device=errs.device)
    try:
        with pytest.raises(RuntimeError, match="impossible"):
            headroom_pilot(q, z, mean_depth=2.5)
    finally:
        alloc.allocate_oracle = real


# -- end-to-end pilot ------------------------------------------------------------------
def test_pilot_reports_rate_matched_policies():
    torch.manual_seed(0)
    q = _quantizer()
    z = torch.randn(256, 8)
    _fit(q, z)
    rep = headroom_pilot(q, z, mean_depth=2.5)
    for name in ("uniform", "random", "oracle"):
        assert abs(rep.by_name(name).mean_depth - rep.mean_depth_budget) < 1e-2
    assert rep.oracle_gain_vs_uniform >= -1e-9
    assert "VERDICT" in rep.summary()


def test_synthetic_verdict_is_never_a_refutation():
    """Only real latents can refute the thesis; synthetic data validates the instrument."""
    torch.manual_seed(0)
    q = _quantizer()
    z = torch.randn(256, 8)
    _fit(q, z)
    rep = headroom_pilot(q, z, mean_depth=2.5, real_data=False)
    if rep.oracle_gain_vs_uniform < rep.effect_floor:
        assert "NOT a refutation" in rep.verdict
        assert "THESIS REFUTED" not in rep.verdict


def test_degenerate_budget_is_avoided():
    """A budget pinned at D_max leaves no allocation freedom; the pilot must not use it."""
    torch.manual_seed(0)
    q = _quantizer()
    z = torch.randn(128, 8)
    _fit(q, z)
    rep = headroom_pilot(q, z, mean_depth=float(q.max_depth))
    assert rep.mean_depth_budget < q.max_depth
