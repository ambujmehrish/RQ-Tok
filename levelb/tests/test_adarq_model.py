"""AdaRQ-Flow arm on upstream's model: shared hidden states, heads, sampling rules."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from levelb import adarq_model as am
from levelb import bifrost_build as bb

from .test_bifrost_build import HIDDEN, tiny_vlm  # noqa: F401  (fixture)

K, D = 16, 4


def _batch(model, n=2, seed=0):
    conv = bb.make_conversation()
    torch.manual_seed(seed)
    z = torch.randn(n, bb.NUM_TOKENS, HIDDEN)
    mask = bb.sample_mar_mask(n, bb.NUM_TOKENS, np.random.default_rng(seed))
    return bb.t2i_flow(conv, ["tench", "goldfish"][:n], z, mask), z, mask


def test_branch_hidden_reproduces_upstream_forward(tiny_vlm):  # noqa: F811
    model = bb.build_model(tiny_vlm, inner_dim=HIDDEN).float().eval()
    flow, z, _ = _batch(model)
    with torch.no_grad():
        up = model(t2i_flow=flow, **bb.forward_kwargs(2))["loss"]
        h = am.branch_hidden(model, flow, 2)
        ours = F.mse_loss(model.vision_head(h).float(), z.float())
    assert torch.equal(up, ours)


def _heads(seed=0):
    g = torch.Generator().manual_seed(seed)
    scale = torch.tensor([1.0, 0.5, 0.25, 0.1])[:, None, None]
    books = torch.randn(D, K, HIDDEN, generator=g) * scale
    books[:, 0] = 0                                           # reserved zero codeword
    return am.AdaRQHeads(HIDDEN, books, res_rms=0.1, flow_width=32, flow_depth=2)


def test_heads_loss_and_gradients(tiny_vlm):  # noqa: F811
    model = bb.build_model(tiny_vlm, inner_dim=HIDDEN).float()
    heads = _heads()
    counts = am.set_trainable_adarq(model, heads)
    assert counts["adarq_heads"] > 0
    assert not any(p.requires_grad for p in model.vision_head.parameters())
    flow, z, mask = _batch(model)
    codes = torch.randint(0, K, (2, bb.NUM_TOKENS, D))
    codes[..., 2:] = K                                        # depth 2 everywhere
    h = am.branch_hidden(model, flow, 2)
    out = heads.loss(h, z, codes, mask.bool())
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    q = [p.grad for n, p in model.named_parameters() if "vision_branch_q_proj" in n]
    assert all(g is not None and g.abs().sum() > 0 for g in q)
    assert heads.code_out[0].weight.grad.abs().sum() > 0
    assert heads.flow.out.weight.grad.abs().sum() > 0          # zero-init output still learns


def test_sampling_respects_halt_rules():
    heads = _heads()
    h = torch.randn(512, HIDDEN)
    lat, depth = heads.sample(h, generator=torch.Generator().manual_seed(0), flow_steps=4)
    assert lat.shape == (512, HIDDEN) and torch.isfinite(lat).all()
    assert depth.min() >= 1 and depth.max() <= D             # level 1 never halts
    assert len(set(depth.tolist())) > 1                       # depth actually varies


def test_guided_sampling_has_no_nan_at_level_one():
    """Guidance on a masked (-inf) halt logit gives -inf - (-inf) = NaN; guide raw logits."""
    heads = _heads()
    h_c, h_u = torch.randn(64, HIDDEN), torch.randn(64, HIDDEN)
    lat, depth = heads.sample(h_c, h_u, cfg_weight=1.0, flow_steps=2,
                              generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(lat).all() and depth.min() >= 1
    lat2, _ = heads.sample(h_c, h_u, cfg_weight=3.0, flow_steps=2,
                           generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(lat2).all()


def _level_draws(heads, lg, lvl, n, **kw):
    """Draw n symbols for one row of logits with fresh uniforms."""
    g = torch.Generator().manual_seed(0)
    rows = lg.expand(n, -1)
    u_h, u_c = torch.rand(n, generator=g), torch.rand(n, generator=g)
    return heads.sample_level(rows, lvl, kw.get("temp", 1.0), kw.get("top_p"),
                              kw.get("greedy", False), u_h, u_c)


def test_gated_sampler_equals_categorical_at_temperature_one():
    heads = _heads()
    torch.manual_seed(3)
    lg = torch.randn(1, K + 1) * 2
    q = _level_draws(heads, lg, 2, 200_000)
    freq = torch.bincount(q, minlength=K + 1).float() / len(q)
    assert torch.allclose(freq, torch.softmax(lg[0], -1), atol=4e-3)
    # level 1: never halts, codes in proportion to the ordinary-code softmax
    q0 = _level_draws(heads, lg, 0, 200_000)
    assert (q0 != K).all()
    f0 = torch.bincount(q0, minlength=K + 1).float()[:K] / len(q0)
    assert torch.allclose(f0, torch.softmax(lg[0, :K], -1), atol=4e-3)


def test_temperature_and_top_p_leave_the_halt_probability_unchanged():
    heads = _heads()
    torch.manual_seed(4)
    lg = torch.randn(1, K + 1)
    lg[0, K] = 1.5                                       # a substantial halt probability
    p_halt = float(torch.softmax(lg[0], -1)[K])
    for kw in ({"temp": 0.5}, {"top_p": 0.5}, {"temp": 0.7, "top_p": 0.9}, {"greedy": True}):
        q = _level_draws(heads, lg, 1, 200_000, **kw)
        assert abs(float((q == K).float().mean()) - p_halt) < 4e-3, kw


def test_greedy_and_top_p_restrict_the_ordinary_codes():
    heads = _heads()
    torch.manual_seed(5)
    lg = torch.randn(1, K + 1)
    lg[0, K] = -1e4                                      # never halts: look at codes only
    q = _level_draws(heads, lg, 1, 50_000, greedy=True)
    assert (q == lg[0, :K].argmax()).all()
    p = torch.softmax(lg[0, :K], -1)
    order = p.argsort(descending=True)
    nucleus = order[: int((p[order].cumsum(0) - p[order] < 0.6).sum())]
    q = _level_draws(heads, lg, 1, 50_000, top_p=0.6)
    assert set(q.unique().tolist()) <= set(nucleus.tolist())
    assert len(set(q.unique().tolist())) == len(nucleus)  # every nucleus code reachable


def test_separate_streams_keep_halting_and_residuals_aligned():
    """Changing the code temperature must not change the halting or residual draws."""
    heads = _heads().eval()
    torch.manual_seed(6)
    h = torch.randn(64, HIDDEN)

    def run(temps):
        gens = {k: torch.Generator().manual_seed(s) for k, s in (("code", 1), ("halt", 2),
                                                                 ("res", 3))}
        return heads.sample(h, None, code_temps=temps, generators=gens, flow_steps=2)

    a_lat, a_dep = run([1.0] * D)
    b_lat, b_dep = run([1.0] * D)
    assert torch.equal(a_lat, b_lat) and torch.equal(a_dep, b_dep)       # reproducible
    for temps in ([0.5] * D, [0.0] * D):
        lat, dep = run(temps)
        assert torch.isfinite(lat).all() and (dep >= 1).all()
    # the residual noise comes from its own stream: with no flow steps the latent is
    # zhat + s * scale * x0, so (latent - zhat) / s is the same x0 whatever the code settings
    x0 = torch.randn(64, heads.books.shape[-1], generator=torch.Generator().manual_seed(3))
    for temps, code_seed in (([1.0] * D, 1), ([0.3] * D, 9)):
        gens = {k: torch.Generator().manual_seed(s) for k, s in (("code", code_seed),
                                                                 ("halt", 2), ("res", 3))}
        lat, dep = heads.sample(h, None, code_temps=temps, generators=gens, flow_steps=0,
                                residual_scale=0.5)
        # same streams with residual_scale 0 gives zhat alone
        gens0 = {k: torch.Generator().manual_seed(s) for k, s in (("code", code_seed),
                                                                  ("halt", 2), ("res", 3))}
        lat0, dep0 = heads.sample(h, None, code_temps=temps, generators=gens0, flow_steps=0,
                                  residual_scale=0.0)
        assert torch.equal(dep, dep0)
        assert torch.allclose((lat - lat0) / heads.res_rms, 0.5 * x0, atol=1e-5)
