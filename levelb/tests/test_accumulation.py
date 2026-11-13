"""Gradient accumulation must reproduce the full-batch update (code test)."""

from __future__ import annotations

import numpy as np
import torch

from levelb import adarq_model as am
from levelb import bifrost_build as bb

from .test_adarq_model import D, K, _heads
from .test_bifrost_build import HIDDEN, tiny_vlm  # noqa: F401  (fixture)


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_bifrost_accumulation_equals_full_batch(tiny_vlm):  # noqa: F811
    model = bb.build_model(tiny_vlm, inner_dim=HIDDEN).float()
    bb.set_trainable(model)
    conv = bb.make_conversation()
    torch.manual_seed(0)
    z = torch.randn(4, bb.NUM_TOKENS, HIDDEN)
    mask = bb.sample_mar_mask(4, bb.NUM_TOKENS, np.random.default_rng(0))
    prompts = ["tench", "goldfish", "great white shark", "tiger shark"]

    model.zero_grad()
    full = model(t2i_flow=bb.t2i_flow(conv, prompts, z, mask), **bb.forward_kwargs(4))["loss"]
    full.backward()
    g_full = _grads(model)

    model.zero_grad()
    for s in (0, 2):
        part = model(t2i_flow=bb.t2i_flow(conv, prompts[s:s + 2], z[s:s + 2], mask[s:s + 2]),
                     **bb.forward_kwargs(2))["loss"]
        (part * 0.5).backward()
    g_acc = _grads(model)
    assert g_full.keys() == g_acc.keys()
    for n in g_full:
        assert torch.allclose(g_full[n], g_acc[n], rtol=1e-4, atol=1e-6), n


def test_adarq_normalisation_matches_full_batch():
    heads = _heads()
    torch.manual_seed(0)
    h = torch.randn(4, 64, HIDDEN)
    z = torch.randn(4, 64, HIDDEN)
    codes = torch.randint(1, K, (4, 64, D))
    depth = torch.randint(1, D + 1, (4, 64))
    for k in range(D):
        codes[..., k][depth <= k] = K                       # halt suffix
    masked = torch.rand(4, 64) < 0.8
    totals = heads.counts(codes, masked)
    parts = [heads.counts(codes[s:s + 2], masked[s:s + 2]) for s in (0, 2)]
    assert totals == (parts[0][0] + parts[1][0], parts[0][1] + parts[1][1])
    full = heads.loss(h, z, codes, masked)["code_ce"]
    split = sum(heads.loss(h[s:s + 2], z[s:s + 2], codes[s:s + 2], masked[s:s + 2],
                           totals=totals)["code_ce"] for s in (0, 2))
    assert torch.allclose(full, split, rtol=1e-6)
    assert isinstance(am.AdaRQHeads, type)


def _greedy_codes(heads, z, depth):
    """Codes consistent with the heads' codebooks: greedy RVQ, then a <halt> suffix."""
    flat = z.reshape(-1, z.shape[-1]).float()
    codes = torch.full((flat.shape[0], heads.D), heads.K, dtype=torch.long)
    zhat = torch.zeros_like(flat)
    for k in range(heads.D):
        d2 = heads.sq_distances(flat, zhat, k)
        codes[:, k] = d2.argmin(-1)
        zhat = zhat + heads.books[k][codes[:, k]]
    codes = codes.reshape(*z.shape[:-1], heads.D)
    for k in range(heads.D):
        codes[..., k][depth <= k] = heads.K
    return codes


def test_soft_targets_reduce_to_hard_and_accumulate_exactly():
    heads = _heads()
    torch.manual_seed(1)
    h, z = torch.randn(4, 64, HIDDEN), torch.randn(4, 64, HIDDEN)
    depth = torch.randint(1, D + 1, (4, 64))
    codes = _greedy_codes(heads, z, depth)
    masked = torch.rand(4, 64) < 0.8
    # T -> 0: the soft target is the one-hot of the stored (nearest) code
    heads.soft_temps = [1e-8] * D
    out = heads.loss(h, z, codes, masked)
    assert torch.allclose(out["code_ce"], out["code_ce_hard"], rtol=1e-5)
    # a real temperature: a different, finite loss with gradients, and exact accumulation
    heads.soft_temps = [float(t) for t in torch.linspace(5.0, 1.0, D)]
    full = heads.loss(h, z, codes, masked)
    assert torch.isfinite(full["loss"]) and not torch.allclose(full["code_ce"], full["code_ce_hard"])
    full["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in heads.parameters())
    totals = heads.counts(codes, masked)
    split = sum(heads.loss(h[s:s + 2], z[s:s + 2], codes[s:s + 2], masked[s:s + 2],
                           totals=totals)["code_ce"] for s in (0, 2))
    assert torch.allclose(full["code_ce"], split, rtol=1e-6)
