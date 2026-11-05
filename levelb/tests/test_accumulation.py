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
