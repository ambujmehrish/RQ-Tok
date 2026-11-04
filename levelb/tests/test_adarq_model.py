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
