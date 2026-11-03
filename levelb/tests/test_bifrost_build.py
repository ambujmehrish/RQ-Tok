"""Level B Stage-1 mechanics on a TINY random Qwen2.5-VL built with upstream's classes.

A code test, not an experiment: it checks branch initialisation, the trainable set, and
that upstream's real training forward + backward run on a batch built with upstream's
real prompt builder. Run in the bifrost1 environment with BIFROST_CODE set:

    BIFROST_CODE=... python -m pytest -q levelb/tests
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from levelb import bifrost_build as bb
from levelb.upstream import add_upstream_to_path

HIDDEN = 64


@pytest.fixture(scope="module")
def tiny_vlm(tmp_path_factory):
    add_upstream_to_path()
    from bifrost.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from bifrost.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    cfg = Qwen2_5_VLConfig(
        vocab_size=151936, hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
        rope_scaling={"type": "mrope", "mrope_section": [4, 2, 2]},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                       "out_hidden_size": HIDDEN, "patch_size": 14, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "window_size": 112,
                       "fullatt_block_indexes": [1], "in_chans": 3},
        vision_start_token_id=151652, image_token_id=151655, video_token_id=151656)
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg)
    out = tmp_path_factory.mktemp("tiny_qwen25vl")
    model.save_pretrained(out)
    return str(out)


@pytest.fixture(scope="module")
def model(tiny_vlm):
    m = bb.build_model(tiny_vlm, inner_dim=HIDDEN)
    return m.float()


def test_branch_is_initialised_from_mllm(model):
    lm = model.vision_language_model.model
    for layer in lm.layers:
        att = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert torch.equal(getattr(att, f"vision_branch_{name}").weight, getattr(att, name).weight)
        for a, b in zip(layer.vision_branch_mlp.parameters(), layer.mlp.parameters(), strict=True):
            assert torch.equal(a, b)
        assert torch.equal(layer.vision_branch_input_layernorm.weight, layer.input_layernorm.weight)
    assert torch.equal(lm.vision_branch_norm.weight, lm.norm.weight)
    # copies, not shared storage: training the branch must not move the frozen MLLM
    att = lm.layers[0].self_attn
    assert att.vision_branch_q_proj.weight.data_ptr() != att.q_proj.weight.data_ptr()


def test_trainable_set(model):
    counts = bb.set_trainable(model)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert counts["trainable"] > 0
    assert all(("vision_branch_" in n) or n.startswith(("vision_head.", "mask_token",
                                                        "learnable_pos_emb")) for n in names)
    assert not any(n.startswith("vision_language_model.visual") for n in names)
    assert "mask_token" in names and "learnable_pos_emb" in names


def test_mar_mask_ratio_range():
    m = bb.sample_mar_mask(512, 64, np.random.default_rng(0))
    frac = m.mean(1)
    assert frac.min() >= 0.7 - 1e-6 and frac.max() <= 1.0
    assert set(m.unique().tolist()) <= {0.0, 1.0}


def test_upstream_forward_backward(model):
    bb.set_trainable(model)
    conv = bb.make_conversation()          # real Qwen2.5-VL-3B processor (tokenizer only)
    prompts = ["tench", "goldfish"]
    torch.manual_seed(0)
    clip = torch.randn(2, bb.NUM_TOKENS, HIDDEN)
    mask = bb.sample_mar_mask(2, bb.NUM_TOKENS, np.random.default_rng(0))
    flow = bb.t2i_flow(conv, prompts, clip, mask)
    model.train()
    out = model(t2i_flow=flow, **bb.forward_kwargs(len(prompts)))
    loss = out["loss"]
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    assert grads["vision_head.weight"] is not None and grads["vision_head.weight"].abs().sum() > 0
    q = [g for n, g in grads.items() if "vision_branch_q_proj" in n]
    assert q and all(g is not None and g.abs().sum() > 0 for g in q)
    frozen = [p.grad for n, p in model.named_parameters() if not p.requires_grad]
    assert all(g is None for g in frozen)
