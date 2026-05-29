"""Phase 2 tests: MLLM vision-generation branch + hybrid head (torch, CPU).

Covers backbone/branch/head shapes, the flow-matching utilities, MAR masked
training (losses finite, decrease, frozen backbone gets no grad), MaskGIT decoding
(valid codes/depths, CFG, determinism), tokenizer<->model integration, and the
no-fallback contracts.
"""

import pytest

torch = pytest.importorskip("torch")

from bifrost_flow.config import MLLMConfig, TokenizerConfig, get_preset
from bifrost_flow.mllm import (
    DummyMLLMBackbone,
    VisionGenBranch,
    VisionGenModel,
    build_backbone,
    build_vision_gen_model,
    flow_matching_loss,
    flow_sample,
    rectified_flow_target,
)
from bifrost_flow.mllm.heads import CodeClassifierHead, FlowResidualHead
from bifrost_flow.tokenizer import build_tokenizer


def _cfgs():
    tok = TokenizerConfig(clip_dim=16, num_patches=9, codebook_size=12, max_depth=3)
    mllm = MLLMConfig(backbone="dummy", hidden_dim=32, num_layers=2, num_heads=4,
                      flow_head_dim=32, flow_head_layers=2, cfg_text_dropout=0.0,
                      decode_steps=6)
    return mllm, tok


def _text(b, t=5, vocab=256):
    return torch.randint(0, vocab, (b, t))


# -- backbone --------------------------------------------------------------------------
def test_backbone_shapes_and_frozen():
    bb = DummyMLLMBackbone(hidden_dim=32, num_layers=2, num_heads=4)
    h = bb(_text(2))
    assert h.shape == (2, 5, 32)
    assert all(not p.requires_grad for p in bb.parameters())


def test_backbone_rejects_bad_tokens():
    bb = DummyMLLMBackbone(32, 2, 4, vocab_size=10)
    with pytest.raises(ValueError):
        bb(torch.tensor([[99]]))  # id out of range


def test_real_backbone_raises():
    with pytest.raises(NotImplementedError):
        build_backbone(MLLMConfig(backbone="Qwen/Qwen2.5-VL-7B-Instruct"))


# -- branch / heads --------------------------------------------------------------------
def test_branch_shapes_and_mismatch():
    br = VisionGenBranch(32, 2, 4)
    out = br(torch.randn(2, 9, 32), torch.randn(2, 5, 32))
    assert out.shape == (2, 9, 32)
    with pytest.raises(ValueError):
        br(torch.randn(2, 9, 32), torch.randn(3, 5, 32))  # batch mismatch


def test_heads_shapes():
    ch = CodeClassifierHead(32, max_depth=3, vocab=13)
    assert ch(torch.randn(2, 9, 32)).shape == (2, 9, 3, 13)
    fh = FlowResidualHead(latent_dim=16, hidden_dim=32, width=32, num_layers=2)
    v = fh(torch.randn(4, 16), torch.rand(4), torch.randn(4, 32), torch.randn(4, 16))
    assert v.shape == (4, 16)


# -- flow matching ---------------------------------------------------------------------
def test_rectified_flow_target():
    x1 = torch.randn(5, 8)
    x0 = torch.zeros(5, 8)
    t = torch.ones(5)
    x_t, u = rectified_flow_target(x1, x0, t)
    assert torch.allclose(x_t, x1)         # t=1 -> data
    assert torch.allclose(u, x1)           # u = x1 - 0


def test_flow_sample_shape_and_determinism():
    torch.manual_seed(0)
    head = FlowResidualHead(8, 16, 16, 2)
    h = torch.randn(4, 16)
    z = torch.randn(4, 8)

    def fn(x_t, t):
        return head(x_t, t, h, z)

    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = flow_sample(fn, 4, 8, steps=5, device=torch.device("cpu"), generator=g1)
    b = flow_sample(fn, 4, 8, steps=5, device=torch.device("cpu"), generator=g2)
    assert a.shape == (4, 8) and torch.allclose(a, b)


def test_flow_head_learns_target():
    torch.manual_seed(0)
    head = FlowResidualHead(4, 8, 32, 2)
    h = torch.randn(64, 8)
    z = torch.zeros(64, 4)
    target = torch.tanh(h[:, :4])          # deterministic function of conditioning
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(0)
    for _ in range(150):
        opt.zero_grad()
        loss = flow_matching_loss(lambda x_t, t: head(x_t, t, h, z), target, g)
        loss.backward()
        opt.step()
    sample = flow_sample(lambda x_t, t: head(x_t, t, h, z), 64, 4, 50,
                         torch.device("cpu"), generator=torch.Generator().manual_seed(1))
    assert (sample - target).pow(2).mean() < 0.1 * target.pow(2).mean()


# -- training (MAR) --------------------------------------------------------------------
def _random_targets(model, b):
    codes = torch.randint(0, model.K + 1, (b, model.N, model.D))
    res = torch.randn(b, model.N, model.d) * 0.1
    zhat = torch.randn(b, model.N, model.d)
    return codes, res, zhat


def test_compute_loss_finite_and_backbone_frozen():
    mllm, tok = _cfgs()
    model = VisionGenModel(mllm, tok)
    codes, res, zhat = _random_targets(model, 2)
    losses = model.compute_loss(_text(2), codes, res, zhat)
    assert torch.isfinite(losses.total) and losses.code_ce > 0 and losses.flow >= 0
    losses.total.backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert any(p.grad is not None for p in model.branch.parameters())


def test_compute_loss_decreases():
    torch.manual_seed(0)
    mllm, tok = _cfgs()
    model = VisionGenModel(mllm, tok)
    text = _text(2)
    codes, res, zhat = _random_targets(model, 2)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2)

    def step():
        g = torch.Generator().manual_seed(123)   # fix mask + flow noise -> deterministic
        opt.zero_grad()
        loss = model.compute_loss(text, codes, res, zhat, generator=g)
        loss.total.backward()
        opt.step()
        return loss.total.detach().item()

    first = step()
    for _ in range(60):
        last = step()
    assert last < first


# -- decoding --------------------------------------------------------------------------
def test_generate_valid_outputs():
    torch.manual_seed(0)
    mllm, tok = _cfgs()
    model = VisionGenModel(mllm, tok)
    # Dequantize must use this model's own tokenizer config (matching D_max / dim).
    from bifrost_flow.tokenizer import AdaptiveResidualQuantizer
    q = AdaptiveResidualQuantizer(tok)
    out = model.generate(_text(2), q.dequantize, steps=6, cfg_scale=2.0, temperature=0.0)
    assert out.codes.shape == (2, model.N, model.D)
    assert int(out.codes.min()) >= 0 and int(out.codes.max()) <= model.K
    assert out.depths.shape == (2, model.N)
    assert int(out.depths.max()) <= model.D
    assert out.latents.shape == (2, model.N, model.d)
    assert torch.allclose(out.latents, out.zhat + out.residual, atol=1e-6)


def test_generate_greedy_codes_deterministic():
    torch.manual_seed(0)
    mllm, tok = _cfgs()
    model = VisionGenModel(mllm, tok)
    from bifrost_flow.tokenizer import AdaptiveResidualQuantizer
    q = AdaptiveResidualQuantizer(tok)
    text = _text(1)
    a = model.generate(text, q.dequantize, steps=6, cfg_scale=2.0, temperature=0.0)
    b = model.generate(text, q.dequantize, steps=6, cfg_scale=2.0, temperature=0.0)
    assert torch.equal(a.codes, b.codes)   # greedy codes are deterministic


# -- integration with tokenizer --------------------------------------------------------
def test_tokenizer_to_model_roundtrip():
    torch.manual_seed(0)
    cfg = get_preset("tiny_cpu")
    tok = build_tokenizer(cfg)
    model = build_vision_gen_model(cfg)
    imgs = torch.randn(2, 3, 8, 8)
    z = tok.encode(imgs)
    qo = tok(z, update_codebook=False)
    B, N, D, d = 2, cfg.tokenizer.num_patches, cfg.tokenizer.max_depth, cfg.tokenizer.clip_dim
    codes = qo.codes.reshape(B, N, D)
    zhat = qo.zhat.reshape(B, N, d)
    res = qo.res.reshape(B, N, d)
    losses = model.compute_loss(_text(B), codes, res, zhat)
    assert torch.isfinite(losses.total)
    out = model.generate(_text(B), tok.dequantize, steps=4, temperature=0.0)
    assert out.codes.shape == (B, N, D)


# -- no-fallback contracts -------------------------------------------------------------
def test_bad_target_shape_raises():
    mllm, tok = _cfgs()
    model = VisionGenModel(mllm, tok)
    with pytest.raises(ValueError):
        model.compute_loss(_text(2), torch.zeros(2, 3, 3, dtype=torch.long),
                           torch.randn(2, model.N, model.d),
                           torch.randn(2, model.N, model.d))


def test_codes_out_of_range_raises():
    mllm, tok = _cfgs()
    model = VisionGenModel(mllm, tok)
    bad = torch.full((2, model.N, model.D), model.K + 5, dtype=torch.long)
    with pytest.raises(ValueError):
        model.compute_loss(_text(2), bad, torch.randn(2, model.N, model.d),
                           torch.randn(2, model.N, model.d))
