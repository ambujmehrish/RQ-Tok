"""Phase 1 tests: adaptive RVQ-CLIP tokenizer (requires torch, CPU only).

Covers shapes/dtypes, the adaptive-depth halting rule, dequantization round-trip,
EMA codebook learning, and the no-fallback (fail-loud) contracts.
"""

import pytest

torch = pytest.importorskip("torch")

from adarq_flow.config import TokenizerConfig, get_preset
from adarq_flow.tokenizer import (
    AdaptiveResidualQuantizer,
    Codebook,
    DummyCLIPEncoder,
    RVQCLIPTokenizer,
    build_clip_encoder,
    build_tokenizer,
    fit_tokenizer,
)


def _cfg(**kw):
    base = dict(clip_dim=8, num_patches=4, codebook_size=16, max_depth=4)
    base.update(kw)
    return TokenizerConfig(**base)


# -- encoder ---------------------------------------------------------------------------
def test_encoder_shape_and_frozen():
    enc = DummyCLIPEncoder(clip_dim=8, num_patches=16)
    imgs = torch.randn(2, 3, 8, 8)
    z = enc(imgs)
    assert z.shape == (2, 16, 8)
    assert all(not p.requires_grad for p in enc.parameters())


def test_encoder_deterministic():
    torch.manual_seed(0)
    enc = DummyCLIPEncoder(8, 16)
    imgs = torch.randn(1, 3, 8, 8)
    assert torch.allclose(enc(imgs), enc(imgs))


def test_encoder_rejects_bad_shape():
    enc = DummyCLIPEncoder(8, 16)
    with pytest.raises(ValueError):
        enc(torch.randn(2, 3, 8))  # missing spatial dim


def test_encoder_non_square_patches_raises():
    with pytest.raises(ValueError):
        DummyCLIPEncoder(8, num_patches=15)


# -- quantizer shapes ------------------------------------------------------------------
def test_quantize_output_shapes_and_dtypes():
    q = AdaptiveResidualQuantizer(_cfg())
    z = torch.randn(3, 4, 8)  # [B, N, d] -> M = 12
    out = q(z, update_codebook=False)
    assert out.codes.shape == (12, 4) and out.codes.dtype == torch.long
    assert out.depths.shape == (12,) and out.depths.dtype == torch.long
    assert out.zhat.shape == (12, 8)
    assert out.res.shape == (12, 8)
    # codes are real (< K) or the <halt> sentinel (== K).
    assert int(out.codes.min()) >= 0
    assert int(out.codes.max()) <= q.codebook_size
    # depth is the count of real codes; within [1, D_max].
    assert int(out.depths.min()) >= 1 and int(out.depths.max()) <= 4
    assert torch.equal(out.depths, (out.codes != q.codebook_size).sum(-1))


def test_res_is_exact_remainder():
    q = AdaptiveResidualQuantizer(_cfg())
    z = torch.randn(10, 8)
    out = q(z, update_codebook=False)
    assert torch.allclose(out.res, z - out.zhat, atol=1e-6)


def test_empty_batch_raises():
    q = AdaptiveResidualQuantizer(_cfg())
    with pytest.raises(ValueError):
        q(torch.zeros(0, 8), update_codebook=False)


def test_wrong_dim_raises():
    q = AdaptiveResidualQuantizer(_cfg())
    with pytest.raises(ValueError):
        q(torch.randn(5, 7), update_codebook=False)  # d != clip_dim


# -- adaptive-depth halting ------------------------------------------------------------
def _fixed_quantizer():
    cfg = _cfg(clip_dim=2, codebook_size=4, max_depth=3, halt_residual_threshold=0.05)
    q = AdaptiveResidualQuantizer(cfg)
    embed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    book: Codebook = q.codebooks[0]
    book.embed.copy_(embed)
    book.initted.fill_(True)  # freeze: no first-batch reinit
    return q


def test_halting_depth_varies_by_detail():
    q = _fixed_quantizer()
    # [1,0] is exactly a code -> residual 0 -> halt at depth 1.
    # [1,1] needs two codes ([1,0] then [0,1]) -> depth 2.
    z = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    out = q(z, update_codebook=False)
    assert int(out.depths[0]) == 1
    assert int(out.depths[1]) == 2
    # Halted positions carry the sentinel.
    assert int(out.codes[0, 1]) == q.codebook_size
    assert int(out.codes[0, 2]) == q.codebook_size


def test_non_adaptive_uses_full_depth():
    cfg = _cfg(adaptive_depth=False, max_depth=3)
    q = AdaptiveResidualQuantizer(cfg)
    out = q(torch.randn(20, 8), update_codebook=False)
    assert torch.all(out.depths == 3)
    assert int((out.codes == q.codebook_size).sum()) == 0


# -- dequantization round-trip ---------------------------------------------------------
def test_dequantize_matches_zhat():
    q = AdaptiveResidualQuantizer(_cfg())
    z = torch.randn(15, 8)
    out = q(z, update_codebook=False)
    zhat = q.dequantize(out.codes)
    assert torch.allclose(zhat, out.zhat, atol=1e-6)


def test_dequantize_bad_shape_raises():
    q = AdaptiveResidualQuantizer(_cfg())
    with pytest.raises(ValueError):
        q.dequantize(torch.zeros(5, 99, dtype=torch.long))


def test_per_depth_codebooks_roundtrip():
    cfg = _cfg(shared_codebook=False)
    q = AdaptiveResidualQuantizer(cfg)
    assert len(q.codebooks) == cfg.max_depth
    z = torch.randn(12, 8)
    out = q(z, update_codebook=False)
    assert torch.allclose(q.dequantize(out.codes), out.zhat, atol=1e-6)


# -- EMA learning ----------------------------------------------------------------------
def test_ema_reduces_reconstruction_error():
    torch.manual_seed(0)
    cfg = _cfg(clip_dim=8, codebook_size=32, max_depth=4)
    tok = RVQCLIPTokenizer(cfg, backbone="dummy")

    # Clustered latents: 8 gaussian centers + small noise.
    centers = torch.randn(8, 8) * 3.0
    assign = torch.randint(8, (512,))
    latents = centers[assign] + 0.05 * torch.randn(512, 8)

    initial = float(tok(latents, update_codebook=False).losses.recon)
    report = fit_tokenizer(tok, latents, steps=300, batch_size=64, seed=0)
    final = report.final_recon

    assert final < initial
    assert final < 0.5 * initial
    assert report.final_usage > 0.0


def test_update_flag_is_honored_exactly():
    cfg = _cfg()
    q = AdaptiveResidualQuantizer(cfg)
    z = torch.randn(64, 8)
    before = q.codebooks[0].embed.clone()
    q.eval()
    q(z, update_codebook=False)
    assert torch.equal(q.codebooks[0].embed, before)  # eval + no update: unchanged
    q(z, update_codebook=True)
    assert not torch.equal(q.codebooks[0].embed, before)  # explicit update: changed


# -- factory / config contracts --------------------------------------------------------
def test_build_tokenizer_from_preset():
    tok = build_tokenizer(get_preset("tiny_cpu"))
    imgs = torch.randn(2, 3, 8, 8)
    out = tok.tokenize(imgs)
    cfg = get_preset("tiny_cpu").tokenizer
    n = cfg.num_patches
    assert tok.encode(imgs).shape == (2, n, cfg.clip_dim)
    assert out.codes.shape == (2 * n, cfg.max_depth)
    assert tok.codebook_vocab == cfg.codebook_size + 1


def test_unknown_backbone_raises_not_implemented():
    """Unknown encoders refuse. Qwen is wired and covered by tests/test_qwen_backbone.py."""
    with pytest.raises(NotImplementedError):
        build_clip_encoder("some/unknown-model", _cfg())


def test_bad_threshold_raises():
    with pytest.raises(ValueError):
        AdaptiveResidualQuantizer(_cfg(halt_residual_threshold=1.5))


def test_bad_max_depth_raises():
    with pytest.raises(ValueError):
        AdaptiveResidualQuantizer(_cfg(max_depth=0))


def test_codebook_bad_decay_raises():
    with pytest.raises(ValueError):
        Codebook(codebook_size=4, dim=2, ema_decay=1.0)
