"""Phase 6 tests: metrics, tokenizer evaluator, ablation configs (torch, CPU)."""

import pytest

torch = pytest.importorskip("torch")

from adarq_flow.config import AdaRQFlowConfig, get_preset
from adarq_flow.eval import (
    ablation_configs,
    codebook_perplexity,
    dump_ablation_configs,
    evaluate_tokenizer,
    fid,
    geneval_score,
    mse,
    psnr,
    reconstruction_vs_depth,
    ssim,
)
from adarq_flow.tokenizer import build_tokenizer, fit_tokenizer


# -- self-contained metrics ------------------------------------------------------------
def test_mse_psnr_identity():
    x = torch.randn(4, 8)
    assert float(mse(x, x)) == 0.0
    assert float(psnr(x, x)) > 100  # ~inf, clamped


def test_psnr_orders_by_error():
    t = torch.randn(2, 3, 8, 8)
    near = t + 0.01 * torch.randn_like(t)
    far = t + 0.5 * torch.randn_like(t)
    assert float(psnr(near, t)) > float(psnr(far, t))


def test_ssim_range_and_identity():
    t = torch.rand(2, 3, 8, 8)
    s_self = float(ssim(t, t))
    s_noisy = float(ssim(t + 0.3 * torch.randn_like(t), t))
    assert 0.99 <= s_self <= 1.0001
    assert s_noisy < s_self


def test_codebook_perplexity_bounds():
    uniform = torch.ones(16)
    peaked = torch.tensor([100.0] + [0.0] * 15)
    assert codebook_perplexity(uniform) == pytest.approx(16, abs=1e-3)
    assert codebook_perplexity(peaked) == pytest.approx(1, abs=1e-3)


def test_external_metrics_raise():
    for fn in (fid, geneval_score):
        with pytest.raises(NotImplementedError):
            fn()


# -- tokenizer evaluator ---------------------------------------------------------------
def test_reconstruction_vs_depth_monotone():
    torch.manual_seed(0)
    cfg = get_preset("tiny_cpu")
    tok = build_tokenizer(cfg)
    centers = torch.randn(8, cfg.tokenizer.clip_dim) * 3.0
    latents = centers[torch.randint(8, (256,))] + 0.05 * torch.randn(256, cfg.tokenizer.clip_dim)
    fit_tokenizer(tok, latents, steps=200, seed=0)

    curve = reconstruction_vs_depth(tok, latents)
    assert len(curve) == cfg.tokenizer.max_depth
    errs = [e for _, e in curve]
    # More residual codes never increase reconstruction error.
    assert all(errs[i + 1] <= errs[i] + 1e-6 for i in range(len(errs) - 1))


def test_evaluate_tokenizer_report():
    torch.manual_seed(0)
    cfg = get_preset("tiny_cpu")
    tok = build_tokenizer(cfg)
    latents = torch.randn(128, cfg.tokenizer.clip_dim)
    fit_tokenizer(tok, latents, steps=150, seed=0)
    rep = evaluate_tokenizer(tok, latents)
    assert rep.recon_full_mse < 1e-5          # ẑ + res == z by construction
    assert 1 <= rep.mean_depth <= cfg.tokenizer.max_depth
    assert 0 < rep.codebook_perplexity <= cfg.tokenizer.codebook_size + 1e-3
    assert rep.depth_curve[-1][1] <= rep.depth_curve[0][1] + 1e-6


# -- ablations -------------------------------------------------------------------------
def test_ablation_configs_present_and_valid():
    cfgs = ablation_configs("tiny_cpu")
    for key in ["baseline", "fixed_depth", "depth_8", "per_depth_codebook",
                "no_flow_head", "no_exposure_fix", "bifrost_continuous"]:
        assert key in cfgs
    assert cfgs["fixed_depth"].tokenizer.adaptive_depth is False
    assert cfgs["depth_8"].tokenizer.max_depth == 8
    assert cfgs["per_depth_codebook"].tokenizer.shared_codebook is False
    assert cfgs["no_flow_head"].mllm.flow_residual_head is False
    # every variant round-trips through (de)serialization
    for cfg in cfgs.values():
        assert AdaRQFlowConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_dump_ablation_configs(tmp_path):
    paths = dump_ablation_configs(str(tmp_path), base="tiny_cpu")
    assert len(paths) == len(ablation_configs("tiny_cpu"))
    loaded = AdaRQFlowConfig.from_yaml(paths[0])
    assert isinstance(loaded, AdaRQFlowConfig)
