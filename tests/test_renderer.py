"""Phase 3 tests: flow-matching latent ControlNet + renderer (torch, CPU)."""

import pytest

torch = pytest.importorskip("torch")

from adarq_flow.config import RendererConfig, get_preset
from adarq_flow.renderer import (
    DummyFluxBackbone,
    FlowRenderer,
    LatentControlNet,
    build_renderer,
    build_renderer_backbone,
)


def _cfg(**kw):
    base = dict(backbone="dummy", latent_dim=8, model_dim=32, num_blocks=2, num_heads=4,
                num_image_tokens=9, controlnet_double_blocks=1, controlnet_single_blocks=1,
                downsample_factor=2, num_inference_steps=10)
    base.update(kw)
    return RendererConfig(**base)


def _renderer(control_dim=8, n_ctrl=16, **kw):
    # control grid 4x4 (=16), downsample 2 -> 2x2 control tokens
    return FlowRenderer(_cfg(**kw), control_dim=control_dim, num_control_tokens=n_ctrl)


# -- backbone --------------------------------------------------------------------------
def test_backbone_velocity_shape_and_frozen():
    bb = DummyFluxBackbone(latent_dim=8, model_dim=32, num_blocks=2, num_heads=4,
                           num_tokens=9)
    v = bb(torch.randn(2, 9, 8), torch.rand(2))
    assert v.shape == (2, 9, 8)
    assert all(not p.requires_grad for p in bb.parameters())


def test_backbone_bad_shape_raises():
    bb = DummyFluxBackbone(8, 32, 2, 4, 9)
    with pytest.raises(ValueError):
        bb(torch.randn(2, 5, 8), torch.rand(2))


def test_unknown_backbone_raises_but_flux_is_wired():
    """Unknown ids still refuse; FLUX now routes to the real adapter.

    FLUX coverage lives in tests/test_flux_renderer.py (against the genuine diffusers
    classes); here we only assert that an unrecognised id is never substituted.
    """
    with pytest.raises(NotImplementedError, match="Refusing to substitute"):
        build_renderer_backbone(_cfg(backbone="some/unknown-model"))


# -- controlnet ------------------------------------------------------------------------
def test_controlnet_residuals_zero_init():
    cn = LatentControlNet(control_dim=8, num_control_tokens=16, cfg=_cfg())
    res = cn(torch.randn(2, 9, 8), torch.rand(2), torch.randn(2, 16, 8))
    assert len(res) == 2  # double + single
    for r in res:
        assert r.shape == (2, 9, 32)
        assert torch.allclose(r, torch.zeros_like(r))  # zero-init => identity at start


def test_controlnet_grid_validation():
    with pytest.raises(ValueError):
        LatentControlNet(8, num_control_tokens=15, cfg=_cfg())  # not a square


# -- renderer end to end ---------------------------------------------------------------
def test_velocity_shape_and_grad_routing():
    r = _renderer()
    v = r.velocity(torch.randn(2, 9, 8), torch.rand(2), torch.randn(2, 16, 8))
    assert v.shape == (2, 9, 8)
    loss = r.compute_loss(torch.randn(2, 9, 8), torch.randn(2, 16, 8))
    loss.flow.backward()
    # frozen backbone gets no grad; trainable controlnet does.
    assert all(p.grad is None for p in r.backbone.parameters())
    assert any(p.grad is not None for p in r.controlnet.parameters())


def test_sample_shape():
    r = _renderer()
    out = r.sample(torch.randn(2, 16, 8), steps=5)
    assert out.shape == (2, 9, 8)


def test_loss_bad_image_shape_raises():
    r = _renderer()
    with pytest.raises(ValueError):
        r.compute_loss(torch.randn(2, 5, 8), torch.randn(2, 16, 8))


def test_renderer_learns_control_to_latent():
    torch.manual_seed(0)
    r = _renderer()
    # Target image latents are a fixed (pooled) function of the control -> learnable.
    control = torch.randn(16, 16, 8)
    W = torch.randn(8, 8)
    target = (control.mean(dim=1, keepdim=True).expand(-1, 9, -1) @ W)  # [16,9,8]
    opt = torch.optim.Adam([p for p in r.parameters() if p.requires_grad], lr=5e-3)
    g = torch.Generator().manual_seed(0)
    first = None
    for _ in range(120):
        opt.zero_grad()
        loss = r.compute_loss(target, control, generator=g)
        loss.flow.backward()
        opt.step()
        if first is None:
            first = loss.flow.detach().item()
    last = loss.flow.detach().item()
    assert last < first
    sample = r.sample(control, steps=50, generator=torch.Generator().manual_seed(1))
    assert (sample - target).pow(2).mean() < target.pow(2).mean()


def test_build_renderer_from_preset():
    cfg = get_preset("tiny_cpu")
    r = build_renderer(cfg)
    n = cfg.tokenizer.num_patches
    d = cfg.tokenizer.clip_dim
    v = r.velocity(torch.randn(2, cfg.renderer.num_image_tokens, cfg.renderer.latent_dim),
                   torch.rand(2), torch.randn(2, n, d))
    assert v.shape == (2, cfg.renderer.num_image_tokens, cfg.renderer.latent_dim)
