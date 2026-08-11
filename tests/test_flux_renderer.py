"""Real FLUX renderer + VAE adapter.

Structural tests always run. Tests needing a checkpoint use the ungated
``hf-internal-testing/tiny-flux-pipe`` fixture — the genuine diffusers classes with tiny
random weights — and skip when it is unreachable, so CI stays green offline.

Real FLUX.1-dev is gated and was NOT available during development; see the module
docstring of `adarq_flow/renderer/flux.py` for exactly what is and is not verified.
"""

import dataclasses

import pytest

from adarq_flow.config import AdaRQFlowConfig, RendererConfig, get_preset

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from adarq_flow.renderer import build_image_latent_encoder, build_renderer_backbone, is_flux
from adarq_flow.renderer.flux import is_test_fixture

TINY = "hf-internal-testing/tiny-flux-pipe"


def _fixture_available() -> bool:
    try:
        from adarq_flow.renderer import flux_vae_dims

        flux_vae_dims(TINY)
        return True
    except Exception:
        return False


needs_fixture = pytest.mark.skipif(
    not _fixture_available(), reason="tiny-flux fixture not cached/reachable")


# -- routing ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_id,expected", [
    ("black-forest-labs/FLUX.1-dev", True),
    ("black-forest-labs/FLUX.1-schnell", True),
    ("hf-internal-testing/tiny-flux-pipe", True),
    ("Qwen/Qwen2.5-VL-3B-Instruct", False),
    ("dummy", False),
])
def test_flux_routing(model_id, expected):
    assert is_flux(model_id) is expected


def test_unknown_renderer_ids_still_refuse():
    with pytest.raises(NotImplementedError, match="Refusing to substitute"):
        build_renderer_backbone(RendererConfig(backbone="some/unknown-model"))
    with pytest.raises(NotImplementedError, match="Refusing to substitute"):
        build_image_latent_encoder(RendererConfig(vae="some/unknown-model"))


# -- a random-weight fixture must never count as a real component ----------------------
@pytest.mark.parametrize("model_id", [
    TINY, "hf-internal-testing/tiny-random-Qwen2VL", "some/tiny-flux-thing"])
def test_test_fixtures_are_recognised(model_id):
    assert is_test_fixture(model_id)


def test_fixture_cannot_satisfy_experiment_mode():
    """Genuine architecture + random weights is worthless as a result.

    Without this guard a config could pass run_mode='experiment' simply because the id
    is not the literal string 'dummy', and report numbers from random weights.
    """
    cfg = get_preset("tiny_cpu")
    cfg = dataclasses.replace(
        cfg,
        renderer=dataclasses.replace(cfg.renderer, backbone=TINY, vae=TINY),
        train=dataclasses.replace(cfg.train, run_mode="experiment"))
    dummies = cfg.dummy_components()
    assert "renderer.backbone" in dummies and "renderer.vae" in dummies
    with pytest.raises(ValueError, match="forbids development stand-ins"):
        cfg.validate_run_mode()


def test_real_flux_ids_are_not_flagged_as_fixtures():
    assert not is_test_fixture("black-forest-labs/FLUX.1-dev")
    cfg = AdaRQFlowConfig.from_yaml("configs/experiment.yaml")
    assert "renderer.backbone" not in cfg.dummy_components()


# -- packing is FLUX's, not ours --------------------------------------------------------
def test_pack_shape_and_invertibility():
    from adarq_flow.renderer import FluxVAE

    x = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
    packed = FluxVAE.pack(x)
    assert packed.shape == (2, (4 // 2) * (6 // 2), 3 * 4)
    # packing must be a permutation: no value invented or lost
    assert torch.equal(packed.flatten().sort().values, x.flatten().sort().values)


def test_pack_rejects_odd_grid():
    from adarq_flow.renderer import FluxVAE

    with pytest.raises(ValueError, match="2x2-pack"):
        FluxVAE.pack(torch.randn(1, 2, 3, 4))


# -- against the genuine diffusers classes ---------------------------------------------
@needs_fixture
def test_vae_dims_read_from_config():
    from adarq_flow.renderer import flux_vae_dims

    d = flux_vae_dims(TINY)
    assert d["packed_dim"] == d["latent_channels"] * 4
    assert d["scaling_factor"] != 0.0


@needs_fixture
def test_vae_produces_packed_latents_and_checks_width():
    from adarq_flow.renderer import FluxVAE, flux_vae_dims

    d = flux_vae_dims(TINY)
    cfg = RendererConfig(vae=TINY, latent_dim=int(d["packed_dim"]),
                         num_image_tokens=(64 // 2) ** 2)
    vae = FluxVAE(TINY, cfg, dtype=torch.float32)
    out = vae(torch.rand(2, 3, 64, 64) * 2 - 1)
    assert out.shape == (2, (64 // 2) ** 2, int(d["packed_dim"]))

    with pytest.raises(ValueError, match="does not match"):
        FluxVAE(TINY, dataclasses.replace(cfg, latent_dim=999), dtype=torch.float32)


@needs_fixture
def test_backbone_requires_text_conditioning():
    """FLUX is text-conditioned; zeroing the prompt silently would be a fabrication."""
    from diffusers import FluxTransformer2DModel

    from adarq_flow.renderer import FluxRendererBackbone

    tc = FluxTransformer2DModel.load_config(TINY, subfolder="transformer")
    cfg = RendererConfig(backbone=TINY, latent_dim=int(tc["in_channels"]),
                         controlnet_double_blocks=1, controlnet_single_blocks=1)
    bb = FluxRendererBackbone(TINY, cfg, dtype=torch.float32)
    x = torch.randn(1, 16, int(tc["in_channels"]))
    with pytest.raises(ValueError, match="text-conditioned"):
        bb(x, torch.tensor([0.5]), control_residuals=None, text=None)


@needs_fixture
def test_too_many_controlnet_blocks_raises():
    from diffusers import FluxTransformer2DModel

    from adarq_flow.renderer import FluxRendererBackbone

    tc = FluxTransformer2DModel.load_config(TINY, subfolder="transformer")
    cfg = RendererConfig(backbone=TINY, latent_dim=int(tc["in_channels"]),
                         controlnet_double_blocks=int(tc["num_layers"]) + 5)
    with pytest.raises(ValueError, match="exceeds"):
        FluxRendererBackbone(TINY, cfg, dtype=torch.float32)
