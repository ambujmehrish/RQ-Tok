"""Ablations must CHANGE BEHAVIOUR, not just config values.

An ablation whose flag is never read by runtime code is worse than no ablation: it
silently reports "mechanism removed, metric unchanged" and thereby fabricates a null
result. Asserting on `cfg.x is False` cannot catch that — these tests assert on the
constructed model and on the tensors the training step actually consumes.
"""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from adarq_flow.eval import ablation_configs
from adarq_flow.mllm import build_vision_gen_model
from adarq_flow.training import Trainer


def _cfgs():
    return ablation_configs("tiny_cpu")


def _model_signature(cfg):
    """Structure + parameter count + which heads exist."""
    torch.manual_seed(0)
    m = build_vision_gen_model(cfg)
    return (
        sum(p.numel() for p in m.parameters()),
        m.code_head is not None,
        m.flow_head is not None,
        m.residual_objective,
    )


@pytest.mark.parametrize("name", [
    "continuous_mse", "continuous_flow", "hybrid_mse",
    "depth_2", "depth_8", "codebook_16", "codebook_256", "discrete_only",
])
def test_ablation_changes_model(name):
    cfgs = _cfgs()
    assert _model_signature(cfgs[name]) != _model_signature(cfgs["baseline"]), (
        f"ablation '{name}' produces a model identical to baseline — the flag it sets "
        "is not read by runtime code, so any experiment using it is a silent no-op"
    )


def test_head_toggles_are_honored():
    cfgs = _cfgs()
    torch.manual_seed(0)
    disc = build_vision_gen_model(cfgs["discrete_only"])
    assert disc.flow_head is None and disc.code_head is not None
    torch.manual_seed(0)
    cont = build_vision_gen_model(cfgs["continuous_mse"])
    assert cont.code_head is None and cont.flow_head is not None


def test_disabling_both_heads_raises():
    cfg = _cfgs()["baseline"]
    cfg = dataclasses.replace(cfg, mllm=dataclasses.replace(
        cfg.mllm, code_head=False, flow_residual_head=False))
    with pytest.raises(ValueError):
        build_vision_gen_model(cfg)


def test_bad_residual_objective_raises():
    cfg = _cfgs()["baseline"]
    cfg = dataclasses.replace(cfg, mllm=dataclasses.replace(
        cfg.mllm, residual_objective="diffusion"))
    with pytest.raises(ValueError):
        build_vision_gen_model(cfg)


def test_continuous_arm_loss_has_no_code_term():
    """With no code head there is no cross-entropy to report."""
    cfg = _cfgs()["continuous_flow"]
    torch.manual_seed(0)
    m = build_vision_gen_model(cfg)
    N, D, d = cfg.tokenizer.num_patches, cfg.tokenizer.max_depth, cfg.tokenizer.clip_dim
    losses = m.compute_loss(
        torch.randint(0, 256, (2, 5)),
        torch.randint(0, m.K + 1, (2, N, D)),
        torch.randn(2, N, d), torch.randn(2, N, d))
    assert losses.code_ce.detach().item() == 0.0
    assert losses.flow.detach().item() > 0.0


def test_exposure_flag_changes_renderer_conditioning():
    """train_on_dequantized must change the tensor the ControlNet is conditioned on."""
    seen = {}

    def capture(name, cfg_bool):
        cfg = _cfgs()["baseline"]
        cfg = dataclasses.replace(
            cfg,
            train=dataclasses.replace(cfg.train, stage="renderer", max_steps=1,
                                      batch_size=2),
            renderer=dataclasses.replace(cfg.renderer,
                                         train_on_dequantized=cfg_bool))
        t = Trainer(cfg, dataset_length=8, setup_dist=False)
        real = t._model.compute_loss

        def spy(image_latents, control, generator=None):
            seen[name] = control.detach().clone()
            return real(image_latents, control, generator)

        t._model.compute_loss = spy  # type: ignore[method-assign]
        t.train()

    capture("dequant", True)
    capture("groundtruth", False)
    assert not torch.allclose(seen["dequant"], seen["groundtruth"]), (
        "renderer.train_on_dequantized did not change the conditioning tensor; "
        "the exposure-bias experiment would be vacuous"
    )


def test_scheduled_sampling_fails_loud_when_unimplemented():
    cfg = _cfgs()["baseline"]
    cfg = dataclasses.replace(
        cfg,
        train=dataclasses.replace(cfg.train, stage="renderer", max_steps=1),
        renderer=dataclasses.replace(cfg.renderer, scheduled_sampling_prob=0.5))
    with pytest.raises(NotImplementedError):
        Trainer(cfg, dataset_length=8, setup_dist=False)
