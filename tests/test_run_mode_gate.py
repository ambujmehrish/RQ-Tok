"""No dummy component may be reachable from a run declared as an experiment.

The failure this guards against is not a crash — it is a run that completes, writes a
checkpoint and reports a loss curve, while silently having trained against a randomly
initialized stand-in or synthetic data. Those numbers look exactly like real ones.
"""

import dataclasses

import pytest

from adarq_flow.config import AdaRQFlowConfig, get_preset

torch = pytest.importorskip("torch")

from adarq_flow.training import Trainer


def _experiment(cfg: AdaRQFlowConfig, **train_kw) -> AdaRQFlowConfig:
    return dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, run_mode="experiment", **train_kw))


# -- the gate --------------------------------------------------------------------------
def test_smoke_preset_is_all_dummy_and_valid():
    cfg = get_preset("tiny_cpu")
    assert cfg.train.run_mode == "smoke"
    assert set(cfg.dummy_components()) == {
        "mllm.backbone", "renderer.backbone", "renderer.vae", "train.dataset"}
    cfg.validate_run_mode()          # all-dummy smoke config is coherent


@pytest.mark.parametrize("field,value", [
    ("mllm.backbone", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("renderer.backbone", "black-forest-labs/FLUX.1-dev"),
    ("renderer.vae", "black-forest-labs/FLUX.1-dev"),
    ("train.dataset", "blip3o"),
])
def test_experiment_mode_rejects_every_remaining_dummy(field, value):
    """Making ONE component real must not be enough — the rest are still stand-ins."""
    cfg = get_preset("tiny_cpu")
    block, name = field.split(".")
    cfg = dataclasses.replace(cfg, **{block: dataclasses.replace(
        getattr(cfg, block), **{name: value})})
    cfg = _experiment(cfg)
    with pytest.raises(ValueError, match="forbids development stand-ins"):
        cfg.validate_run_mode()


def test_mixed_real_and_dummy_is_rejected_even_in_smoke_mode():
    """A real backbone fed synthetic data is uninterpretable in either direction."""
    cfg = get_preset("tiny_cpu")
    cfg = dataclasses.replace(cfg, mllm=dataclasses.replace(
        cfg.mllm, backbone="Qwen/Qwen2.5-VL-7B-Instruct"))
    with pytest.raises(ValueError, match="mixed real/stand-in"):
        cfg.validate_run_mode()


def test_bad_run_mode_raises():
    cfg = get_preset("tiny_cpu")
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, run_mode="real"))
    with pytest.raises(ValueError, match="run_mode"):
        cfg.validate_run_mode()


def test_trainer_enforces_the_gate():
    cfg = _experiment(get_preset("tiny_cpu"), stage="tokenizer", max_steps=1)
    with pytest.raises(ValueError, match="forbids development stand-ins"):
        Trainer(cfg, dataset_length=8, setup_dist=False)


# -- the shipped experiment config ------------------------------------------------------
def test_experiment_config_has_no_dummy_components():
    cfg = AdaRQFlowConfig.from_yaml("configs/experiment.yaml")
    assert cfg.train.run_mode == "experiment"
    assert cfg.dummy_components() == {}
    cfg.validate_run_mode()
    # and the allocation prerequisites from ISSUES.md D1/D2 are set
    assert cfg.tokenizer.shared_codebook is False
    assert cfg.tokenizer.include_zero_code is True


def test_experiment_config_fails_loud_rather_than_substituting():
    """Real adapters are unwired; the run must raise, not fall back to stand-ins."""
    cfg = AdaRQFlowConfig.from_yaml("configs/experiment.yaml")
    with pytest.raises(NotImplementedError):
        Trainer(cfg, dataset_length=8, setup_dist=False)


# -- the image-latent encoder can no longer be hard-wired to a dummy --------------------
def test_real_vae_raises_instead_of_using_the_stand_in():
    from adarq_flow.renderer import build_image_latent_encoder

    cfg = get_preset("tiny_cpu").renderer
    cfg = dataclasses.replace(cfg, vae="black-forest-labs/FLUX.1-dev")
    with pytest.raises(NotImplementedError, match="Refusing to substitute"):
        build_image_latent_encoder(cfg)


# -- YAML must not silently become defaults --------------------------------------------
def test_empty_yaml_raises_instead_of_defaulting(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ValueError, match="empty or is not a YAML mapping"):
        AdaRQFlowConfig.from_yaml(str(p))


def test_non_mapping_yaml_raises(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="empty or is not a YAML mapping"):
        AdaRQFlowConfig.from_yaml(str(p))
