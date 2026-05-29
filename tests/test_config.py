"""Phase 0 scaffold tests: config system + package imports (pure Python, no torch)."""

import os

import pytest

import bifrost_flow
from bifrost_flow.config import (
    BifrostFlowConfig,
    TokenizerConfig,
    available_presets,
    get_preset,
)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


def test_package_imports():
    import importlib

    assert bifrost_flow.__version__
    # Submodules import without torch installed.
    for sub in [
        "tokenizer",
        "mllm",
        "renderer",
        "training",
        "inference",
        "eval",
        "data",
        "utils",
    ]:
        importlib.import_module(f"bifrost_flow.{sub}")


def test_presets_exist():
    assert set(available_presets()) >= {"tiny_cpu", "base_gpu"}


@pytest.mark.parametrize("name", ["tiny_cpu", "base_gpu"])
def test_preset_roundtrip(name):
    cfg = get_preset(name)
    assert isinstance(cfg, BifrostFlowConfig)
    back = BifrostFlowConfig.from_dict(cfg.to_dict())
    assert back.to_dict() == cfg.to_dict()


def test_tiny_is_cpu_and_dummy():
    cfg = get_preset("tiny_cpu")
    assert cfg.train.device == "cpu"
    assert cfg.mllm.backbone == "dummy"
    assert cfg.renderer.backbone == "dummy"


def test_base_uses_real_backbones():
    cfg = get_preset("base_gpu")
    assert "Qwen" in cfg.mllm.backbone
    assert "FLUX" in cfg.renderer.backbone
    assert cfg.mllm.freeze_backbone is True  # core invariant


def test_freeze_backbone_invariant_in_all_presets():
    for name in available_presets():
        assert get_preset(name).mllm.freeze_backbone is True


def test_adaptive_depth_enabled_by_default():
    assert TokenizerConfig().adaptive_depth is True


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        BifrostFlowConfig.from_dict({"tokenizer": {"not_a_field": 1}})


@pytest.mark.parametrize("fname", ["tiny_cpu.yaml", "base_gpu.yaml"])
def test_yaml_configs_load(fname):
    path = os.path.join(CONFIG_DIR, fname)
    cfg = BifrostFlowConfig.from_yaml(path)
    assert isinstance(cfg, BifrostFlowConfig)
    # YAML matches the corresponding preset.
    preset_name = "tiny_cpu" if "tiny" in fname else "base_gpu"
    assert cfg.to_dict() == get_preset(preset_name).to_dict()
