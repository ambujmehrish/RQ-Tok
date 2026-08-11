"""Real Qwen2.5-VL adapter.

Structural tests always run. Tests needing the checkpoint skip when it is not cached,
so CI stays green without a 7 GB download — but they are the ones that actually proved
the adapter (see the `qwen2_5_vl_3b` preset docstring for the verified geometry).
"""

import dataclasses

import pytest

from adarq_flow.config import MLLMConfig, TokenizerConfig, get_preset

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from adarq_flow.mllm import build_backbone, is_qwen_vl
from adarq_flow.tokenizer.encoder import build_clip_encoder

REAL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def _config_available(model_id: str = REAL_ID) -> bool:
    """True when the checkpoint config is reachable (cache or network)."""
    try:
        from adarq_flow.mllm.qwen import load_qwen_config

        load_qwen_config(model_id)
        return True
    except Exception:
        return False


needs_ckpt = pytest.mark.skipif(
    not _config_available(), reason=f"{REAL_ID} config not cached/reachable")


# -- routing (no weights needed) --------------------------------------------------------
@pytest.mark.parametrize("model_id,expected", [
    ("Qwen/Qwen2.5-VL-3B-Instruct", True),
    ("Qwen/Qwen2.5-VL-7B-Instruct", True),
    ("Qwen/Qwen2-VL-7B-Instruct", True),
    ("openai/clip-vit-base-patch32", False),
    ("dummy", False),
])
def test_qwen_routing(model_id, expected):
    assert is_qwen_vl(model_id) is expected


def test_unknown_backbone_still_raises():
    """An unrecognised id must not silently fall back to the stand-in."""
    with pytest.raises(NotImplementedError, match="Refusing to substitute"):
        build_backbone(MLLMConfig(backbone="some/unknown-model"))
    with pytest.raises(NotImplementedError, match="Refusing to substitute"):
        build_clip_encoder("some/unknown-model", TokenizerConfig())


# -- the preset encodes the VERIFIED geometry ------------------------------------------
def test_qwen_preset_geometry():
    cfg = get_preset("qwen2_5_vl_3b")
    # Verified against the real checkpoint: text hidden 2048, post-merger visual 2048,
    # 64 visual tokens per image at a 16x16 grid (2x2 merged).
    assert cfg.mllm.backbone == REAL_ID
    assert cfg.mllm.hidden_dim == 2048
    assert cfg.tokenizer.clip_dim == 2048
    assert cfg.tokenizer.num_patches == 64
    # allocation prerequisites (ISSUES.md D1/D2)
    assert cfg.tokenizer.shared_codebook is False
    assert cfg.tokenizer.include_zero_code is True


def test_qwen_preset_removes_mllm_from_dummy_components():
    """The MLLM is now real; only the renderer/data stand-ins should remain."""
    cfg = get_preset("qwen2_5_vl_3b")
    dummies = cfg.dummy_components()
    assert "mllm.backbone" not in dummies
    assert set(dummies) == {"renderer.backbone", "renderer.vae", "train.dataset"}


# -- geometry read from the checkpoint, never hardcoded --------------------------------
@needs_ckpt
def test_qwen_dims_match_the_preset():
    from adarq_flow.mllm.qwen import qwen_dims

    d = qwen_dims(REAL_ID)
    cfg = get_preset("qwen2_5_vl_3b")
    assert d["text_hidden"] == cfg.mllm.hidden_dim
    assert d["visual_out"] == cfg.tokenizer.clip_dim
    # The tower's internal width differs from what the LLM consumes; quantizing the
    # former would break the MLLM-native property.
    assert d["vision_hidden"] != d["visual_out"]


@needs_ckpt
def test_dim_mismatch_raises_rather_than_projecting():
    from adarq_flow.mllm.qwen import QwenVisionEncoder

    bad = dataclasses.replace(get_preset("qwen2_5_vl_3b").tokenizer, clip_dim=999)
    with pytest.raises(ValueError, match="does not match"):
        QwenVisionEncoder(REAL_ID, bad)


@needs_ckpt
def test_unfrozen_backbone_is_rejected():
    from adarq_flow.mllm.qwen import QwenBackbone

    cfg = dataclasses.replace(get_preset("qwen2_5_vl_3b").mllm, freeze_backbone=False)
    with pytest.raises(ValueError, match="freeze_backbone"):
        QwenBackbone(cfg)
