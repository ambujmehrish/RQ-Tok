"""Phase 5 tests: end-to-end inference pipeline (torch, CPU)."""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from bifrost_flow.config import get_preset
from bifrost_flow.inference import BifrostFlowPipeline, build_pipeline
from bifrost_flow.training import Trainer


def _text(b=2, t=5):
    return torch.randint(0, 256, (b, t))


def test_pipeline_generate_shapes():
    torch.manual_seed(0)
    cfg = get_preset("tiny_cpu")
    pipe = build_pipeline(cfg)
    out = pipe.generate(_text(2), cfg_scale=3.0, temperature=0.0,
                        decode_steps=4, render_steps=4)
    N = cfg.tokenizer.num_patches
    D = cfg.tokenizer.max_depth
    d = cfg.tokenizer.clip_dim
    assert out.codes.shape == (2, N, D)
    assert out.depths.shape == (2, N)
    assert out.latents.shape == (2, N, d)
    assert out.image_latents.shape == (2, cfg.renderer.num_image_tokens,
                                       cfg.renderer.latent_dim)
    assert int(out.codes.max()) <= cfg.tokenizer.codebook_size  # K == <halt>


def test_pipeline_greedy_codes_deterministic():
    torch.manual_seed(0)
    pipe = build_pipeline(get_preset("tiny_cpu"))
    text = _text(1)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(1)
    a = pipe.generate(text, temperature=0.0, decode_steps=4, render_steps=4, generator=g1)
    b = pipe.generate(text, temperature=0.0, decode_steps=4, render_steps=4, generator=g2)
    assert torch.equal(a.codes, b.codes)
    assert torch.allclose(a.image_latents, b.image_latents)


def test_bad_text_shape_raises():
    pipe = build_pipeline(get_preset("tiny_cpu"))
    with pytest.raises(ValueError):
        pipe.generate(torch.randint(0, 256, (5,)))  # not [B, T]


def test_pipeline_loads_trainer_checkpoints(tmp_path):
    cfg = get_preset("tiny_cpu")
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(
        cfg.train, stage="branch", max_steps=6, batch_size=4, ckpt_dir=str(tmp_path)))
    t = Trainer(cfg, dataset_length=32, setup_dist=False)
    t.train()
    path = t.save_checkpoint(6)

    pipe = BifrostFlowPipeline(cfg)
    stage = pipe.load_stage(path)
    assert stage == "branch"
    out = pipe.generate(_text(2), temperature=0.0, decode_steps=4, render_steps=4)
    assert out.codes.shape[0] == 2
