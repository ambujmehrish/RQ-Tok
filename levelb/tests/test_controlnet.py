"""Level B Stage 2 mechanics on diffusers' tiny FLUX pipe (real classes, tiny weights).

A code test, not an experiment. Checks the ControlNet starts as upstream's recipe says,
that at initialisation it leaves FLUX's prediction unchanged (zero residuals), the text
precompute, and the trainer's step / held-out eval / resume.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from levelb import controlnet_build as cb
from levelb.flux_text import bf16_bits, encode

TINY_FLUX = "hf-internal-testing/tiny-flux-pipe"
WIDTH = 8                        # bridge latent width in the tiny test (2048 for Qwen-3B)


def build_controlnet(tr, width):   # tiny FLUX has 1 double / 1 single block (real: 19 / 38)
    return cb.build_controlnet(tr, width, num_double=1, num_single=1)


@pytest.fixture(scope="module")
def tiny_flux():
    from huggingface_hub import snapshot_download

    return snapshot_download(TINY_FLUX)


@pytest.fixture(scope="module")
def transformer(tiny_flux):
    from diffusers import FluxTransformer2DModel

    return FluxTransformer2DModel.from_pretrained(tiny_flux, subfolder="transformer").eval()


def _inputs(tr, b=2, tokens=256, gen=0):
    g = torch.Generator().manual_seed(gen)
    c = tr.config
    from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline

    return dict(
        hidden_states=torch.randn(b, tokens, c.in_channels, generator=g),
        timestep=torch.full((b,), 0.5),
        pooled_projections=torch.randn(b, c.pooled_projection_dim, generator=g),
        encoder_hidden_states=torch.randn(b, 7, c.joint_attention_dim, generator=g),
        txt_ids=torch.zeros(7, 3),
        img_ids=FluxControlNetPipeline._prepare_latent_image_ids(b, 16, 16, "cpu", torch.float32),
        guidance=None)


def test_controlnet_starts_as_upstream_recipe(transformer):
    cn = build_controlnet(transformer, WIDTH)           # verify_init raises on any deviation
    assert cn.controlnet_x_embedder.in_features == WIDTH
    assert cn.up_sampling_block.upconv.in_channels == WIDTH
    assert len(cn.transformer_blocks) == 1 and len(cn.single_transformer_blocks) == 1
    with pytest.raises(ValueError, match="cannot copy"):     # 4 double > tiny FLUX's 1
        cb.build_controlnet(transformer, WIDTH)


@torch.no_grad()
def test_zero_init_leaves_flux_unchanged(transformer):
    cn = build_controlnet(transformer, WIDTH).eval()
    x = _inputs(transformer)
    control = torch.randn(2, 64, WIDTH)
    blocks, singles = cn(controlnet_cond=control, return_dict=False, **x)
    assert all(torch.count_nonzero(s) == 0 for s in [*blocks, *singles])
    plain = transformer(**x, return_dict=False)[0]
    steered = transformer(**x, controlnet_block_samples=blocks,
                          controlnet_single_block_samples=singles, return_dict=False)[0]
    assert torch.equal(plain, steered)


def test_text_precompute(tiny_flux):
    a, pa = encode(tiny_flux, ["tench", "goldfish"], device="cpu")
    b, pb = encode(tiny_flux, ["tench", "goldfish"], device="cpu")
    assert a.shape[:2] == (2, 512) and pa.shape[0] == 2
    assert torch.equal(a, b) and torch.equal(pa, pb)


def _data(root, tiny_flux, n=24, res=32):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 255, res)[None, None, :, None] * np.ones((n, res, 1, 3))
    crops = (x + rng.integers(0, 60, size=(n, res, res, 3))).clip(0, 255).astype(np.uint8)
    np.save(root / "crops_train.npy", crops)
    lat = torch.from_numpy(rng.standard_normal((n, 64, WIDTH)).astype(np.float32))
    np.save(root / "latents_train.npy", bf16_bits(lat))
    np.save(root / "labels_train.npy", rng.integers(0, 1000, n).astype(np.int16))
    names = [f"class {i}" for i in range(1000)]
    (root / "class_names.json").write_text(json.dumps(names))
    t5, pooled = encode(tiny_flux, names, device="cpu", batch=250)
    np.save(root / "flux_t5.npy", bf16_bits(t5))
    np.save(root / "flux_pooled.npy", bf16_bits(pooled))
    np.savez(root / "split_seed0.npz", train=np.arange(4, n), heldout=np.arange(4), seed=0)
    (root / "meta.json").write_text(json.dumps({"rows": n, "latent_width": WIDTH}))


def _run(data, out, flux, *extra):
    cmd = [sys.executable, "-m", "levelb.train_controlnet", "--data", str(data),
           "--out", str(out), "--flux", flux, "--batch", "4", "--epochs", "1",
           "--save-every", "2", "--eval-every", "2", "--eval-n", "4", "--device", "cpu",
           "--num-double", "1", "--num-single", "1", *extra]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def test_train_and_resume(tiny_flux, transformer, tmp_path):
    data, out = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    _data(data, tiny_flux)
    r = _run(data, out, tiny_flux, "--max-steps", "3")          # 20 train rows / 4 = 5 steps
    assert "stopped at step 3/5" in r.stdout
    r = _run(data, out, tiny_flux)
    assert "[resume] step 3" in r.stdout and "finished 5 steps" in r.stdout

    final = torch.load(out / "final_controlnet.pt")
    init = build_controlnet(transformer, WIDTH).state_dict()
    moved = [k for k in final if not torch.equal(final[k].float(), init[k].float())]
    assert any(k.startswith("controlnet_x_embedder") for k in moved)     # left zero init
    assert any(k.startswith("up_sampling_block") for k in moved)
    recs = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    vals = [r["heldout_flow_loss"] for r in recs if "heldout_flow_loss" in r]
    assert vals and all(np.isfinite(vals))


def test_heldout_eval_does_not_touch_training_rng(tiny_flux, tmp_path):
    """heldout_loss reseeds for fixed noise; training with and without it is bit-identical."""
    data = tmp_path / "data"
    data.mkdir()
    _data(data, tiny_flux)
    states = []
    for every in ("1", "1000"):
        out = tmp_path / f"run{every}"
        _run(data, out, tiny_flux, "--max-steps", "3", "--eval-every", every,
             "--save-every", "1000")
        states.append(torch.load(out / "ckpt.pt", weights_only=False)["controlnet"])
    a, b = states
    assert all(torch.equal(a[k], b[k]) for k in a)
