"""levelb.oracle_render: bridge inputs per mode, and the CLI on tiny FLUX (code test only)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from levelb import adarq_tokenizer as at
from levelb import controlnet_build as cb
from levelb.class_accuracy import heldout_rows
from levelb.flux_text import bf16_bits, encode
from levelb.oracle_render import bridge_latents, load_quantizers

from .test_controlnet import tiny_flux, transformer  # noqa: F401  (fixtures)

WIDTH = 16


def _tokenizer(tmp, lat: torch.Tensor):
    q = at.make_quantizer(WIDTH, 32, 4)
    flat = lat.reshape(-1, WIDTH)
    torch.manual_seed(0)
    for _ in range(40):
        q.train()(flat[torch.randperm(len(flat))[:1024]], update_codebook=True)
    tau = at.choose_tau(at.residual_ratios(q.eval(), flat), 2.0)
    tok = tmp / "tok"
    tok.mkdir()
    qa = at.make_quantizer(WIDTH, 32, 4, tau=tau, adaptive=True)
    qa.load_state_dict(q.state_dict())
    codes = qa.eval()(flat, update_codebook=False).codes.reshape(len(lat), 64, 4)
    np.save(tok / "codes_train.npy", codes.numpy().astype(np.int16))
    torch.save(
        {
            "state_dict": qa.state_dict(),
            "tau": tau,
            "res_rms": 1.0,
            "codebook_size": 32,
            "max_depth": 4,
            "width": WIDTH,
        },
        tok / "tokenizer.pt",
    )
    return tok


def _latents(n: int) -> torch.Tensor:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((32, WIDTH)) * 3
    x = centers[rng.integers(0, 32, (n, 64))] + 0.5 * rng.standard_normal((n, 64, WIDTH))
    # as stored: bf16 bit patterns
    return torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).float()


def test_bridge_latents_modes(tmp_path):
    z = _latents(8)
    qa, qf = load_quantizers(_tokenizer(tmp_path, z))
    same, depth, codes = bridge_latents(z, qa, qf, "latent")
    assert same is z and depth is None and codes is None
    err = {}
    for mode in ("depth1", "depth2", "depth3", "depth4", "adaptive"):
        lat, depth, codes = bridge_latents(z, qa, qf, mode)
        assert lat.shape == z.shape and depth.shape == z.shape[:2]
        err[mode] = float((z - lat).pow(2).sum())
    assert err["depth1"] > err["depth2"] > err["depth3"] > err["depth4"]  # more codes, closer
    assert err["depth4"] <= err["adaptive"] <= err["depth1"]
    # full depth equals the quantizer's own dequantized prefix
    full = qf(z.reshape(-1, WIDTH), update_codebook=False).zhat.reshape(z.shape)
    assert torch.allclose(bridge_latents(z, qa, qf, "depth4")[0], full, atol=1e-5)
    with pytest.raises(ValueError):
        bridge_latents(z, qa, qf, "depth5")
    with pytest.raises(ValueError):
        bridge_latents(z, qa, qf, "codes")


def test_cli_end_to_end(tiny_flux, transformer, tmp_path):  # noqa: F811
    n_rows = 1100  # 1000 held-out (one per class) + 100 train
    lat = _latents(n_rows)
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "latents_train.npy", bf16_bits(lat))
    labels = np.concatenate([np.arange(1000), np.arange(100)])
    np.save(data / "labels_train.npy", labels.astype(np.int16))
    np.savez(data / "split_seed0.npz", train=np.arange(1000, n_rows), heldout=np.arange(1000))
    names = [f"class {i}" for i in range(1000)]
    t5, pooled = encode(tiny_flux, names, device="cpu", batch=250)
    np.save(data / "flux_t5.npy", bf16_bits(t5))
    np.save(data / "flux_pooled.npy", bf16_bits(pooled))
    (data / "meta.json").write_text(json.dumps({"rows": n_rows, "latent_width": WIDTH}))
    tok = _tokenizer(tmp_path, lat)
    cdir = tmp_path / "cn"
    cdir.mkdir()
    torch.save(
        cb.build_controlnet(transformer, WIDTH, num_double=1, num_single=1).state_dict(),
        cdir / "final_controlnet.pt",
    )
    (cdir / "config.json").write_text(json.dumps({"num_double": 1, "num_single": 1}))

    rows = heldout_rows(data, 1000)
    assert rows.tolist() == list(range(1000))
    metas = {}
    for mode in ("latent", "adaptive", "depth4"):
        out = tmp_path / mode
        cmd = [
            sys.executable,
            "-m",
            "levelb.oracle_render",
            "--mode",
            mode,
            "--tok",
            str(tok),
            "--controlnet",
            str(cdir / "final_controlnet.pt"),
            "--data",
            str(data),
            "--flux",
            tiny_flux,
            "--n",
            "1000",
            "--nshards",
            "500",
            "--shard",
            "1",
            "--batch",
            "2",
            "--steps",
            "2",
            "--res",
            "32",
            "--device",
            "cpu",
            "--out",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        z = np.load(out / "shard01of500.npz")
        assert z["arr_0"].shape == (2, 32, 32, 3) and z["arr_0"].dtype == np.uint8
        assert z["labels"].tolist() == [2, 3] and z["rows"].tolist() == [2, 3]
        metas[mode] = json.loads((out / "shard01of500.json").read_text())
    assert metas["latent"]["energy_unexplained"] == 0.0
    assert metas["latent"]["stored_code_agreement"] is None
    assert metas["adaptive"]["stored_code_agreement"] == 1.0
    assert 1.0 <= metas["adaptive"]["mean_depth"] <= 4.0 and metas["depth4"]["mean_depth"] == 4.0
    assert 0 < metas["depth4"]["energy_unexplained"] <= metas["adaptive"]["energy_unexplained"]
