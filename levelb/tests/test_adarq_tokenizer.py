"""AdaRQ-Flow Stage 0 on small structured latents (code test only)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch

from levelb import adarq_tokenizer as at
from levelb.flux_text import bf16_bits


def _latents(n=80, width=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((32, width)) * 3
    spread = np.where(np.arange(32) < 16, 0.05, 1.0)          # easy and hard patches
    a = rng.integers(0, 32, size=(n, 64))
    x = centers[a] + spread[a][..., None] * rng.standard_normal((n, 64, width))
    return torch.from_numpy(x.astype(np.float32))


def test_prefix_nesting_and_depth_rule():
    z = _latents().reshape(-1, 16)
    q = at.make_quantizer(16, 32, 4)
    for _ in range(60):
        q.train()(z[torch.randperm(len(z))[:1024]], update_codebook=True)
    q.eval()
    full = q(z, update_codebook=False).codes
    ratios = at.residual_ratios(q, z)
    tau = at.choose_tau(ratios, 2.0)
    qa = at.make_quantizer(16, 32, 4, tau=tau, adaptive=True)
    qa.load_state_dict(q.state_dict())
    out = qa.eval()(z, update_codebook=False)
    d = at.depth_at(ratios, tau)
    assert torch.equal(out.depths, d)                        # our rule == quantizer's rule
    assert abs(d.float().mean().item() - 2.0) < 0.1
    for k in range(4):                                       # prefix codes are identical
        live = d > k
        assert torch.equal(out.codes[live, k], full[live, k])
        assert (out.codes[~live, k] == 32).all()             # <halt> suffix


def test_cli_end_to_end(tmp_path):
    lat = _latents(n=60)
    np.save(tmp_path / "latents_train.npy", bf16_bits(lat))
    np.savez(tmp_path / "split_seed0.npz", train=np.arange(20, 60), heldout=np.arange(20),
             seed=0)
    (tmp_path / "meta.json").write_text(json.dumps({"rows": 60, "latent_width": 16}))
    out = tmp_path / "tok"
    subprocess.run([sys.executable, "-m", "levelb.adarq_tokenizer", "--data", str(tmp_path),
                    "--out", str(out), "--codebook-size", "32", "--steps", "40",
                    "--images-per-step", "16", "--val-images", "20", "--device", "cpu"],
                   check=True, capture_output=True, text=True)
    codes = np.load(out / "codes_train.npy")
    meta = json.loads((out / "tokenizer.json").read_text())
    assert codes.shape == (60, 64, 4) and codes.dtype == np.int16
    assert abs(meta["val_mean_depth"] - 2.0) < 0.15 and 0 < meta["tau"] < 1
    assert meta["res_rms"] > 0
