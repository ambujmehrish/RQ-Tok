"""levelb.latent_stats on small synthetic latents with a class-dependent mean."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch

from levelb.flux_text import bf16_bits
from levelb.latent_stats import pick_train

from .test_oracle_render import WIDTH, _tokenizer

CLASSES = 20


def test_pick_train_is_balanced_and_seeded():
    labels = np.arange(100) % 5
    train = np.arange(10, 100)
    a = pick_train(train, labels, 3, seed=0)
    assert len(a) == 15 and np.bincount(labels[a]).tolist() == [3] * 5
    assert np.array_equal(a, pick_train(train, labels, 3, seed=0))
    assert set(a) <= set(train.tolist())


def test_cli_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    n = 30 * CLASSES
    labels = np.arange(n) % CLASSES
    means = rng.standard_normal((CLASSES, WIDTH)) * 2
    x = 3.0 + means[labels][:, None] + rng.standard_normal((n, 64, WIDTH))
    lat = torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).float()
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "latents_train.npy", bf16_bits(lat))
    np.save(data / "labels_train.npy", labels.astype(np.int16))
    np.savez(data / "split_seed0.npz", train=np.arange(100, n), heldout=np.arange(100))
    tok = _tokenizer(tmp_path, lat)
    out = tmp_path / "stats.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "levelb.latent_stats",
            "--data",
            str(data),
            "--tok",
            str(tok),
            "--out",
            str(out),
            "--per-class",
            "20",
            "--probe-epochs",
            "30",
            "--batch",
            "64",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    r = json.loads(out.read_text())
    assert r["train_images"] == 20 * CLASSES and r["heldout_images"] == 100
    assert r["patches"] == 20 * CLASSES * 64 and r["width"] == WIDTH
    # mean 3 per element against unit noise and class spread 2: the global mean dominates
    assert 0.5 < r["share_global_mean"] < 0.8
    # between-class variance: 4 / (4 + 1) per element
    assert abs(r["share_between_class_of_variance"] - 0.8) < 0.08
    cu = r["codes_unexplained"]
    assert cu["depth1"]["of_energy"] > cu["adaptive"]["of_energy"] >= cu["depth4"]["of_energy"] > 0
    # within-class covariance is ~isotropic: effective dimension near the width
    assert r["spectrum_within_class"]["participation_ratio"] > 0.8 * WIDTH
    assert (
        r["spectrum_total"]["participation_ratio"]
        < r["spectrum_within_class"]["participation_ratio"]
    )
    assert r["probe"]["latent"]["top1"] > 0.9  # classes are linearly separable
    assert set(r["probe"]) == {"latent", "depth1", "adaptive", "depth4", "residual"}
