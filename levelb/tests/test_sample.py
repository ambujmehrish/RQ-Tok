"""levelb.sample end to end on tiny upstream-class Qwen + tiny FLUX (code test only)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch

from levelb import bifrost_build as bb
from levelb import controlnet_build as cb
from levelb.flux_text import bf16_bits, encode
from levelb.sample import mar_decode, merge

from .test_bifrost_build import HIDDEN, tiny_vlm  # noqa: F401  (fixture)
from .test_controlnet import tiny_flux, transformer  # noqa: F401  (fixtures)


def _setup(tmp, vlm, flux, tr):
    data = tmp / "data"
    data.mkdir()
    names = [f"class {i}" for i in range(1000)]
    (data / "class_names.json").write_text(json.dumps(names))
    t5, pooled = encode(flux, names, device="cpu", batch=250)
    np.save(data / "flux_t5.npy", bf16_bits(t5))
    np.save(data / "flux_pooled.npy", bf16_bits(pooled))
    (data / "meta.json").write_text(json.dumps({"rows": 1, "latent_width": HIDDEN}))

    model = bb.build_model(vlm, inner_dim=HIDDEN)
    bb.set_trainable(model)
    branch = tmp / "branch.pt"
    torch.save({n: p.detach() for n, p in model.named_parameters() if p.requires_grad}, branch)

    cdir = tmp / "cn"
    cdir.mkdir()
    cn = cb.build_controlnet(tr, HIDDEN, num_double=1, num_single=1)
    torch.save(cn.state_dict(), cdir / "final_controlnet.pt")
    (cdir / "config.json").write_text(json.dumps({"num_double": 1, "num_single": 1}))
    return data, branch, cdir / "final_controlnet.pt"


def _run(data, branch, cn, vlm, flux, out, seed=0):
    cmd = [sys.executable, "-m", "levelb.sample", "--arm", "bifrost", "--branch", str(branch),
           "--controlnet", str(cn), "--data", str(data), "--vlm", vlm, "--flux", flux,
           "--n", "3", "--seed", str(seed), "--batch", "2", "--steps", "2", "--res", "32",
           "--device", "cpu", "--out", str(out)]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def test_mar_reveals_one_token_per_step(tiny_vlm):  # noqa: F811
    model = bb.build_model(tiny_vlm, inner_dim=HIDDEN).float().eval()
    conv = bb.make_conversation()
    tokens, revealed = mar_decode(model, conv, ["tench", "goldfish"], np_seed=0)
    assert revealed == [1] * bb.NUM_TOKENS
    assert tokens.shape == (2, bb.NUM_TOKENS, HIDDEN)
    # every position was written: no mask token left anywhere
    assert not torch.isclose(tokens, model.mask_token.expand_as(tokens)).all(-1).any()


def test_sample_end_to_end_and_reproducible(tiny_vlm, tiny_flux, transformer, tmp_path):  # noqa: F811
    data, branch, cn = _setup(tmp_path, tiny_vlm, tiny_flux, transformer)
    a, b = tmp_path / "a", tmp_path / "b"
    _run(data, branch, cn, tiny_vlm, tiny_flux, a)
    _run(data, branch, cn, tiny_vlm, tiny_flux, b)
    za, zb = np.load(a / "shard00of01.npz"), np.load(b / "shard00of01.npz")
    assert za["arr_0"].shape == (3, 32, 32, 3) and za["arr_0"].dtype == np.uint8
    assert np.array_equal(za["arr_0"], zb["arr_0"])                       # same seed
    assert za["labels"].tolist() == np.random.default_rng(0).integers(0, 1000, 3).tolist()
    path = merge(str(a), nshards=1, n=3)
    assert np.load(path)["arr_0"].shape[0] == 3
