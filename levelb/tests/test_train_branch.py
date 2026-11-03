"""levelb.train_branch: steps, held-out eval, checkpoint/resume (tiny model, code test)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch

from levelb import bifrost_build as bb

from .test_bifrost_build import HIDDEN, tiny_vlm  # noqa: F401  (fixture)


def _data(root, n=24, width=HIDDEN):
    rng = np.random.default_rng(0)
    lat = torch.from_numpy(rng.standard_normal((n, 64, width)).astype(np.float32))
    lat = lat.to(torch.bfloat16)
    np.save(root / "latents_train.npy", lat.view(torch.int16).numpy().view(np.uint16))
    np.save(root / "labels_train.npy", rng.integers(0, 1000, n).astype(np.int16))
    (root / "class_names.json").write_text(json.dumps([f"class {i}" for i in range(1000)]))
    np.savez(root / "split_seed0.npz", train=np.arange(4, n), heldout=np.arange(4), seed=0)
    (root / "meta.json").write_text(json.dumps({"rows": n, "latent_width": width}))


def _run(data, out, vlm, *extra):
    cmd = [sys.executable, "-m", "levelb.train_branch", "--data", str(data), "--out", str(out),
           "--vlm", vlm, "--batch", "4", "--warmup", "2", "--epochs", "1", "--save-every", "2",
           "--eval-every", "2", "--eval-n", "4", "--device", "cpu", *extra]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def test_train_resume_and_freeze(tiny_vlm, tmp_path):  # noqa: F811
    data, out = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    _data(data)
    r = _run(data, out, tiny_vlm, "--max-steps", "3")          # 20 train rows / 4 = 5 steps
    assert "stopped at step 3/5" in r.stdout
    state = torch.load(out / "ckpt.pt", weights_only=False)
    assert state["step"] == 3
    r = _run(data, out, tiny_vlm)
    assert "[resume] step 3" in r.stdout and "finished 5 steps" in r.stdout

    final = torch.load(out / "final_trainable.pt")
    fresh = bb.build_model(tiny_vlm, inner_dim=HIDDEN).float()
    named = dict(fresh.named_parameters())
    moved = [n for n, t in final.items() if not torch.equal(t.float(), named[n].detach().float())]
    assert any("vision_branch_q_proj" in n for n in moved) and "vision_head.weight" in moved
    assert all(("vision_branch_" in n) or n.startswith(("vision_head.", "mask_token",
                                                        "learnable_pos_emb")) for n in final)
    recs = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert any("heldout_mse" in r for r in recs)
    assert all(np.isfinite(r.get("heldout_mse", 0.0)) for r in recs)
