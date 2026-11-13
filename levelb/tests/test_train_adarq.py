"""levelb.train_adarq: Stage 0 -> AdaRQ Stage 1 steps, held-out eval, resume (code test)."""

from __future__ import annotations

import json
import subprocess
import sys

import torch

from levelb.flux_text import bf16_bits

from .test_adarq_tokenizer import _latents
from .test_bifrost_build import HIDDEN, tiny_vlm  # noqa: F401  (fixture)
from .test_train_branch import _data


def test_adarq_train_resume(tiny_vlm, tmp_path):  # noqa: F811
    data, tok, out = tmp_path / "data", tmp_path / "tok", tmp_path / "run"
    data.mkdir()
    _data(data)
    # compressible latents (easy and hard patches), so a mean depth of 2 is reachable
    import numpy as np
    np.save(data / "latents_train.npy", bf16_bits(_latents(n=24, width=HIDDEN)))
    subprocess.run([sys.executable, "-m", "levelb.adarq_tokenizer", "--data", str(data),
                    "--out", str(tok), "--codebook-size", "16", "--steps", "20",
                    "--images-per-step", "8", "--val-images", "4", "--target-mean-depth", "2.5",
                    "--device", "cpu"],
                   check=True, capture_output=True, text=True)
    base = [sys.executable, "-m", "levelb.train_adarq", "--data", str(data), "--tok", str(tok),
            "--out", str(out), "--vlm", tiny_vlm, "--batch", "4", "--warmup", "2",
            "--epochs", "1", "--save-every", "2", "--eval-every", "2", "--eval-n", "4",
            "--flow-width", "32", "--flow-depth", "2", "--micro-batch", "2", "--device", "cpu"]
    r = subprocess.run(base + ["--max-steps", "3"], check=True, capture_output=True, text=True)
    assert "stopped at step 3/5" in r.stdout
    r = subprocess.run(base, check=True, capture_output=True, text=True)
    assert "[resume] step 3" in r.stdout and "finished 5 steps" in r.stdout
    heads = torch.load(out / "final_heads.pt")
    assert heads["flow.out.weight"].abs().sum() > 0               # left its zero init
    trainable = torch.load(out / "final_trainable.pt")
    assert not any(k.startswith("vision_head.") for k in trainable)   # unused by this arm
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["adarq_heads"] > 0 and 0 < cfg["tau"] < 1
    recs = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert any("heldout_code_ce" in r for r in recs)


def test_heldout_eval_does_not_touch_training_rng(tiny_vlm, tmp_path, monkeypatch):  # noqa: F811
    """--seed fixes the initial weights and evals reseed their own stream only: two runs
    without evals and one with an eval every step are bit-identical (the flow loss draws t
    and noise from the global torch RNG)."""
    # one CPU thread: multithreaded reductions can differ by an ulp between identical runs
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    data, tok = tmp_path / "data", tmp_path / "tok"
    data.mkdir()
    _data(data)
    import numpy as np
    np.save(data / "latents_train.npy", bf16_bits(_latents(n=24, width=HIDDEN)))
    subprocess.run([sys.executable, "-m", "levelb.adarq_tokenizer", "--data", str(data),
                    "--out", str(tok), "--codebook-size", "16", "--steps", "20",
                    "--images-per-step", "8", "--val-images", "4", "--target-mean-depth", "2.5",
                    "--device", "cpu"],
                   check=True, capture_output=True, text=True)
    states = []
    for name, every in (("a", "1000"), ("b", "1000"), ("c", "1")):
        out = tmp_path / name
        subprocess.run([sys.executable, "-m", "levelb.train_adarq", "--data", str(data),
                        "--tok", str(tok), "--out", str(out), "--vlm", tiny_vlm, "--batch", "4",
                        "--warmup", "2", "--epochs", "1", "--save-every", "1000",
                        "--eval-every", every, "--eval-n", "4", "--flow-width", "32",
                        "--flow-depth", "2", "--micro-batch", "2", "--device", "cpu",
                        "--max-steps", "3"], check=True, capture_output=True, text=True)
        states.append(torch.load(out / "ckpt.pt", weights_only=False))
    for other in states[1:]:
        for part in ("heads", "trainable"):
            a, b = states[0][part], other[part]
            assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), part


def test_total_steps_sets_the_budget(tiny_vlm, tmp_path):  # noqa: F811
    """--total-steps overrides --epochs: the schedule ends and the final checkpoint is written
    at that step (the validation-tuned AdaRQ-Flow run)."""
    data, tok, out = tmp_path / "data", tmp_path / "tok", tmp_path / "run"
    data.mkdir()
    _data(data)
    import numpy as np
    np.save(data / "latents_train.npy", bf16_bits(_latents(n=24, width=HIDDEN)))
    subprocess.run([sys.executable, "-m", "levelb.adarq_tokenizer", "--data", str(data),
                    "--out", str(tok), "--codebook-size", "16", "--steps", "20",
                    "--images-per-step", "8", "--val-images", "4", "--target-mean-depth", "2.5",
                    "--device", "cpu"],
                   check=True, capture_output=True, text=True)
    r = subprocess.run([sys.executable, "-m", "levelb.train_adarq", "--data", str(data),
                        "--tok", str(tok), "--out", str(out), "--vlm", tiny_vlm, "--batch", "4",
                        "--warmup", "2", "--epochs", "16", "--total-steps", "3",
                        "--save-every", "1000", "--eval-every", "1000", "--eval-n", "4",
                        "--flow-width", "32", "--flow-depth", "2", "--micro-batch", "2",
                        "--device", "cpu"], check=True, capture_output=True, text=True)
    assert "finished 3 steps" in r.stdout and (out / "final_heads.pt").exists()
    assert json.loads((out / "config.json").read_text())["steps_total"] == 3
    st = torch.load(out / "ckpt.pt", weights_only=False)
    assert st["step"] == 3 and st["sched"]["last_epoch"] == 3


def test_soft_targets_train_end_to_end(tiny_vlm, tmp_path):  # noqa: F811
    """--soft-targets: temperatures measured on the train split, recorded, training runs and
    logs both the soft and the hard held-out code CE."""
    data, tok, out = tmp_path / "data", tmp_path / "tok", tmp_path / "run"
    data.mkdir()
    _data(data)
    import numpy as np
    np.save(data / "latents_train.npy", bf16_bits(_latents(n=24, width=HIDDEN)))
    subprocess.run([sys.executable, "-m", "levelb.adarq_tokenizer", "--data", str(data),
                    "--out", str(tok), "--codebook-size", "16", "--steps", "20",
                    "--images-per-step", "8", "--val-images", "4", "--target-mean-depth", "2.5",
                    "--device", "cpu"],
                   check=True, capture_output=True, text=True)
    r = subprocess.run([sys.executable, "-m", "levelb.train_adarq", "--data", str(data),
                        "--tok", str(tok), "--out", str(out), "--vlm", tiny_vlm, "--batch", "4",
                        "--warmup", "2", "--total-steps", "3", "--soft-targets",
                        "--save-every", "1000", "--eval-every", "1", "--eval-n", "4",
                        "--flow-width", "32", "--flow-depth", "2", "--micro-batch", "2",
                        "--device", "cpu"], check=True, capture_output=True, text=True)
    assert "soft code targets" in r.stdout and "finished 3 steps" in r.stdout
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["soft_temps"] and all(t > 0 for t in cfg["soft_temps"])
    recs = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    held = [r for r in recs if "heldout_code_ce_hard" in r]
    assert held and all(np.isfinite(r["heldout_code_ce"]) for r in held)
