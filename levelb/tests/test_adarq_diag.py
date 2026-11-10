"""levelb.adarq_diag: runs end to end and reproduces the tokenizer's codes (code test)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from levelb.flux_text import bf16_bits

from .test_adarq_tokenizer import _latents
from .test_bifrost_build import HIDDEN, tiny_vlm  # noqa: F401  (fixture)
from .test_train_branch import _data


def test_diag_end_to_end(tiny_vlm, tmp_path):  # noqa: F811
    data, tok, run = tmp_path / "data", tmp_path / "tok", tmp_path / "run"
    data.mkdir()
    _data(data)
    np.save(data / "latents_train.npy", bf16_bits(_latents(n=24, width=HIDDEN)))
    py = sys.executable
    subprocess.run([py, "-m", "levelb.adarq_tokenizer", "--data", str(data), "--out", str(tok),
                    "--codebook-size", "16", "--steps", "20", "--images-per-step", "8",
                    "--val-images", "4", "--target-mean-depth", "2.5", "--device", "cpu"],
                   check=True, capture_output=True, text=True)
    subprocess.run([py, "-m", "levelb.train_adarq", "--data", str(data), "--tok", str(tok),
                    "--out", str(run), "--vlm", tiny_vlm, "--batch", "4", "--warmup", "2",
                    "--total-steps", "3", "--save-every", "1000", "--eval-every", "1000",
                    "--eval-n", "4", "--flow-width", "32", "--flow-depth", "2",
                    "--micro-batch", "2", "--device", "cpu"],
                   check=True, capture_output=True, text=True)
    r = subprocess.run([py, "-m", "levelb.adarq_diag", "--data", str(data), "--tok", str(tok),
                        "--run", str(run), "--vlm", tiny_vlm, "--n", "4", "--batch", "2",
                        "--device", "cpu"], check=True, capture_output=True, text=True)
    assert "H1" in r.stdout and "H2" in r.stdout
    res = json.loads((run / "diag" / "overfit_diag.json").read_text())
    assert res["weights"] == "final" and res["step"] == 3
    for key, cell in res["cells"].items():
        if key.startswith("gap@"):
            continue
        assert cell["argmin_matches_stored_code"] >= 0.99, key
        assert np.isfinite(cell["ce_all"]) and 0 <= cell["top1_acc_code_levels"] <= 1
    # mask 1.00: every position is a target; same images and masks across ratios per split
    assert res["cells"]["heldout@1.00"]["targets_per_level"][0] == 4 * 64
