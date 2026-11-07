#!/usr/bin/env python3
"""Training status of the three Level B runs, from their metrics.jsonl (all jobs so far).

    python scripts/levelb/status.py            # LEVELB_ROOT defaults to ~/work/levelb

Per run: progress, speed, time left (and 24 h jobs left), the latest training loss, and
the held-out curve. Held-out numbers are for monitoring training dynamics only; the arms
are compared on the test set after sampling.
"""

from __future__ import annotations

import json
import math
import os
import pathlib

RUNS = [("ControlNet", "controlnet", ["heldout_flow_loss"]),
        ("Bifrost-1 branch", "branch_bifrost", ["heldout_mse"]),
        ("AdaRQ-Flow branch", "branch_adarq", ["heldout_loss", "heldout_code_ce", "heldout_flow"])]
JOB_HOURS = 23.8          # usable hours of a 24 h job (load + checkpoint at the end)


def records(path: pathlib.Path) -> tuple[dict[int, dict], dict[int, dict]]:
    """Last record per step (a resume replays the steps after the last checkpoint)."""
    train, held = {}, {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        (held if any(k.startswith("heldout") for k in r) else train)[int(r["step"])] = r
    return train, held


def main() -> None:
    root = pathlib.Path(os.environ.get("LEVELB_ROOT", pathlib.Path.home() / "work" / "levelb"))
    for name, sub, keys in RUNS:
        out = root / sub
        print(f"=== {name}  ({out})")
        if not (out / "metrics.jsonl").exists():
            print("  not started")
            continue
        total = json.loads((out / "config.json").read_text())["steps_total"]
        train, held = records(out / "metrics.jsonl")
        if (out / "final_trainable.pt").exists() or (out / "final_controlnet.pt").exists():
            print(f"  FINISHED ({total} steps)")
        if train:
            last = train[max(train)]
            step, rate = last["step"], last["steps_per_s"]
            left_h = (total - step) / rate / 3600 if rate > 0 else math.inf
            extra = "".join(f" {k} {last[k]:.4f}" for k in ("code_ce", "flow") if k in last)
            print(f"  step {step}/{total} ({100 * step / total:.1f}%) epoch {last['epoch']} | "
                  f"{rate:.3f} it/s | ~{left_h / 24:.1f} days left "
                  f"(~{math.ceil(left_h / JOB_HOURS)} more 24 h jobs)")
            print(f"  train loss {last['loss']:.4f}{extra} | grad norm {last['grad_norm']:.3f} | "
                  f"lr {last['lr']:.2e}")
        if held:
            print("  held-out:  step  " + "  ".join(f"{k[8:]:>9s}" for k in keys))
            prev = None
            for s in sorted(held):
                vals = [held[s][k] for k in keys]
                trend = "" if prev is None else f"  ({100 * (vals[0] - prev) / prev:+.2f}%)"
                print(f"        {s:>8d}  " + "  ".join(f"{v:9.5f}" for v in vals) + trend)
                prev = vals[0]
        print()


if __name__ == "__main__":
    main()
