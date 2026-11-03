#!/usr/bin/env python3
"""Run ELLA's official DPG-Bench scorer, loading mPLUG-large from a local copy.

    accelerate launch --num_processes 1 --mixed_precision fp16 scripts/bench/dpg_compute.py \
        --mplug-dir DIR --ella-dir DIR -- <compute_dpg_bench.py arguments>

ELLA's ``compute_dpg_bench.py`` hardcodes the ModelScope id
``damo/mplug_visual-question-answering_coco_large_en``. The ModelScope version that matches
the benchmark's release (1.13.3) can no longer download from today's ModelScope hub (it
rejects responses without a content length), so the same model is downloaded once with a
current ModelScope CLI and loaded from disk. That path is the ONLY change (deviation D1):
the scoring code, questions, dependency rule and averaging are ELLA's, imported unmodified.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mplug-dir", required=True,
                    help="local copy of damo/mplug_visual-question-answering_coco_large_en")
    ap.add_argument("--ella-dir", required=True, help="ELLA checkout (pinned commit)")
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    mplug = pathlib.Path(args.mplug_dir)
    for f in ("configuration.json", "pytorch_model.bin", "config_bert.json"):
        if not (mplug / f).is_file():
            raise SystemExit(f"{mplug} is not a complete mPLUG-large copy (missing {f})")
    script = pathlib.Path(args.ella_dir) / "dpg_bench" / "compute_dpg_bench.py"
    if not script.is_file():
        raise SystemExit(f"{script} not found")

    spec = importlib.util.spec_from_file_location("compute_dpg_bench", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # D1: same model, loaded from disk instead of the hardcoded hub id.
    original_init = mod.MPLUG.__init__

    def init_from_disk(self, ckpt=None, device="gpu"):
        original_init(self, ckpt=str(mplug), device=device)

    mod.MPLUG.__init__ = init_from_disk
    sys.argv = [str(script), *rest]
    mod.main()


if __name__ == "__main__":
    main()
