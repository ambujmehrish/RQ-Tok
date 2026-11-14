#!/usr/bin/env python3
"""Which evaluation pipeline reproduces the Bifrost-1 paper's ImageNet numbers?

    python scripts/bench/eval_search_report.py        # LEVELB_ROOT defaults to ~/work/levelb

Reads the metrics written by adm_score.sbatch for the FLUX-only samples (and, when
present, the Bifrost-1 samples) under every reference set, and compares them with the
paper's Table 1 rows. A pipeline "reproduces" the paper only if the training-free FLUX row
matches (FID within 5%, sFID within 10%; IS does not depend on the reference).
"""

from __future__ import annotations

import json
import os
import pathlib

PAPER = {"flux": {"FID": 51.20, "sFID": 224.82, "IS": 157.46},
         "bifrost": {"FID": 25.77, "sFID": 53.67, "IS": 98.57}}
PAPER_T4 = {"bifrost": {"FID": 18.64, "sFID": 43.64, "IS": 156.01}}   # Table 4, 64 steps
REFS = ["adm", "val", "adm64", "adm128", "adm512", "valsquash", "heldout"]


def main() -> None:
    root = pathlib.Path(os.environ.get("LEVELB_ROOT", pathlib.Path.home() / "work" / "levelb"))
    for arm in ("flux", "bifrost"):
        print(f"=== {arm} seed 0   paper Table 1: " + "  ".join(
            f"{k} {v}" for k, v in PAPER[arm].items())
            + ("" if arm not in PAPER_T4 else "   | Table 4: " + "  ".join(
                f"{k} {v}" for k, v in PAPER_T4[arm].items())))
        for ref in REFS:
            f = root / "samples" / f"{arm}_seed0" / f"samples_10000.{ref}.metrics.json"
            if not f.exists():
                print(f"  {ref:10s} (not scored)")
                continue
            m = json.loads(f.read_text())
            dev = {k: 100 * (m[k] - PAPER[arm][k]) / PAPER[arm][k] for k in ("FID", "sFID", "IS")}
            ok = abs(dev["FID"]) <= 5 and abs(dev["sFID"]) <= 10
            print(f"  {ref:10s} FID {m['FID']:8.3f} ({dev['FID']:+6.1f}%)  "
                  f"sFID {m['sFID']:8.3f} ({dev['sFID']:+6.1f}%)  IS {m['IS']:8.2f} "
                  f"({dev['IS']:+6.1f}%)" + ("   <== matches" if ok and arm == "flux" else ""))
        print()


if __name__ == "__main__":
    main()
