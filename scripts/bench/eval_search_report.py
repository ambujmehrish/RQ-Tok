#!/usr/bin/env python3
"""Which evaluation pipeline reproduces the Bifrost-1 paper's ImageNet numbers?

    python scripts/bench/eval_search_report.py        # LEVELB_ROOT defaults to ~/work/levelb

Reads the metrics written by adm_score.sbatch for the FLUX-only samples (and, when
present, the Bifrost-1 samples) under every reference set, and compares them with the
paper's Table 1 rows. Folders flux*_seed0 hold plain-FLUX samples at different generation
sizes (flux = 256 px, flux1024 = 1024 px); variants are reference sets (adm, val, adm64, ...)
and resized copies (r256.adm = downscaled to 256 before scoring). A pipeline reproduces the
paper only if the training-free FLUX row matches: FID within 5%, sFID within 10%, IS within 5%.
"""

from __future__ import annotations

import json
import os
import pathlib

PAPER = {
    "flux": {"FID": 51.20, "sFID": 224.82, "IS": 157.46},
    "bifrost": {"FID": 25.77, "sFID": 53.67, "IS": 98.57},
}
PAPER_T4 = {"bifrost": {"FID": 18.64, "sFID": 43.64, "IS": 156.01}}  # Table 4, 64 steps


def main() -> None:
    root = pathlib.Path(os.environ.get("LEVELB_ROOT", pathlib.Path.home() / "work" / "levelb"))
    for arm, folders in (
        ("flux", sorted((root / "samples").glob("flux*_seed0"))),
        ("bifrost", [root / "samples" / "bifrost_seed0"]),
    ):
        print(
            f"=== {arm}   paper Table 1: "
            + "  ".join(f"{k} {v}" for k, v in PAPER[arm].items())
            + (
                ""
                if arm not in PAPER_T4
                else "   | Table 4: " + "  ".join(f"{k} {v}" for k, v in PAPER_T4[arm].items())
            )
        )
        for folder in folders:
            files = sorted(folder.glob("samples_10000*.metrics.json"))
            if not files:
                print(f"  {folder.name}: not scored")
            for f in files:
                m = json.loads(f.read_text())
                variant = f.name[len("samples_10000.") : -len(".metrics.json")]
                dev = {
                    k: 100 * (m[k] - PAPER[arm][k]) / PAPER[arm][k] for k in ("FID", "sFID", "IS")
                }
                ok = abs(dev["FID"]) <= 5 and abs(dev["sFID"]) <= 10 and abs(dev["IS"]) <= 5
                print(
                    f"  {folder.name:14s} {variant:12s} FID {m['FID']:8.3f} ({dev['FID']:+6.1f}%)  "
                    f"sFID {m['sFID']:8.3f} ({dev['sFID']:+6.1f}%)  IS {m['IS']:8.2f} "
                    f"({dev['IS']:+6.1f}%)" + ("   <== matches" if ok else "")
                )
        print()


if __name__ == "__main__":
    main()
