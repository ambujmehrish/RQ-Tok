#!/usr/bin/env python3
"""Merge sharded Bifrost-1 generation outputs into the full benchmark layout.

    python merge_shards.py {geneval|dpgbench} CODE_DIR OUT_DIR SHARD_OUT_DIR [...]

GenEval: each shard numbers its prompt folders from 00000; they are renumbered to the
global index from shard.json, and every folder's metadata.jsonl must equal the official
line for that index. DPG-Bench: grids are named after the prompt file, so they are copied
as is. Refuses gaps, overlaps, missing images or a shard that did not finish.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys


def main() -> None:
    bench, code, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    shards = [pathlib.Path(p) for p in sys.argv[4:]]
    if not shards:
        raise SystemExit("no shard dirs given")
    specs = [json.loads((s / "shard.json").read_text()) for s in shards]
    if {sp["bench"] for sp in specs} != {bench}:
        raise SystemExit("shards are from a different benchmark")
    n = {sp["nshards"] for sp in specs}
    if len(n) != 1 or len(specs) != n.pop():
        raise SystemExit("need exactly one output dir per shard of the same split")
    if sorted(sp["shard"] for sp in specs) != list(range(len(specs))):
        raise SystemExit("missing or duplicated shard index")
    for s in shards:
        if not (s / "RUN.json").exists():
            raise SystemExit(f"{s} has no RUN.json: that shard did not finish")

    if bench == "geneval":
        official = [ln for ln in (code / "evaluation/GenEval/prompts/evaluation_metadata.jsonl")
                    .read_text(encoding="utf-8").split("\n") if ln.strip()]
        dst = out / "geneval_bifrost"
        dst.mkdir(parents=True, exist_ok=False)
        seen = set()
        for s, sp in zip(shards, specs, strict=True):
            for local in range(sp["end"] - sp["start"]):
                g = sp["start"] + local
                src = s / "geneval_bifrost" / f"{local:05d}"
                meta = json.loads((src / "metadata.jsonl").read_text().strip())
                if meta != json.loads(official[g]):
                    raise SystemExit(f"{src}: metadata does not match official prompt {g}")
                if len(list((src / "samples").glob("*.png"))) != 4:
                    raise SystemExit(f"{src}: expected 4 samples")
                shutil.copytree(src, dst / f"{g:05d}")
                seen.add(g)
        if seen != set(range(len(official))):
            raise SystemExit(f"merged {len(seen)} of {len(official)} GenEval prompts")
        print(f"merged {len(seen)} GenEval prompts x 4 images -> {dst}")
    elif bench == "dpgbench":
        names = sorted(os.listdir(code / "evaluation/DPGBench/dpg_bench/prompts"))
        want = {nm.replace(".txt", ".png") for nm in names}
        dst = out / "dpgbench_bifrost"
        dst.mkdir(parents=True, exist_ok=False)
        for s in shards:
            for f in (s / "dpgbench_bifrost").glob("*.png"):
                if (dst / f.name).exists():
                    raise SystemExit(f"{f.name} produced by more than one shard")
                shutil.copy2(f, dst / f.name)
        got = {f.name for f in dst.glob("*.png")}
        if got != want:
            raise SystemExit(f"merged {len(got)} grids; missing {len(want - got)}, "
                             f"unexpected {len(got - want)}")
        print(f"merged {len(got)} DPG-Bench grids -> {dst}")
    else:
        raise SystemExit(f"unknown benchmark {bench!r}")
    (out / "MERGED.json").write_text(json.dumps(
        {"bench": bench, "shards": [str(s) for s in shards]}, indent=1))


if __name__ == "__main__":
    main()
