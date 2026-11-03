#!/usr/bin/env python3
"""Build a work dir holding one contiguous slice of the official benchmark prompts.

    python shard_prompts.py {geneval|dpgbench} CODE_DIR WORK_DIR SHARD NSHARDS

Upstream ``inference_geneval_dpgbench.py`` reads its prompts from ``evaluation/...``
relative to the working directory. Running it from WORK_DIR therefore makes it generate
exactly this slice, with no change to its code. The slice is read the same way upstream
reads the full file (GenEval: non-empty lines + matching metadata lines; DPG-Bench: sorted
directory listing), and recorded in WORK_DIR/shard.json. Prints the slice size only.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys


def bounds(n: int, shard: int, nshards: int) -> tuple[int, int]:
    return shard * n // nshards, (shard + 1) * n // nshards


def main() -> None:
    bench, code, work, shard, nshards = sys.argv[1], *sys.argv[2:4], *map(int, sys.argv[4:6])
    code_p, work_p = pathlib.Path(code), pathlib.Path(work)
    if not 0 <= shard < nshards:
        raise SystemExit("need 0 <= SHARD < NSHARDS")

    if bench == "geneval":
        src = code_p / "evaluation/GenEval/prompts"
        text = (src / "generation_prompts.txt").read_text(encoding="utf-8").split("\n")
        prompts = [t for t in text if len(t) > 0]                       # as upstream
        meta = [ln for ln in (src / "evaluation_metadata.jsonl").read_text(
            encoding="utf-8").split("\n") if ln.strip()]                # as upstream
        if len(prompts) != 553 or len(meta) != 553:
            raise SystemExit(f"expected 553 GenEval prompts/metadata, got "
                             f"{len(prompts)}/{len(meta)}")
        for p, m in zip(prompts, meta, strict=True):
            if json.loads(m)["prompt"] != p:
                raise SystemExit(f"prompt/metadata mismatch: {p!r}")
        a, b = bounds(len(prompts), shard, nshards)
        dst = work_p / "evaluation/GenEval/prompts"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "generation_prompts.txt").write_text("\n".join(prompts[a:b]) + "\n",
                                                    encoding="utf-8")
        (dst / "evaluation_metadata.jsonl").write_text("\n".join(meta[a:b]) + "\n",
                                                       encoding="utf-8")
        items: list = list(range(a, b))
    elif bench == "dpgbench":
        src = code_p / "evaluation/DPGBench/dpg_bench/prompts"
        names = sorted(os.listdir(src))                                 # as upstream
        if len(names) != 1065:
            raise SystemExit(f"expected 1065 DPG-Bench prompts, got {len(names)}")
        a, b = bounds(len(names), shard, nshards)
        dst = work_p / "evaluation/DPGBench/dpg_bench/prompts"
        dst.mkdir(parents=True, exist_ok=True)
        for name in names[a:b]:
            link = dst / name
            if not link.exists():
                link.symlink_to(src / name)
        items = names[a:b]
    else:
        raise SystemExit(f"unknown benchmark {bench!r}")

    (work_p / "shard.json").write_text(json.dumps(
        {"bench": bench, "shard": shard, "nshards": nshards, "start": a, "end": b,
         "items": items}, indent=1))
    print(b - a)


if __name__ == "__main__":
    main()
