"""Locate and import the pinned upstream Bifrost-1 code. No fallback to anything else."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

UPSTREAM_COMMIT = "3047de271b54dbd37228b038cc1f949899b0f1a8"


def add_upstream_to_path(code_dir: str | None = None) -> pathlib.Path:
    """Put upstream's repo on ``sys.path`` after checking it is the pinned commit."""
    code = pathlib.Path(code_dir or os.environ.get("BIFROST_CODE", ""))
    if not (code / "bifrost" / "models" / "modeling_bifrost.py").is_file():
        raise SystemExit(f"upstream Bifrost-1 code not found at {code!s}: set BIFROST_CODE "
                         "(scripts/bifrost1/setup.sbatch writes it to bifrost1_env.sh)")
    if (code / ".git").exists():
        head = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"],
                                       text=True).strip()
        if head != UPSTREAM_COMMIT:
            raise SystemExit(f"{code} is at {head}, expected {UPSTREAM_COMMIT}")
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    return code


OFFICIAL_VLMS = ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct")


def require_official_model(name: str) -> None:
    """Real runs load the official Qwen2.5-VL checkpoints only. A local or tiny model path
    is accepted solely inside the unit tests (ADARQ_ALLOW_STAND_INS=1, set by conftest)."""
    if name in OFFICIAL_VLMS:
        return
    if os.environ.get("ADARQ_ALLOW_STAND_INS") == "1":
        return
    raise SystemExit(f"{name!r} is not an official Qwen2.5-VL checkpoint {OFFICIAL_VLMS}; "
                     "real runs use the real model only")
