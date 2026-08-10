"""Structured, rank-aware logging.

A thin wrapper over the stdlib :mod:`logging` that (a) prefixes records with the
distributed rank and (b) by default emits **only from rank 0**, so multi-GPU runs
don't produce N copies of every line. Use :func:`get_logger` everywhere instead of
``print``.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _current_rank() -> int:
    """Best-effort rank from the launcher env (no torch import needed)."""
    for key in ("RANK", "SLURM_PROCID"):
        val = os.environ.get(key)
        if val is not None and val != "":
            try:
                return int(val)
            except ValueError as e:
                # Silently returning 0 would make every rank log as rank 0 and hide a
                # broken launcher environment.
                raise ValueError(f"environment variable {key}={val!r} is not an int") from e
    return 0


class _RankFilter(logging.Filter):
    def __init__(self, rank: int, main_only: bool) -> None:
        super().__init__()
        self.rank = rank
        self.main_only = main_only

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return (not self.main_only) or self.rank == 0


def configure_logging(level: int | str = logging.INFO, main_only: bool = True,
                      fmt: str = _DEFAULT_FMT) -> None:
    """Configure the ``adarq_flow`` root logger once (idempotent)."""
    global _CONFIGURED
    rank = _current_rank()
    root = logging.getLogger("adarq_flow")
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(f"[rank{rank}] {fmt}"))
    handler.addFilter(_RankFilter(rank, main_only))
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str, level: int | str = logging.INFO,
               main_only: bool = True) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging(level=level, main_only=main_only)
    return logging.getLogger(f"adarq_flow.{name}")
