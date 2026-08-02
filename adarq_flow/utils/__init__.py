"""Shared utilities (seeding, logging, device helpers).

Kept torch-free at package import. The multi-GPU primitives live in the
``distributed`` submodule (which imports torch) — import it explicitly:

    from adarq_flow.utils.distributed import setup_distributed, wrap_model
"""

__all__ = []
