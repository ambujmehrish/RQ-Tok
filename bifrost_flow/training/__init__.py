"""Decoupled training pipelines: Stage 0 (tokenizer) -> A (branch) -> B (renderer).

Requires torch. See :class:`Trainer` and the ``train`` entrypoint.
"""

from .trainer import Trainer, TrainReport, VALID_STAGES

__all__ = ["Trainer", "TrainReport", "VALID_STAGES"]
