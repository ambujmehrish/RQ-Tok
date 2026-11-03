"""Hard lock on development stand-ins: they exist ONLY for the unit tests.

Every dummy component (``DummyMLLMBackbone``, ``DummyCLIPEncoder``, ``DummyFluxBackbone``,
``DummyImageTextDataset``) calls :func:`require_test_context` in its constructor. It
raises unless ``ADARQ_ALLOW_STAND_INS=1``, which only the test suites' ``conftest.py`` set.
So no script, job or notebook can build a stand-in, whatever config it is given — real
runs use real models and real data or they fail.
"""

from __future__ import annotations

import os

ENV_FLAG = "ADARQ_ALLOW_STAND_INS"


def stand_ins_allowed() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def require_test_context(what: str) -> None:
    if not stand_ins_allowed():
        raise RuntimeError(
            f"{what} is a development stand-in and may only be built inside the unit tests "
            f"({ENV_FLAG}=1 is set by tests/conftest.py). Real runs must use real models and "
            "real data: see configs/experiment.yaml and scripts/train_real.py."
        )
