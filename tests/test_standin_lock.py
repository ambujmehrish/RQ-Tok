"""Stand-ins must be unbuildable outside the unit tests (adarq_flow/standins.py)."""

import os
import subprocess
import sys

import pytest

BUILD = {
    "DummyMLLMBackbone": "from adarq_flow.mllm import DummyMLLMBackbone as C; C(32, 1, 2)",
    "DummyCLIPEncoder": "from adarq_flow.tokenizer import DummyCLIPEncoder as C; C(16, 4)",
    "DummyFluxBackbone": "from adarq_flow.renderer import DummyFluxBackbone as C; C(8, 16, 1, 2, 4)",
    "DummyImageTextDataset": "from adarq_flow.data import DummyImageTextDataset as C; C(4)",
    "tiny_cpu tokenizer": ("from adarq_flow.config import get_preset; "
                           "from adarq_flow.tokenizer import build_tokenizer; "
                           "build_tokenizer(get_preset('tiny_cpu'))"),
}


@pytest.mark.parametrize("name", sorted(BUILD))
def test_stand_in_refused_outside_tests(name):
    env = {k: v for k, v in os.environ.items() if k != "ADARQ_ALLOW_STAND_INS"}
    r = subprocess.run([sys.executable, "-c", BUILD[name]], env=env, capture_output=True,
                       text=True)
    assert r.returncode != 0
    assert "development stand-in" in r.stderr


def test_disabled_dummy_clis():
    env = {k: v for k, v in os.environ.items() if k != "ADARQ_ALLOW_STAND_INS"}
    for mod in ("adarq_flow.inference.generate", "adarq_flow.training.train"):
        r = subprocess.run([sys.executable, "-m", mod], env=env, capture_output=True, text=True)
        assert r.returncode != 0 and "disabled" in r.stderr
