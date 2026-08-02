"""Component A — Adaptive RVQ-CLIP tokenizer.

Residual-quantizes MLLM-native CLIP patch latents to an adaptive per-patch depth.

The model classes require torch (a real dependency from Phase 1 on). Imports fail
loudly if torch is missing — there are no silent fallbacks.
"""

from .codebook import Codebook
from .encoder import DummyCLIPEncoder, build_clip_encoder
from .fit import FitReport, fit_tokenizer
from .quantizer import (
    AdaptiveResidualQuantizer,
    QuantizeOutput,
    TokenizerLosses,
)
from .tokenizer import RVQCLIPTokenizer, build_tokenizer

__all__ = [
    "Codebook",
    "AdaptiveResidualQuantizer",
    "QuantizeOutput",
    "TokenizerLosses",
    "DummyCLIPEncoder",
    "build_clip_encoder",
    "RVQCLIPTokenizer",
    "build_tokenizer",
    "FitReport",
    "fit_tokenizer",
]
