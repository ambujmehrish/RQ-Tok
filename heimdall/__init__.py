"""Heimdall: a flow-matching, adaptive residual-quantized successor to Bifrost-1.

Top-level imports are kept light (config only) so the package imports without torch.
Model components import torch lazily inside their own modules.
"""

from .config import (
    HeimdallConfig,
    TokenizerConfig,
    MLLMConfig,
    RendererConfig,
    TrainConfig,
    get_preset,
    available_presets,
)

__version__ = "0.0.1"

__all__ = [
    "HeimdallConfig",
    "TokenizerConfig",
    "MLLMConfig",
    "RendererConfig",
    "TrainConfig",
    "get_preset",
    "available_presets",
    "__version__",
]
