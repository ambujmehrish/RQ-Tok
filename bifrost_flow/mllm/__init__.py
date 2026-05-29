"""Component B — frozen MLLM + trainable vision generation branch (hybrid head).

Requires torch (a real dependency from Phase 1 on); imports fail loudly without it.

Note: ``DummyMLLMBackbone`` is a development stand-in (CPU-runnable). It will be
replaced by the real frozen Qwen2.5-VL after the final development phase; the rest of
the branch/head/training/decoding code is backbone-agnostic and unchanged by that swap.
"""

from .backbone import DummyMLLMBackbone, build_backbone
from .branch import VisionGenBranch
from .heads import CodeClassifierHead, FlowResidualHead
from .flow import flow_matching_loss, flow_sample, rectified_flow_target
from .model import (
    BranchLosses,
    GenerateOutput,
    VisionGenModel,
    build_vision_gen_model,
)

__all__ = [
    "DummyMLLMBackbone",
    "build_backbone",
    "VisionGenBranch",
    "CodeClassifierHead",
    "FlowResidualHead",
    "flow_matching_loss",
    "flow_sample",
    "rectified_flow_target",
    "BranchLosses",
    "GenerateOutput",
    "VisionGenModel",
    "build_vision_gen_model",
]
