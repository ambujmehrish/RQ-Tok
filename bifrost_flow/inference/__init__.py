"""Inference pipeline: text -> adaptive RVQ codes (CFG) -> dequantize + flow residual
-> latent ControlNet -> flow ODE -> image latents. Requires torch.
"""

from .pipeline import BifrostFlowPipeline, PipelineOutput, build_pipeline

__all__ = ["BifrostFlowPipeline", "PipelineOutput", "build_pipeline"]
