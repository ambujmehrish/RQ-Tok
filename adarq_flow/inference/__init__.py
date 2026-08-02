"""Inference pipeline: text -> adaptive RVQ codes (CFG) -> dequantize + flow residual
-> latent ControlNet -> flow ODE -> image latents. Requires torch.
"""

from .pipeline import AdaRQFlowPipeline, PipelineOutput, build_pipeline

__all__ = ["AdaRQFlowPipeline", "PipelineOutput", "build_pipeline"]
