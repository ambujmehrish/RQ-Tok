"""Component C — Flow-matching latent ControlNet + FLUX renderer.

FLUX.1-dev (rectified flow) conditioned on dequantized + flow-refined latents via a
flow-matching latent ControlNet. No DDPM-style diffusion. Requires torch.

Note: ``DummyFluxBackbone`` is a CPU development stand-in for the frozen FLUX.1-dev
transformer; it is replaced after the final development phase (DESIGN.md §9).
"""

from .backbone import DummyFluxBackbone, build_renderer_backbone
from .controlnet import LatentControlNet
from .flux import (
    FluxRendererBackbone,
    FluxTextConditioning,
    FluxVAE,
    flux_vae_dims,
    is_flux,
    is_test_fixture,
)
from .renderer import FlowRenderer, RendererLoss, build_renderer
from .vae import build_image_latent_encoder

__all__ = [
    "DummyFluxBackbone",
    "build_renderer_backbone",
    "LatentControlNet",
    "FlowRenderer",
    "RendererLoss",
    "build_renderer",
    "build_image_latent_encoder",
    "FluxVAE",
    "FluxRendererBackbone",
    "FluxTextConditioning",
    "flux_vae_dims",
    "is_flux",
    "is_test_fixture",
]
