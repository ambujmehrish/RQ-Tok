"""Image-latent encoder ("VAE") producing the Stage-B regression targets.

Stage B trains the ControlNet to steer the renderer toward the latent representation of
a real image. That target must come from the renderer's own encoder. Previously the
trainer hard-wired a ``DummyCLIPEncoder`` here with no config field, so Stage B regressed
toward a *randomly initialized projection* and its loss curve measured optimizer
plumbing rather than generative quality — with no way for a caller to opt out.

Selection is explicit via ``renderer.vae``; a real id raises rather than degrading.
"""

from __future__ import annotations

from torch import nn

from ..config import RendererConfig
from ..tokenizer.encoder import DummyCLIPEncoder


def build_image_latent_encoder(cfg: RendererConfig) -> nn.Module:
    """Construct the encoder that produces Stage-B image-latent targets."""
    if cfg.vae == "dummy":
        return DummyCLIPEncoder(cfg.latent_dim, cfg.num_image_tokens)
    from .flux import FluxVAE, is_flux

    if is_flux(cfg.vae):
        return FluxVAE(cfg.vae, cfg)
    raise NotImplementedError(
        f"no adapter for image-latent encoder '{cfg.vae}'. Supported: 'dummy' (CPU "
        "development) or a FLUX checkpoint whose VAE supplies Stage-B targets. Refusing "
        "to substitute the stand-in, which would make the renderer regress toward a "
        "random projection."
    )
