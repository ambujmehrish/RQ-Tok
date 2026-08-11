"""Real FLUX renderer + VAE (Component C).

Replaces the CPU stand-ins for the renderer half: `DummyFluxBackbone` (a randomly
initialized DiT) and, critically, the `DummyCLIPEncoder` that was producing Stage-B
regression **targets** — a random projection, which made every renderer loss curve
meaningless (`ISSUES.md` B3).

Two objects:

``FluxVAE``
    the real `AutoencoderKL`, producing the image latents Stage B regresses toward.
    Latents are shifted/scaled and 2x2-packed exactly as FLUX does, so the targets live
    in the space the transformer actually operates on.

``FluxRendererBackbone``
    the real frozen `FluxTransformer2DModel`. ControlNet conditioning uses FLUX's
    **native** ``controlnet_block_samples`` / ``controlnet_single_block_samples``
    interface rather than hand-patched residual injection.

Verification status
-------------------
The API contract here was verified against the genuine diffusers classes using the
ungated ``hf-internal-testing/tiny-flux-pipe`` fixture (real architecture, tiny random
weights): latent channels / scaling / shift are read from the VAE config, and the
transformer's forward really does accept the controlnet arguments used below.

**Real FLUX.1-dev weights are NOT verified here** — the repo is gated and no token was
available in the development container. Geometry is therefore *derived from the
checkpoint config at load time* and checked, never hardcoded, and every mismatch raises.
Validate on the GPU node before citing any renderer number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..config import RendererConfig

FLUX_PREFIXES = ("black-forest-labs/FLUX",)
# Genuine architecture, random weights: usable to verify plumbing, never a real result.
TEST_FIXTURE_MARKERS = ("hf-internal-testing/", "tiny-random", "tiny-flux")


def is_flux(model_id: str) -> bool:
    return model_id.startswith(FLUX_PREFIXES) or "flux" in model_id.lower()


def is_test_fixture(model_id: str) -> bool:
    """Tiny random-weight checkpoints — real classes, meaningless outputs."""
    lowered = model_id.lower()
    return any(m in lowered for m in TEST_FIXTURE_MARKERS)


def _require_diffusers() -> Any:
    try:
        import diffusers
    except ImportError as e:  # no fallback: a stand-in would invalidate the run
        raise SystemExit(
            'diffusers is required for the real FLUX renderer (pip install -e ".[models]"). '
            "Refusing to substitute a development stand-in."
        ) from e
    return diffusers


@dataclass
class FluxTextConditioning:
    """Text conditioning FLUX requires and AdaRQ-Flow does not itself produce.

    FLUX is a text-conditioned model: its transformer needs T5 sequence embeddings and
    pooled CLIP embeddings. AdaRQ-Flow's contribution is the *bridge* conditioning
    (via ControlNet), so these must be supplied by the caller. They are **not** defaulted
    to zeros: silently feeding a null prompt would change what the renderer is being
    asked to draw while still producing a plausible loss curve.
    """

    encoder_hidden_states: Tensor   # [B, T, joint_attention_dim]
    pooled_projections: Tensor      # [B, pooled_projection_dim]
    txt_ids: Tensor                 # [T, 3]
    img_ids: Tensor                 # [L, 3]


def flux_vae_dims(model_id: str, subfolder: str = "vae") -> dict[str, float]:
    """VAE geometry read from the checkpoint (never hardcoded)."""
    d = _require_diffusers()
    cfg = d.AutoencoderKL.load_config(model_id, subfolder=subfolder)
    latent_channels = int(cfg["latent_channels"])
    n_down = len(cfg.get("down_block_types", []))
    return {
        "latent_channels": latent_channels,
        "packed_dim": latent_channels * 4,          # FLUX packs 2x2 spatial into channels
        "scaling_factor": float(cfg.get("scaling_factor", 1.0)),
        "shift_factor": float(cfg.get("shift_factor", 0.0) or 0.0),
        "spatial_downsample": 2 ** max(n_down - 1, 0),
    }


class FluxVAE(nn.Module):
    """Real frozen FLUX VAE -> packed image latents ``[B, L, C*4]``.

    These are the Stage-B regression targets. The transformer consumes 2x2-packed
    latents, so targets are packed identically; otherwise the ControlNet would be trained
    to steer toward a representation the renderer never uses.
    """

    def __init__(self, model_id: str, cfg: RendererConfig, device: str = "cpu",
                 dtype: torch.dtype = torch.bfloat16, subfolder: str = "vae") -> None:
        super().__init__()
        d = _require_diffusers()
        dims = flux_vae_dims(model_id, subfolder)
        if cfg.latent_dim != int(dims["packed_dim"]):
            raise ValueError(
                f"renderer.latent_dim={cfg.latent_dim} does not match {model_id}'s packed "
                f"latent width {int(dims['packed_dim'])} "
                f"({int(dims['latent_channels'])} channels x 2x2 packing). Set latent_dim "
                f"to {int(dims['packed_dim'])}; reshaping silently would change the target."
            )
        self.model_id = model_id
        self.scaling = dims["scaling_factor"]
        self.shift = dims["shift_factor"]
        self.latent_channels = int(dims["latent_channels"])
        self.packed_dim = int(dims["packed_dim"])
        self.expected_tokens = cfg.num_image_tokens
        # diffusers uses `torch_dtype` (transformers>=5 renamed it to `dtype`).
        vae = d.AutoencoderKL.from_pretrained(
            model_id, subfolder=subfolder, torch_dtype=dtype)
        self.vae = vae.to(torch.device(device)).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @staticmethod
    def pack(latents: Tensor) -> Tensor:
        """``[B, C, H, W]`` -> ``[B, (H/2)*(W/2), C*4]`` (FLUX's packing)."""
        b, c, h, w = latents.shape
        if h % 2 or w % 2:
            raise ValueError(f"latent grid {h}x{w} must be even to 2x2-pack")
        x = latents.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, (h // 2) * (w // 2), c * 4)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """Pixels in ``[-1, 1]`` ``[B, 3, H, W]`` -> packed latents ``[B, L, C*4]``."""
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"expected images [B, 3, H, W], got {tuple(images.shape)}")
        posterior = self.vae.encode(images.to(next(self.vae.parameters()).dtype))
        latents = posterior.latent_dist.mode()          # deterministic targets
        latents = (latents - self.shift) * self.scaling
        packed = self.pack(latents).float()
        if packed.shape[1] != self.expected_tokens:
            raise ValueError(
                f"this resolution yields {packed.shape[1]} image tokens but "
                f"renderer.num_image_tokens={self.expected_tokens}. Fix the input "
                "resolution or the config; a mismatch would silently change the target."
            )
        return packed


class FluxRendererBackbone(nn.Module):
    """Real frozen ``FluxTransformer2DModel`` with native ControlNet injection.

    Frozen but **grad-transparent**: ControlNet residuals enter through FLUX's own
    ``controlnet_block_samples`` arguments, so gradients reach the trainable ControlNet
    while no FLUX parameter is updated.
    """

    def __init__(self, model_id: str, cfg: RendererConfig, device: str = "cpu",
                 dtype: torch.dtype = torch.bfloat16, subfolder: str = "transformer") -> None:
        super().__init__()
        d = _require_diffusers()
        tr = d.FluxTransformer2DModel.from_pretrained(
            model_id, subfolder=subfolder, torch_dtype=dtype)
        conf = tr.config
        if int(conf.in_channels) != cfg.latent_dim:
            raise ValueError(
                f"renderer.latent_dim={cfg.latent_dim} does not match {model_id}'s "
                f"transformer in_channels={int(conf.in_channels)}."
            )
        n_double, n_single = int(conf.num_layers), int(conf.num_single_layers)
        if cfg.controlnet_double_blocks > n_double:
            raise ValueError(
                f"controlnet_double_blocks={cfg.controlnet_double_blocks} exceeds the "
                f"transformer's {n_double} double blocks")
        if cfg.controlnet_single_blocks > n_single:
            raise ValueError(
                f"controlnet_single_blocks={cfg.controlnet_single_blocks} exceeds the "
                f"transformer's {n_single} single blocks")
        self.model_id = model_id
        self.num_double = n_double
        self.num_single = n_single
        self.guidance_embeds = bool(getattr(conf, "guidance_embeds", False))
        self.transformer = tr.to(torch.device(device)).eval()
        for p in self.transformer.parameters():
            p.requires_grad_(False)

    def forward(self, x_t: Tensor, t: Tensor,
                control_residuals: list[Tensor] | None = None,
                text: FluxTextConditioning | None = None,
                guidance: Tensor | None = None) -> Tensor:
        """Predict the flow velocity for packed latents ``[B, L, C*4]``."""
        if text is None:
            raise ValueError(
                "FLUX is text-conditioned: encoder_hidden_states / pooled_projections / "
                "txt_ids / img_ids must be supplied (see FluxTextConditioning). Defaulting "
                "them to zeros would silently change the prompt the renderer is drawing."
            )
        if self.guidance_embeds and guidance is None:
            raise ValueError(
                f"{self.model_id} was trained with guidance embeddings; pass `guidance`.")
        double = control_residuals[:self.num_double] if control_residuals else None
        single = (control_residuals[self.num_double:] if control_residuals else None) or None
        out = self.transformer(
            hidden_states=x_t,
            encoder_hidden_states=text.encoder_hidden_states,
            pooled_projections=text.pooled_projections,
            timestep=t,
            img_ids=text.img_ids,
            txt_ids=text.txt_ids,
            guidance=guidance,
            controlnet_block_samples=double,
            controlnet_single_block_samples=single,
            return_dict=False,
        )
        return (out[0] if isinstance(out, tuple) else out).float()


def _main(argv: list[str] | None = None) -> None:
    """Print a FLUX checkpoint's geometry so configs are derived, never guessed.

        python -m adarq_flow.renderer.flux black-forest-labs/FLUX.1-dev

    Requires access to the (gated) repo: `hf auth login` and accept the licence.
    """
    import argparse

    ap = argparse.ArgumentParser(description="FLUX geometry for AdaRQ-Flow configs")
    ap.add_argument("model_id")
    args = ap.parse_args(argv)
    d = flux_vae_dims(args.model_id)
    print(args.model_id)
    print(f"  renderer.latent_dim      = {int(d['packed_dim'])}   "
          f"({int(d['latent_channels'])} VAE channels x 2x2 packing)")
    print(f"  vae scaling / shift      = {d['scaling_factor']} / {d['shift_factor']}")
    print(f"  vae spatial downsample   = {int(d['spatial_downsample'])}x")
    try:
        tr = _require_diffusers().FluxTransformer2DModel.load_config(
            args.model_id, subfolder="transformer")
        print(f"  transformer in_channels  = {int(tr['in_channels'])}  "
              "(must equal renderer.latent_dim)")
        print(f"  double / single blocks   = {int(tr['num_layers'])} / "
              f"{int(tr['num_single_layers'])}   (caps the ControlNet block counts)")
    except Exception as e:  # noqa: BLE001 - reported, not hidden
        print(f"  transformer config unavailable: {type(e).__name__}")
    print("  renderer.num_image_tokens must be measured: (H/8/2)*(W/8/2) at your resolution")


if __name__ == "__main__":
    _main()
