"""Downstream distortion for E0 — measured where it matters, not in latent L2.

`NOVELTY.md` §6 argues that reconstruction error is the *wrong* criterion for a
generative bridge: a textured region has large residual norm and attracts codes, while a
semantically load-bearing region may have small residual norm and be starved. This module
measures distortion **after** the quantized latents have been pushed through a real,
frozen, pretrained decoder.

The decoder used here is the **remaining suffix of the real CLIP vision tower**: quantize
the patch latents at layer ``L``, run layers ``L..end``, apply ``post_layernorm``, take
the pooled token through ``visual_projection``, and compare the resulting CLIP embedding
against the one the unquantized latents produce. That final embedding is the space CLIP
semantics (and CLIPScore-style alignment) actually live in.

Two properties matter:

* it is **real** — a frozen pretrained network, not a randomly-initialized stand-in
  (contrast ISSUES.md B3), and it runs offline from a prefetched cache;
* it is **non-separable across patches** — self-attention in the suffix mixes every patch
  into every output, so total distortion is *not* a sum of per-patch terms. The
  Lagrangian allocator in :mod:`adarq_flow.eval.allocation` therefore does **not** apply,
  and :func:`adarq_flow.eval.allocation.allocate_greedy_downstream` must be used instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TowerContext:
    """Everything needed to resume a real CLIP tower from an intermediate layer.

    Attributes:
        cls_tokens: ``[B, 1, d]`` pooled/CLS token captured at the same layer.
        layer: index into ``hidden_states`` the patch latents were taken from.
        num_images: ``B``.
        patches_per_image: ``N``.
    """

    cls_tokens: Tensor
    layer: int
    num_images: int
    patches_per_image: int


class ClipTowerDistortion:
    """Distortion in real CLIP embedding space, via the frozen tower suffix.

    Args:
        model_id: the same CLIP checkpoint the latents were extracted from.
        ctx: the CLS tokens / geometry saved alongside those latents.
        device: torch device for the suffix.
        metric: ``"cosine"`` (1 - cos similarity, bounded and scale-free) or ``"l2"``.
    """

    def __init__(self, model_id: str, ctx: TowerContext, device: str = "cpu",
                 metric: str = "cosine") -> None:
        if metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")
        try:
            from transformers import CLIPModel
        except ImportError as e:                       # no fallback: a stand-in is useless
            raise SystemExit(
                "transformers is required for downstream distortion "
                "(pip install transformers). Refusing to substitute a random decoder."
            ) from e

        self.metric = metric
        self.ctx = ctx
        self.device = torch.device(device)
        model = CLIPModel.from_pretrained(model_id)
        model = model.to(self.device)  # type: ignore[arg-type]
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        tower = getattr(model, "vision_model", None)
        if tower is None or not hasattr(tower, "encoder"):
            raise RuntimeError(
                f"{type(model).__name__} does not expose .vision_model with an encoder; "
                "load a CLIPModel so visual_projection is available too.")
        self.vision = tower
        self.projection = model.visual_projection
        n_layers = len(self.vision.encoder.layers)
        if not 0 <= ctx.layer <= n_layers:
            raise ValueError(
                f"layer {ctx.layer} out of range for a {n_layers}-layer tower")
        # hidden_states[i] is the INPUT to encoder layer i, so the suffix starts there.
        self.suffix = self.vision.encoder.layers[ctx.layer:]

    @torch.no_grad()
    def embed(self, patch_latents: Tensor, cls: Tensor | None = None) -> Tensor:
        """Patch latents ``[B, N, d]`` -> L2-normalized CLIP embedding ``[B, p]``.

        ``cls`` overrides the stored CLS tokens; pass a ``[B, 1, d]`` (or broadcastable
        ``[1, 1, d]``) tensor to score a batch of candidate allocations for ONE image.
        """
        b, n, _ = patch_latents.shape
        if n != self.ctx.patches_per_image:
            raise ValueError(
                f"expected {self.ctx.patches_per_image} patches, got {n}")
        if cls is None:
            if b != self.ctx.num_images:
                raise ValueError(
                    f"expected batch {self.ctx.num_images}, got {b}; pass `cls=` to "
                    "score candidate allocations for a single image")
            cls = self.ctx.cls_tokens
        cls = cls.to(patch_latents)
        if cls.shape[0] == 1 and b > 1:
            cls = cls.expand(b, -1, -1)
        h = torch.cat([cls, patch_latents], dim=1)
        for layer in self.suffix:
            out = layer(h, None)
            h = out[0] if isinstance(out, tuple) else out
        h = self.vision.post_layernorm(h)
        pooled = h[:, 0, :]                                   # CLS carries the pooled rep
        emb = self.projection(pooled)
        return emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def __call__(self, patch_latents: Tensor, reference: Tensor) -> Tensor:
        """Per-image distortion ``[B]`` between quantized and reference latents."""
        e_q = self.embed(patch_latents)
        e_r = self.embed(reference) if reference.dim() == 3 else reference
        if self.metric == "cosine":
            return 1.0 - (e_q * e_r).sum(-1)
        return (e_q - e_r).pow(2).sum(-1)

    @torch.no_grad()
    def reference_embedding(self, reference: Tensor) -> Tensor:
        return self.embed(reference)


def load_tower_context(path: str, device: str = "cpu") -> TowerContext:
    """Load the sidecar written by ``scripts/extract_clip_latents.py --layer L``."""
    blob = torch.load(path, map_location=device, weights_only=True)
    required = {"cls_tokens", "layer", "num_images", "patches_per_image"}
    missing = required - set(blob)
    if missing:
        raise ValueError(
            f"{path} is not a tower-context file (missing {sorted(missing)}). "
            "Re-extract with --layer to enable downstream distortion.")
    return TowerContext(
        cls_tokens=blob["cls_tokens"],
        layer=int(blob["layer"]),
        num_images=int(blob["num_images"]),
        patches_per_image=int(blob["patches_per_image"]),
    )
