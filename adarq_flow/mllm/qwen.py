"""Real frozen Qwen2.5-VL backbone + its native visual tower.

This replaces the CPU development stand-ins for Component B (`DummyMLLMBackbone`) and
the tokenizer's image encoder (`DummyCLIPEncoder`) with the actual pretrained model, per
`DESIGN.md` §9.

Two objects are exposed, matching the two things AdaRQ-Flow needs from an MLLM:

``QwenVisionEncoder``
    the model's **own** visual tower, producing the patch latents the tokenizer
    quantizes. Qwen2.5-VL's tower ends in a *patch merger* that projects into the text
    embedding space (``vision_config.out_hidden_size``), so these latents are literally
    what the LLM consumes — the "MLLM-native" property the bridge depends on.

``QwenBackbone``
    the frozen text stack, returning context hidden states for the generation branch.

Both are frozen and validated against the checkpoint's own config: nothing is hardcoded,
and a mismatch between the config you supply and the real model raises rather than being
silently reshaped.

Dynamic resolution
------------------
Qwen2.5-VL uses **variable** numbers of visual tokens per image (grid_thw), while
`TokenizerConfig.num_patches` is fixed. `QwenVisionEncoder` therefore requires images
preprocessed to a single resolution and verifies the realized token count, raising if it
differs. Padding or cropping to fit would silently change the bridge's rate, which is the
quantity every E0/C3 measurement controls for.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..config import MLLMConfig, TokenizerConfig

QWEN_PREFIXES = ("Qwen/Qwen2.5-VL", "Qwen/Qwen2-VL")


def is_qwen_vl(model_id: str) -> bool:
    return model_id.startswith(QWEN_PREFIXES)


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as e:  # no fallback: a stand-in would invalidate the run
        raise SystemExit(
            "transformers is required for the real Qwen2.5-VL backbone "
            '(pip install -e ".[models]"). Refusing to substitute a development '
            "stand-in."
        ) from e
    return transformers


def load_qwen_config(model_id: str) -> Any:
    """Config only — cheap, and enough to validate geometry before touching weights."""
    t = _require_transformers()
    return t.AutoConfig.from_pretrained(model_id)


def qwen_dims(model_id: str) -> dict[str, int]:
    """Geometry read from the checkpoint itself (never hardcoded).

    Returns ``text_hidden`` (the branch width), ``vision_hidden`` (tower internal), and
    ``visual_out`` (post-merger, i.e. the latent dimension the tokenizer sees).
    """
    cfg = load_qwen_config(model_id)
    text_cfg = getattr(cfg, "text_config", cfg)
    vis_cfg = getattr(cfg, "vision_config", None)
    if vis_cfg is None:
        raise RuntimeError(f"{model_id} has no vision_config; not a Qwen-VL checkpoint")
    text_hidden = getattr(text_cfg, "hidden_size", None)
    if text_hidden is None:
        raise RuntimeError(f"{model_id}: cannot determine text hidden_size")
    visual_out = getattr(vis_cfg, "out_hidden_size", None) or text_hidden
    return {
        "text_hidden": int(text_hidden),
        "vision_hidden": int(getattr(vis_cfg, "hidden_size", visual_out)),
        "visual_out": int(visual_out),
        "num_layers": int(getattr(text_cfg, "num_hidden_layers", 0)),
        "num_heads": int(getattr(text_cfg, "num_attention_heads", 0)),
    }


# One checkpoint serves both the visual tower and the text stack. Without this cache a
# config using both would load the weights twice (7B in bf16 = ~14 GB each).
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _load_model(model_id: str, device: str, dtype: torch.dtype) -> Any:
    key = (model_id, str(device), str(dtype))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    t = _require_transformers()
    cls = getattr(t, "Qwen2_5_VLForConditionalGeneration", None)
    if cls is None or not is_qwen_vl(model_id):
        cls = t.AutoModelForVision2Seq
    model = cls.from_pretrained(model_id, dtype=dtype)
    model = model.to(torch.device(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _MODEL_CACHE[key] = model
    return model


def _visual_tower(model: Any) -> nn.Module:
    """The visual tower, across the wrapper layouts transformers has used."""
    for path in (("visual",), ("model", "visual"), ("model", "vision_tower")):
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise RuntimeError(
        f"{type(model).__name__} exposes no visual tower (.visual / .model.visual); "
        "cannot produce MLLM-native patch latents.")


def _text_stack(model: Any) -> nn.Module:
    """The text decoder stack that yields context hidden states."""
    for path in (("model", "language_model"), ("language_model",), ("model",)):
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "embed_tokens"):
            return obj
    raise RuntimeError(
        f"{type(model).__name__} exposes no text stack with embed_tokens.")


class QwenVisionEncoder(nn.Module):
    """Frozen Qwen-VL visual tower -> MLLM-native patch latents ``[B, N, d]``.

    ``d`` is the checkpoint's ``vision_config.out_hidden_size`` (post patch-merger), i.e.
    the space the language model itself consumes.
    """

    def __init__(self, model_id: str, tok_cfg: TokenizerConfig,
                 device: str = "cpu", dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        dims = qwen_dims(model_id)
        if tok_cfg.clip_dim != dims["visual_out"]:
            raise ValueError(
                f"tokenizer.clip_dim={tok_cfg.clip_dim} does not match {model_id}'s "
                f"visual output width {dims['visual_out']}. Set clip_dim to "
                f"{dims['visual_out']}; silently projecting would change the bridge."
            )
        self.model_id = model_id
        self.expected_patches = tok_cfg.num_patches
        self.dim = dims["visual_out"]
        self._model = _load_model(model_id, device, dtype)
        self.tower = _visual_tower(self._model)
        self.eval()

    def _merged_features(self, out: Any) -> Tensor:
        """Extract the POST-MERGER visual features — the ones the LLM consumes.

        Qwen2.5-VL's tower ends in a patch merger that 2x2-pools and projects into the
        text embedding space. Verified on Qwen2.5-VL-3B (16x16 grid):

            last_hidden_state -> (256, 1280)   pre-merger, tower-internal
            pooler_output     -> ( 64, 2048)   POST-merger, what the LLM receives

        Taking ``last_hidden_state`` would quantize a representation the language model
        never sees, silently breaking the MLLM-native property the bridge depends on —
        so the width is checked against the config rather than assumed.
        """
        if isinstance(out, Tensor):
            return out                                   # older API: merged directly
        pooled = getattr(out, "pooler_output", None)
        if isinstance(pooled, Tensor) and pooled.shape[-1] == self.dim:
            return pooled
        lhs = getattr(out, "last_hidden_state", None)
        merger = getattr(self.tower, "merger", None)
        if isinstance(lhs, Tensor) and merger is not None:
            merged = merger(lhs)
            if isinstance(merged, Tensor) and merged.shape[-1] == self.dim:
                return merged
        raise RuntimeError(
            f"could not obtain post-merger visual features of width {self.dim} from "
            f"{type(out).__name__}; refusing to quantize a representation the language "
            "model does not consume."
        )

    @torch.no_grad()
    def forward(self, pixel_values: Tensor, image_grid_thw: Tensor) -> Tensor:
        """Preprocessed pixels -> ``[B, N, d]`` patch latents.

        Args:
            pixel_values: as produced by ``Qwen2VLImageProcessor``.
            image_grid_thw: ``[B, 3]`` grid sizes from the same processor.
        """
        if image_grid_thw.dim() != 2 or image_grid_thw.shape[-1] != 3:
            raise ValueError(
                f"image_grid_thw must be [B, 3], got {tuple(image_grid_thw.shape)}")
        b = image_grid_thw.shape[0]
        out = self.tower(pixel_values, grid_thw=image_grid_thw)
        feats = self._merged_features(out)
        if feats.dim() != 2:
            raise RuntimeError(
                f"expected flat visual features [total_tokens, d], got {tuple(feats.shape)}")
        total, d = feats.shape
        if d != self.dim:
            raise RuntimeError(f"visual width {d} != expected {self.dim}")
        if total % b != 0:
            raise RuntimeError(
                f"{total} visual tokens do not divide evenly across {b} images; "
                "Qwen-VL uses dynamic resolution — preprocess every image to the same "
                "size so the bridge rate is constant.")
        n = total // b
        if n != self.expected_patches:
            raise ValueError(
                f"this batch yields {n} visual tokens per image but "
                f"tokenizer.num_patches={self.expected_patches}. Rate is the controlled "
                "variable in every E0/C3 measurement, so it must not vary: fix the "
                "preprocessing resolution or set num_patches accordingly."
            )
        return feats.reshape(b, n, d).float()


class QwenBackbone(nn.Module):
    """Frozen Qwen-VL text stack -> context hidden states ``[B, T, H]``."""

    def __init__(self, cfg: MLLMConfig, device: str = "cpu",
                 dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        dims = qwen_dims(cfg.backbone)
        if cfg.hidden_dim != dims["text_hidden"]:
            raise ValueError(
                f"mllm.hidden_dim={cfg.hidden_dim} does not match {cfg.backbone}'s "
                f"hidden size {dims['text_hidden']}. The generation branch is a copy of "
                "these layers, so the widths must agree exactly."
            )
        if not cfg.freeze_backbone:
            raise ValueError(
                "mllm.freeze_backbone=False is not supported: a frozen backbone is the "
                "invariant that preserves understanding (EXPERIMENTS.md C6)."
            )
        self.model_id = cfg.backbone
        self.hidden_dim = dims["text_hidden"]
        self._model = _load_model(cfg.backbone, device, dtype)
        self.text = _text_stack(self._model)
        self.eval()

    @torch.no_grad()
    def forward(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if token_ids.dim() != 2:
            raise ValueError(f"expected token_ids [B, T], got {tuple(token_ids.shape)}")
        out = self.text(input_ids=token_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if hidden.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                f"context width {hidden.shape[-1]} != expected {self.hidden_dim}")
        return hidden.float()


def _main(argv: list[str] | None = None) -> None:
    """Print a checkpoint's geometry so configs are derived, never guessed.

        python -m adarq_flow.mllm.qwen Qwen/Qwen2.5-VL-7B-Instruct
    """
    import argparse

    ap = argparse.ArgumentParser(description="Qwen-VL geometry for AdaRQ-Flow configs")
    ap.add_argument("model_id")
    args = ap.parse_args(argv)
    d = qwen_dims(args.model_id)
    print(f"{args.model_id}")
    print(f"  mllm.hidden_dim      = {d['text_hidden']}   (branch width)")
    print(f"  tokenizer.clip_dim   = {d['visual_out']}   (POST-merger; what the LLM sees)")
    print(f"  vision hidden        = {d['vision_hidden']}   (tower-internal; do NOT use)")
    print(f"  text layers/heads    = {d['num_layers']}/{d['num_heads']}")
    print("  tokenizer.num_patches must be measured: grid_t*h*w // 4 for your resolution")


if __name__ == "__main__":
    _main()
