"""Configuration system for AdaRQ-Flow.

Pure-Python (dataclasses only) so it imports and tests without torch/transformers.
Every component is selectable between a *tiny* (CPU-runnable) and a *real* (GPU) variant.

Usage:
    from adarq_flow.config import AdaRQFlowConfig, get_preset
    cfg = get_preset("tiny_cpu")

    # or from YAML:
    cfg = AdaRQFlowConfig.from_yaml("configs/base_gpu.yaml")

    # CLI:
    python -m adarq_flow.config --print tiny_cpu
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any


# --------------------------------------------------------------------------------------
# Component configs
# --------------------------------------------------------------------------------------
@dataclass
class TokenizerConfig:
    """Adaptive RVQ-CLIP tokenizer (Component A)."""

    # CLIP / patch geometry
    clip_dim: int = 32              # d: CLIP patch-latent dim (real: e.g. 1280)
    num_patches: int = 16           # N: patches per image (real: 256 = 16x16)

    # Residual vector quantization
    codebook_size: int = 64         # K
    max_depth: int = 4              # D_max residual levels
    shared_codebook: bool = True    # share C across depths vs. per-depth codebooks
    commitment_weight: float = 0.25
    entropy_weight: float = 0.01    # anti-collapse codebook entropy reg
    ema_decay: float = 0.99         # codebook EMA

    # Adaptive depth (halting)
    adaptive_depth: bool = True
    halt_residual_threshold: float = 0.05   # stop when residual norm fraction < this
    rate_penalty: float = 0.001             # encourages shorter depth where possible


@dataclass
class MLLMConfig:
    """Frozen MLLM + trainable vision generation branch (Component B)."""

    backbone: str = "dummy"         # "dummy" (tiny) | "Qwen/Qwen2.5-VL-7B-Instruct"
    hidden_dim: int = 64            # d': MLLM hidden dim (real: 3584 for 7B)
    num_layers: int = 2             # branch layers (real: full backbone depth)
    num_heads: int = 4
    freeze_backbone: bool = True    # core invariant: understanding preserved

    # Hybrid head
    code_head: bool = True          # discrete RVQ code classifier (cross-entropy)
    flow_residual_head: bool = True # continuous residual via flow matching
    flow_head_dim: int = 64
    flow_head_layers: int = 2

    # Classifier-free guidance (text dropout during training)
    cfg_text_dropout: float = 0.1
    cfg_scale: float = 3.0          # inference guidance scale on code logits

    # MAR-style masked decoding
    mask_min: float = 0.7
    mask_max: float = 1.0
    decode_steps: int = 16          # MLLM unmask steps (paper: robust for >= 8)


@dataclass
class RendererConfig:
    """Flow-matching latent ControlNet + FLUX renderer (Component C)."""

    backbone: str = "dummy"         # "dummy" (tiny DiT) | "black-forest-labs/FLUX.1-dev"
    image_size: int = 64            # real: 256/512/1024
    latent_dim: int = 32            # renderer latent channel dim (image-latent I/O)
    model_dim: int = 64             # DiT internal width (real: 3072 for FLUX)
    num_blocks: int = 2             # frozen backbone DiT depth (real: full FLUX)
    num_heads: int = 4              # attention heads
    num_image_tokens: int = 16      # image-latent token count L (real: (H/patch)^2)
    controlnet_double_blocks: int = 1   # real: 4
    controlnet_single_blocks: int = 1   # real: 1
    downsample_factor: int = 2      # 2D conv downsample before ControlNet

    # Flow matching (rectified flow) — NO diffusion
    flow_sigma_min: float = 0.0
    num_inference_steps: int = 8    # ODE solve steps (real default: 28)
    controlnet_conditioning_scale: float = 0.7
    guidance_scale: float = 3.5

    # Exposure-bias-aware training
    train_on_dequantized: bool = True
    scheduled_sampling_prob: float = 0.0   # ramp up to feed MLLM-sampled codes


@dataclass
class DistConfig:
    """Distributed / multi-GPU execution.

    World topology (rank, world_size, local_rank) is read from the launcher
    environment (``torchrun`` or SLURM ``srun``), never hard-coded here. These
    fields control *how* we parallelize. Defaults target a single Leonardo
    (Cineca) Booster node = 4x A100-64GB with DDP.
    """

    strategy: str = "ddp"           # "ddp" | "fsdp" | "none"
    backend: str = "nccl"           # "nccl" (GPU) | "gloo" (CPU)
    find_unused_parameters: bool = False   # DDP; True only if some params get no grad
    bucket_cap_mb: int = 25         # DDP gradient bucket size
    # FSDP (only used when strategy == "fsdp"; for very large trainable params)
    fsdp_sharding: str = "full"     # "full" (ZeRO-3) | "grad_op" (ZeRO-2)
    fsdp_mixed_precision: bool = True
    # Process-group init
    init_timeout_min: int = 30
    seed_per_rank: bool = True      # offset data RNG by rank (model init stays synced)


@dataclass
class TrainConfig:
    """Decoupled training (Stage 0 tokenizer -> A branch -> B renderer)."""

    stage: str = "tokenizer"        # "tokenizer" | "branch" | "renderer"
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 100
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"             # "cpu" | "cuda"
    precision: str = "fp32"         # "fp32" | "bf16"
    dataset: str = "dummy"          # "dummy" | "imagenet" | "blip3o"
    log_every: int = 10
    ckpt_dir: str = "checkpoints"


@dataclass
class AdaRQFlowConfig:
    name: str = "adarq-flow-tiny"
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    mllm: MLLMConfig = field(default_factory=MLLMConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dist: DistConfig = field(default_factory=DistConfig)

    # ---- (de)serialization -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AdaRQFlowConfig:
        d = dict(d or {})
        sub = {
            "tokenizer": (TokenizerConfig, d.pop("tokenizer", {})),
            "mllm": (MLLMConfig, d.pop("mllm", {})),
            "renderer": (RendererConfig, d.pop("renderer", {})),
            "train": (TrainConfig, d.pop("train", {})),
            "dist": (DistConfig, d.pop("dist", {})),
        }
        kwargs: dict[str, Any] = {}
        for key, (klass, raw) in sub.items():
            kwargs[key] = _build(klass, raw)
        # remaining top-level scalars (e.g. name)
        for k, v in d.items():
            kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> AdaRQFlowConfig:
        import yaml  # local import keeps module import light

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _build(klass, raw: dict[str, Any] | None):
    """Construct a dataclass from a dict, ignoring unknown keys (with a clear error)."""
    raw = raw or {}
    valid = {f.name for f in dataclasses.fields(klass)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {klass.__name__}: {sorted(unknown)}")
    return klass(**{k: v for k, v in raw.items() if k in valid})


# --------------------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------------------
def _tiny_cpu() -> AdaRQFlowConfig:
    """Smallest end-to-end-runnable config. Everything 'dummy'; CPU-friendly."""
    return AdaRQFlowConfig(name="adarq-flow-tiny-cpu")


def _base_gpu() -> AdaRQFlowConfig:
    """Real-backbone config (Qwen2.5-VL 7B + FLUX.1-dev). Requires GPU + HF auth."""
    return AdaRQFlowConfig(
        name="adarq-flow-base-gpu",
        tokenizer=TokenizerConfig(
            clip_dim=1280,
            num_patches=256,
            codebook_size=8192,
            max_depth=8,
        ),
        mllm=MLLMConfig(
            backbone="Qwen/Qwen2.5-VL-7B-Instruct",
            hidden_dim=3584,
            num_layers=28,
            num_heads=28,
            flow_head_dim=1280,
        ),
        renderer=RendererConfig(
            backbone="black-forest-labs/FLUX.1-dev",
            image_size=1024,
            latent_dim=64,
            model_dim=3072,
            num_blocks=19,
            num_heads=24,
            num_image_tokens=256,
            controlnet_double_blocks=4,
            controlnet_single_blocks=1,
            num_inference_steps=28,
        ),
        train=TrainConfig(
            batch_size=48,            # per-GPU; global batch = batch_size * world_size
            max_steps=200_000,
            device="cuda",
            precision="bf16",
            dataset="blip3o",
        ),
        dist=DistConfig(strategy="ddp", backend="nccl"),  # 4x A100 (Cineca Leonardo)
    )


_PRESETS = {
    "tiny_cpu": _tiny_cpu,
    "base_gpu": _base_gpu,
}


def get_preset(name: str) -> AdaRQFlowConfig:
    if name not in _PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(_PRESETS)}")
    return _PRESETS[name]()


def available_presets():
    return sorted(_PRESETS)


def _main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description="AdaRQ-Flow config inspector")
    parser.add_argument("--print", dest="preset", help="preset name to print")
    parser.add_argument("--yaml", help="path to a YAML config to load and print")
    parser.add_argument("--list", action="store_true", help="list presets")
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(available_presets()))
        return
    if args.yaml:
        cfg = AdaRQFlowConfig.from_yaml(args.yaml)
    elif args.preset:
        cfg = get_preset(args.preset)
    else:
        parser.error("provide --print <preset>, --yaml <path>, or --list")
    print(json.dumps(cfg.to_dict(), indent=2))


if __name__ == "__main__":
    _main()
