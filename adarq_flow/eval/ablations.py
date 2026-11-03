"""Ablation config generation (DESIGN.md §7 'new ablations').

Programmatically derives AdaRQ-Flow config variants from a base preset for the
planned ablations: residual depth ``D_max``, adaptive vs. fixed depth, codebook size
``K``, shared vs. per-depth codebooks, CFG scale, exposure-bias training on/off,
flow-residual head on/off, and a continuous+MSE-style "Bifrost" baseline.
"""

from __future__ import annotations

import dataclasses
import os

from ..config import AdaRQFlowConfig, get_preset


def _with(cfg: AdaRQFlowConfig, **sub_overrides) -> AdaRQFlowConfig:
    """Return a copy of ``cfg`` with nested dataclass field overrides.

    ``sub_overrides`` keys are ``"<block>.<field>"`` (e.g. ``"tokenizer.max_depth"``).
    """
    blocks: dict[str, dict] = {}
    for dotted, value in sub_overrides.items():
        block, field = dotted.split(".", 1)
        blocks.setdefault(block, {})[field] = value
    kwargs = {}
    for block, fields in blocks.items():
        if not hasattr(cfg, block):
            raise ValueError(f"unknown config block {block!r}")
        kwargs[block] = dataclasses.replace(getattr(cfg, block), **fields)
    return dataclasses.replace(cfg, **kwargs)


def _load_base(base: str) -> tuple[AdaRQFlowConfig, str]:
    """``base`` is a preset name or a YAML path (e.g. ``configs/experiment.yaml``)."""
    if base.endswith((".yaml", ".yml")):
        cfg = AdaRQFlowConfig.from_yaml(base)
        return cfg, os.path.splitext(os.path.basename(base))[0]
    return get_preset(base), base


def ablation_configs(base: str = "tiny_cpu") -> dict[str, AdaRQFlowConfig]:
    """Named ablation variants derived from ``base`` (named accordingly).

    Every variant inherits ``base``'s components and run_mode, and each one is checked
    with ``validate_run_mode``: deriving from ``configs/experiment.yaml`` yields
    experiment-mode ablations with no development stand-ins.
    """
    b, tag = _load_base(base)
    out: dict[str, AdaRQFlowConfig] = {
        "baseline": dataclasses.replace(b, name=f"{tag}-baseline")
    }

    def add(name: str, **ov):
        out[name] = _with(dataclasses.replace(b, name=f"{tag}-{name}"), **ov)

    # --- Interface factorial: {discrete codes on/off} x {distributional/mean objective}
    # This 2x2 is what separates "codes help" from "the distributional objective helps".
    # Head capacity is identical in all four cells (same module, same parameter count),
    # so a difference cannot be attributed to model size.
    add("continuous_mse",                       # conditional-mean continuous bridge
        **{"mllm.code_head": False, "mllm.residual_objective": "mse"})
    add("continuous_flow",                      # continuous bridge, distributional
        **{"mllm.code_head": False, "mllm.residual_objective": "flow"})
    add("hybrid_mse",                           # codes, but mean-regressed residual
        **{"mllm.residual_objective": "mse"})
    # (the 4th cell, codes + flow, IS `baseline`)

    # --- Rate allocation: the central claim.
    add("fixed_depth", **{"tokenizer.adaptive_depth": False})
    add("depth_2", **{"tokenizer.max_depth": 2})
    add("depth_8", **{"tokenizer.max_depth": 8})

    # --- Bridge capacity controls.
    add("codebook_16", **{"tokenizer.codebook_size": 16})
    add("codebook_256", **{"tokenizer.codebook_size": 256})
    # toggle AWAY from the base value; a variant equal to the baseline is a no-op
    if b.tokenizer.shared_codebook:
        add("per_depth_codebook", **{"tokenizer.shared_codebook": False})
    else:
        add("shared_codebook", **{"tokenizer.shared_codebook": True})
    add("discrete_only", **{"mllm.flow_residual_head": False})

    # --- Guidance (only meaningful with a discrete interface).
    add("cfg_1", **{"mllm.cfg_scale": 1.0})
    add("cfg_5", **{"mllm.cfg_scale": 5.0})

    # --- Renderer conditioning distribution.
    add("no_exposure_fix", **{"renderer.train_on_dequantized": False})

    ref = dataclasses.replace(out["baseline"], name="")
    for name, cfg in out.items():
        cfg.validate_run_mode()
        if name != "baseline" and dataclasses.replace(cfg, name="") == ref:
            raise ValueError(f"ablation {name!r} is identical to the baseline for base "
                             f"{base!r}: it would measure nothing")
    return out


def dump_ablation_configs(out_dir: str, base: str = "tiny_cpu") -> list[str]:
    """Write every ablation config to ``out_dir/<name>.yaml``; returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, cfg in ablation_configs(base).items():
        path = os.path.join(out_dir, f"{name}.yaml")
        cfg.to_yaml(path)
        paths.append(path)
    return paths
