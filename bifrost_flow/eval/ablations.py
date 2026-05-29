"""Ablation config generation (DESIGN.md §7 'new ablations').

Programmatically derives Bifrost-Flow config variants from a base preset for the
planned ablations: residual depth ``D_max``, adaptive vs. fixed depth, codebook size
``K``, shared vs. per-depth codebooks, CFG scale, exposure-bias training on/off,
flow-residual head on/off, and a continuous+MSE-style "Bifrost" baseline.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict

from ..config import BifrostFlowConfig, get_preset


def _with(cfg: BifrostFlowConfig, **sub_overrides) -> BifrostFlowConfig:
    """Return a copy of ``cfg`` with nested dataclass field overrides.

    ``sub_overrides`` keys are ``"<block>.<field>"`` (e.g. ``"tokenizer.max_depth"``).
    """
    blocks: Dict[str, dict] = {}
    for dotted, value in sub_overrides.items():
        block, field = dotted.split(".", 1)
        blocks.setdefault(block, {})[field] = value
    kwargs = {}
    for block, fields in blocks.items():
        if not hasattr(cfg, block):
            raise ValueError(f"unknown config block {block!r}")
        kwargs[block] = dataclasses.replace(getattr(cfg, block), **fields)
    return dataclasses.replace(cfg, **kwargs)


def ablation_configs(base: str = "tiny_cpu") -> Dict[str, BifrostFlowConfig]:
    """Named ablation variants derived from ``base`` (named accordingly)."""
    b = get_preset(base)
    out: Dict[str, BifrostFlowConfig] = {"baseline": dataclasses.replace(b, name=f"{base}-baseline")}

    def add(name: str, **ov):
        out[name] = _with(dataclasses.replace(b, name=f"{base}-{name}"), **ov)

    # Adaptive vs. fixed depth.
    add("fixed_depth", **{"tokenizer.adaptive_depth": False})
    # Residual depth D_max.
    add("depth_2", **{"tokenizer.max_depth": 2})
    add("depth_8", **{"tokenizer.max_depth": 8})
    # Codebook size K.
    add("codebook_16", **{"tokenizer.codebook_size": 16})
    add("codebook_256", **{"tokenizer.codebook_size": 256})
    # Shared vs. per-depth codebooks.
    add("per_depth_codebook", **{"tokenizer.shared_codebook": False})
    # CFG scale.
    add("cfg_1", **{"mllm.cfg_scale": 1.0})
    add("cfg_5", **{"mllm.cfg_scale": 5.0})
    # Flow-residual head on/off.
    add("no_flow_head", **{"mllm.flow_residual_head": False})
    # Exposure-bias training on/off.
    add("no_exposure_fix", **{"renderer.train_on_dequantized": False})
    # Continuous + no discrete codes (Bifrost-1-style baseline).
    add("bifrost_continuous", **{"mllm.code_head": False, "tokenizer.max_depth": 1})
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
