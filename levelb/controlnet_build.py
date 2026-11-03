"""Build upstream's latent ControlNet for the 3B bridge (Level B, Stage 2).

``FluxControlNetModel.from_transformer`` (upstream ``controlnet_flux.py``) copies FLUX's
embedders and its first 4 double / 1 single blocks, zero-initialises the per-block output
projections and the control input projection. Upstream hardcodes the bridge width to 3584
(Qwen2.5-VL-7B) in two layers — ``controlnet_x_embedder`` and ``up_sampling_block`` — so
those two are rebuilt for the 3B width (2048) with upstream's own classes and init: a
zero-initialised linear and a fresh ``NHWCUpsampleBlock``. Nothing else changes.
"""

from __future__ import annotations

import torch
from torch import nn

from .upstream import add_upstream_to_path

NUM_DOUBLE, NUM_SINGLE = 4, 1        # paper §4: num_double_layers=4, num_single_layers=1


def build_controlnet(transformer, width: int, num_double: int = NUM_DOUBLE,
                     num_single: int = NUM_SINGLE, seed: int = 0, code_dir: str | None = None):
    add_upstream_to_path(code_dir)
    from bifrost.models.flux_cnet import controlnet_flux as cf

    conf = transformer.config
    have = (len(transformer.transformer_blocks), len(transformer.single_transformer_blocks))
    if num_double > have[0] or num_single > have[1]:
        # upstream copies with strict=False, which would leave the extra blocks RANDOM
        raise ValueError(f"FLUX has {have[0]} double / {have[1]} single blocks; cannot copy "
                         f"{num_double} / {num_single} into the ControlNet")
    torch.manual_seed(seed)
    cn = cf.FluxControlNetModel.from_transformer(
        transformer, num_layers=num_double, num_single_layers=num_single,
        attention_head_dim=conf.attention_head_dim, num_attention_heads=conf.num_attention_heads)
    inner = conf.num_attention_heads * conf.attention_head_dim
    cn.controlnet_x_embedder = cf.zero_module(nn.Linear(width, inner))
    cn.up_sampling_block = cf.NHWCUpsampleBlock(width)
    verify_init(cn, transformer, num_double, num_single)
    return cn


@torch.no_grad()
def verify_init(cn, transformer, num_double: int, num_single: int) -> None:
    """Raise unless the ControlNet starts exactly as upstream's recipe says."""
    for i in range(num_double):
        for (k, a), (_, b) in zip(cn.transformer_blocks[i].state_dict().items(),
                                  transformer.transformer_blocks[i].state_dict().items(),
                                  strict=True):
            if not torch.equal(a, b.to(a.dtype)):
                raise RuntimeError(f"double block {i}.{k} not copied from FLUX")
    for i in range(num_single):
        for (k, a), (_, b) in zip(cn.single_transformer_blocks[i].state_dict().items(),
                                  transformer.single_transformer_blocks[i].state_dict().items(),
                                  strict=True):
            if not torch.equal(a, b.to(a.dtype)):
                raise RuntimeError(f"single block {i}.{k} not copied from FLUX")
    zeros = [cn.controlnet_x_embedder, *cn.controlnet_blocks, *cn.controlnet_single_blocks]
    for m in zeros:
        if any(p.abs().sum() != 0 for p in m.parameters()):
            raise RuntimeError("a zero-initialised ControlNet projection is not zero")
