"""Build upstream's Bifrost-1 MLLM for training (Level B, Stage 1).

The model is upstream's own ``MultiModalityCausalLM``, configured exactly as upstream's
inference script configures it, with three things upstream did not release and the paper
requires:

1. ``init_branch_from_mllm`` — upstream's ``add_vision_*`` create the vision branch as
   freshly *random* layers (the ``copy.deepcopy`` is commented out; at inference the
   checkpoint overwrites them). The paper initialises the branch FROM the MLLM's weights,
   so every branch tensor is copied from its MLLM twin, and any branch tensor left
   uncopied raises.
2. ``set_trainable`` — the frozen MLLM plus the trainable branch, head, mask token and
   learnable position embedding; the visual tower stays frozen.
3. ``sample_mar_mask`` — MAR masking with ratio ~ truncated N(1.0, 0.25) on [0.7, 1.0].
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .upstream import add_upstream_to_path

NUM_TOKENS = 64          # 8x8 post-merger latents at 224 px (EXPERIMENTS.md Level B)
RESOLUTION = 224
MAX_SEQ_LEN = 60         # upstream Conversation default; "only enough for imagenet"

_TRAINABLE_PREFIXES = ("vision_head.", "mask_token", "learnable_pos_emb")
_BRANCH_MARKER = "vision_branch_"


def model_config(vlm_name: str, inner_dim: int) -> dict[str, Any]:
    """Upstream inference script's ``model_config``, for the 64-token / 224-px setting.

    The FLUX VAE is NOT loaded (``vision_gen_vae`` name None): Stage 1 regresses Qwen's
    own visual latents, which upstream's forward computes from ``pixel_values`` or takes
    precomputed via ``image_clip_embs``.
    """
    return {
        "vision_language_model_config": {
            "cls": "Qwen2_5_VLForConditionalGeneration",
            "model_type": "vision_language_model",
            "params": {"model_name_or_path": vlm_name, "load_from_pretrained": True,
                       "remove_vision_und_encoder": False,
                       "frozen_modules_in_vlm": [], "huggingface_token": None},
        },
        "vision_gen_vae_config": {"cls": "AutoencoderKL", "model_type": "vision_gen_vae",
                                  "params": {"model_name_or_path": None,
                                             "load_from_pretrained": False,
                                             "frozen_vision_gen_vae": True,
                                             "huggingface_token": None}},
        # as upstream inference; recorded in the config only, never loaded by the MLLM
        "diffusion_decoder_config": {
            "cls": "black-forest-labs/FLUX.1-dev", "model_type": "diffusion_decoder",
            "params": {"pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
                       "revision": None, "variant": None,
                       "controlnet_model_name_or_path": None, "num_single_layers": 1,
                       "num_double_layers": 4, "diffusion_decoder_text_dropout_prob": 0.0}},
        "timestep_sampling_strategy": "uniform",
        "vision_denoising_type": "mar",
        "max_seq_length": MAX_SEQ_LEN,
        "num_visual_gen_tokens": NUM_TOKENS,
        "add_vision_branch": True,
        "add_vision_branch_reuse_layernorm": False,
        "use_discrete_visual_tokenizer": False,
        "add_timestep_token": False,
        "skip_text_part2": True,
        "add_vision_gen_mask_token": False,
        "add_vision_soi_eoi_tokens": False,
        "add_vision_soi_token": False,
        "vision_head_type": "linear",
        "vision_loss_type": "mse",
        "vision_pos_emb_type": "learnable_pos_emb",
        "use_clip_visual_encoder": True,
        "batch_size_t2i": 1,
        "t2i_resolution": RESOLUTION,
        "lambda_gpu": True,
        "use_2d_query_tokens": False,
        "e2e_training": False,
        "ctrlnet_training": False,
        "proportion_empty_prompts": 0.0,
        "lambda_clip": 0.0,
        "remove_vae": False,
        "vae_w_ctrlnet_training": False,
        "vae_wo_ctrlnet_training": False,
        "inner_dim": inner_dim,
        "vae_scale_by_4": False,
    }


def build_model(vlm_name: str, inner_dim: int, code_dir: str | None = None):
    """Upstream ``MultiModalityCausalLM`` with its branch initialised from the MLLM."""
    add_upstream_to_path(code_dir)
    from bifrost.models.configuration_bifrost import MultiModalityConfig
    from bifrost.models.modeling_bifrost import MultiModalityCausalLM

    model = MultiModalityCausalLM(MultiModalityConfig(**model_config(vlm_name, inner_dim)))
    lm = model.vision_language_model.model
    if lm.config.hidden_size != inner_dim:
        raise ValueError(f"{vlm_name} hidden size {lm.config.hidden_size} != inner_dim {inner_dim}")
    init_branch_from_mllm(model)
    return model


@torch.no_grad()
def init_branch_from_mllm(model) -> dict[str, int]:
    """Copy every vision-branch tensor from its MLLM twin (paper §3.2). Raises if any
    branch tensor is left uncopied or a shape does not match."""
    lm = model.vision_language_model.model
    pairs: list[tuple[str, torch.nn.Module, torch.nn.Module]] = []
    for i, layer in enumerate(lm.layers):
        att = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            pairs.append((f"layers.{i}.self_attn.vision_branch_{name}",
                          getattr(att, f"vision_branch_{name}"), getattr(att, name)))
        pairs.append((f"layers.{i}.vision_branch_mlp", layer.vision_branch_mlp, layer.mlp))
        pairs.append((f"layers.{i}.vision_branch_input_layernorm",
                      layer.vision_branch_input_layernorm, layer.input_layernorm))
        pairs.append((f"layers.{i}.vision_branch_post_attention_layernorm",
                      layer.vision_branch_post_attention_layernorm,
                      layer.post_attention_layernorm))
    pairs.append(("vision_branch_norm", lm.vision_branch_norm, lm.norm))

    copied: set[int] = set()
    for name, dst, src in pairs:
        dst_sd, src_sd = dst.state_dict(), src.state_dict()
        if dst_sd.keys() != src_sd.keys():
            raise ValueError(f"{name}: parameter names differ {sorted(dst_sd)} vs {sorted(src_sd)}")
        for k, v in src_sd.items():
            if dst_sd[k].shape != v.shape:
                raise ValueError(f"{name}.{k}: shape {tuple(dst_sd[k].shape)} != {tuple(v.shape)}")
        dst.load_state_dict({k: v.clone() for k, v in src_sd.items()})
        copied.update(id(p) for p in dst.parameters())

    branch = [(n, p) for n, p in lm.named_parameters() if _BRANCH_MARKER in n]
    missed = [n for n, p in branch if id(p) not in copied]
    if missed:
        raise RuntimeError(f"{len(missed)} vision-branch tensors not initialised from the "
                           f"MLLM, e.g. {missed[:3]}")
    return {"modules": len(pairs), "tensors": len(branch)}


def set_trainable(model) -> dict[str, int]:
    """Freeze the MLLM (incl. the visual tower); train branch + head + tokens."""
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        train = (_BRANCH_MARKER in name) or name.startswith(_TRAINABLE_PREFIXES)
        p.requires_grad_(train)
        if train:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    for name, p in model.vision_language_model.visual.named_parameters():
        if p.requires_grad:
            raise RuntimeError(f"visual tower parameter {name} is trainable")
    if n_train == 0:
        raise RuntimeError("no trainable parameters selected")
    return {"trainable": n_train, "frozen": n_frozen}


def sample_mar_mask(bsz: int, num_tokens: int, rng: np.random.Generator) -> torch.Tensor:
    """MAR mask (1 = masked): ratio ~ truncated N(1.0, 0.25) on [0.7, 1.0] (paper §3.2),
    ceil(ratio * N) positions in a uniformly random order per sample (as MAR)."""
    from scipy import stats

    gen = stats.truncnorm((0.7 - 1.0) / 0.25, 0.0, loc=1.0, scale=0.25)
    ratios = gen.rvs(bsz, random_state=rng)
    mask = torch.zeros(bsz, num_tokens)
    for b, r in enumerate(ratios):
        k = int(math.ceil(num_tokens * float(r)))
        order = rng.permutation(num_tokens)
        mask[b, order[:k]] = 1.0
    return mask


def make_conversation(processor_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                      code_dir: str | None = None):
    """Upstream ``Conversation`` with upstream inference's settings. ``processor_name``
    must be the hub id: upstream keys the image-pad token on it."""
    add_upstream_to_path(code_dir)
    from bifrost.conversation import Conversation

    return Conversation(processor_name_or_path=processor_name, full_vision_mask=True,
                        precise_prompt_mask=True, add_timestep_token=False,
                        cond_dropout_prob=0.0, add_vision_soi_eoi_tokens=False,
                        add_vision_soi_token=False, vision_pos_emb_type="learnable_pos_emb",
                        max_seq_length=MAX_SEQ_LEN)


def forward_kwargs(batch_size: int) -> dict[str, Any]:
    """Keyword arguments upstream's training ``forward`` reads."""
    return {"max_seq_length": MAX_SEQ_LEN, "num_visual_gen_tokens": NUM_TOKENS,
            "precise_prompt_mask": True, "add_vision_branch": True,
            "add_vision_branch_reuse_layernorm": False, "add_timestep_token": False,
            "use_discrete_visual_tokenizer": False, "skip_text_part2": True,
            "add_vision_gen_mask_token": False, "add_vision_soi_eoi_tokens": False,
            "add_vision_soi_token": False, "batch_size_t2i": batch_size}


def t2i_flow(conv, prompts: list[str], clip_embs: torch.Tensor,
             ar_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """One Stage-1 batch in upstream's ``t2i_flow`` format.

    ``clip_embs`` [B, 64, d] are the precomputed post-merger Qwen latents of the real
    images (upstream's forward accepts them as ``image_clip_embs``); ``ar_mask`` [B, 64]
    is 1 where a position is replaced by the mask token.
    """
    b = len(prompts)
    if clip_embs.shape[:2] != (b, NUM_TOKENS) or ar_mask.shape != (b, NUM_TOKENS):
        raise ValueError(f"expected [{b}, {NUM_TOKENS}, d] latents and [{b}, {NUM_TOKENS}] mask, "
                         f"got {tuple(clip_embs.shape)} / {tuple(ar_mask.shape)}")
    input_ids, labels, attn, img_pos, pos_ids = conv.t2i_prompt(
        prompts, img_h=RESOLUTION, img_w=RESOLUTION, num_visual_gen_tokens=NUM_TOKENS)
    side = RESOLUTION // 14
    return {
        "input_ids": input_ids, "labels": labels, "attention_mask": attn,
        "image_position_mask": img_pos, "position_ids": pos_ids,
        "t2i_image_grid_thw": torch.tensor([b, side, side]),
        "ar_mask": ar_mask,
        "image_clip_embs": clip_embs,
        # upstream's forward reads only .device from pixel_values when image_clip_embs is set
        "pixel_values": clip_embs,
    }
