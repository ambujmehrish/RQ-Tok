#!/usr/bin/env python3
"""Level B sampler: class prompt -> MLLM MAR decoding -> latent ControlNet + FLUX image.

    python -m levelb.sample --arm bifrost --branch $R/branch_bifrost/final_trainable.pt \
        --controlnet $R/controlnet/final_controlnet.pt --data $R/imagenet256 \
        --seed 0 --n 10000 --shard 0 --nshards 1 --out $R/samples/bifrost_seed0

Stage 1 (bridge): upstream inference's MAR loop, ported line for line from
``inference_geneval_dpgbench.py`` with 64 tokens: unconditional ("") and conditional
halves, a random order per image (MAR ``sample_orders``), one step per token, cosine
mask schedule, prediction = linear vision head, guidance
``uncond + cfg_weight * (cond - uncond)`` with upstream's default weight 1.0.
Stage 2 (render): upstream's ``FluxControlNetPipeline`` with our ControlNet, the
precomputed class-prompt FLUX embeddings, 28 steps, ControlNet scale 0.7, guidance 3.5,
256 x 256 [paper §4]. Classes: ``n`` draws uniform over the 1000 classes from ``seed``
[paper §5.1]. Upstream leaves NumPy unseeded; here each batch seeds it, so a sample set is
reproducible from (seed, shard) — the decoding rule itself is unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import torch
from einops import rearrange

from . import bifrost_build as bb
from .flux_text import from_bits
from .upstream import add_upstream_to_path, require_official_flux, require_official_model


def _mask_by_order(mask_len, order, bsz, seq_len, device):
    masking = torch.zeros(bsz, seq_len, device=device)
    return torch.scatter(masking, dim=-1, index=order[:, :int(mask_len)],
                         src=torch.ones(bsz, seq_len, device=device)).bool()


@torch.no_grad()
def mar_decode(model, conv, prompts: list[str], cfg_weight: float = 1.0,
               np_seed: int | None = None) -> tuple[torch.Tensor, list[int]]:
    """Upstream's MAR loop. Returns tokens [B, N, d] and the number of tokens revealed per
    step (upstream's guidance slicing is only valid when it is 1 at every step)."""
    add_upstream_to_path()
    from bifrost.models.mar.utils import sample_orders

    n_tok = bb.NUM_TOKENS
    bsz = len(prompts)
    dev = next(model.parameters()).device
    input_ids, _, attn, img_pos, pos_ids = conv.t2i_prompt(
        [""] * bsz + list(prompts), img_h=bb.RESOLUTION, img_w=bb.RESOLUTION,
        num_visual_gen_tokens=n_tok)
    if pos_ids.shape[1] == 3:
        pos_ids = rearrange(pos_ids, "bsz k c -> k bsz c")
    text = model.vision_language_model.model.embed_tokens(input_ids[:, :-n_tok].to(dev))

    if np_seed is not None:
        np.random.seed(np_seed)
    mask = torch.ones(bsz, n_tok, device=dev)
    tokens = model.mask_token.repeat(bsz, n_tok, 1)
    orders = sample_orders(bsz, n_tok).to(dev)
    revealed = []
    for step in range(n_tok):
        cur = tokens.clone()
        tokens = torch.cat([tokens, tokens], dim=0)
        mask = torch.cat([mask, mask], dim=0)
        tokens = tokens + model.learnable_pos_emb
        emb = torch.cat((text, tokens), dim=1)
        out = model.vision_language_model(inputs_embeds=emb, use_cache=None,
                                          attention_mask=attn.to(dev), past_key_values=None,
                                          return_dict=True, position_ids=pos_ids.to(dev),
                                          image_position_mask=img_pos.to(dev), t_emb=None)
        pred = model.vision_head(out["hidden_states"][:, -n_tok:])
        ratio = np.cos(math.pi / 2.0 * (step + 1) / n_tok)
        mask_len = torch.tensor([np.floor(n_tok * ratio)], device=dev)
        mask_len = torch.maximum(torch.tensor([1.0], device=dev),
                                 torch.minimum(mask.sum(-1, keepdim=True) - 1, mask_len))
        mask_next = _mask_by_order(mask_len[0].item(), orders, bsz, n_tok, dev)
        to_pred = mask[:bsz].bool() if step >= n_tok - 1 else torch.logical_xor(
            mask[:bsz].bool(), mask_next)
        revealed.append(int(to_pred[0].sum()))
        mask = mask_next.float()
        to_pred2 = torch.cat([to_pred, to_pred], dim=0)
        sel = pred[to_pred2.nonzero(as_tuple=True)]
        sampled = sel[:bsz] + cfg_weight * (sel[bsz:] - sel[:bsz])
        cur[to_pred.nonzero(as_tuple=True)] = sampled.to(cur.dtype)
        tokens = cur
    return tokens, revealed


def load_bifrost_branch(vlm: str, width: int, ckpt: str, device):
    model = bb.build_model(vlm, inner_dim=width)
    state = torch.load(ckpt, map_location="cpu")
    named = dict(model.named_parameters())
    for n, t in state.items():
        if n not in named:
            raise KeyError(f"{ckpt}: unknown parameter {n}")
        named[n].data.copy_(t.to(named[n].dtype))
    expect = {n for n, _ in model.named_parameters()
              if "vision_branch_" in n or n.startswith(("vision_head.", "mask_token",
                                                        "learnable_pos_emb"))}
    if set(state) != expect:
        raise KeyError(f"{ckpt} does not hold exactly the trainable branch tensors")
    return model.to(device).eval()


def load_pipeline(flux_id: str, controlnet_ckpt: str, width: int, device, dtype):
    add_upstream_to_path()
    from bifrost.models.flux_cnet.pipeline_flux_controlnet import FluxControlNetPipeline
    from diffusers import FluxTransformer2DModel

    from .controlnet_build import build_controlnet

    tr = FluxTransformer2DModel.from_pretrained(flux_id, subfolder="transformer", torch_dtype=dtype)
    trained = json.loads((pathlib.Path(controlnet_ckpt).parent / "config.json").read_text())
    cn = build_controlnet(tr, width, num_double=int(trained["num_double"]),
                          num_single=int(trained["num_single"]))
    cn.load_state_dict(torch.load(controlnet_ckpt, map_location="cpu"))
    pipe = FluxControlNetPipeline.from_pretrained(flux_id, controlnet=cn.to(dtype),
                                                  transformer=tr, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["bifrost"], required=True)
    ap.add_argument("--branch", required=True, help="final_trainable.pt of Stage 1")
    ap.add_argument("--controlnet", required=True, help="final_controlnet.pt of Stage 2")
    ap.add_argument("--data", required=True, help="imagenet256 dir (class names, FLUX text)")
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--processor", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--flux", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cfg-weight", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cn-scale", type=float, default=0.7)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    require_official_model(args.vlm)
    require_official_model(args.processor)
    require_official_flux(args.flux)

    dev = torch.device(args.device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    root = pathlib.Path(args.data)
    meta = json.loads((root / "meta.json").read_text())
    width = int(meta["latent_width"])
    t5 = from_bits(np.load(root / "flux_t5.npy"))
    pooled = from_bits(np.load(root / "flux_pooled.npy"))
    names = json.loads((root / "class_names.json").read_text())

    classes = np.random.default_rng(args.seed).integers(0, 1000, args.n)
    lo, hi = args.shard * args.n // args.nshards, (args.shard + 1) * args.n // args.nshards
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    part = out / f"shard{args.shard:02d}of{args.nshards:02d}.npz"
    if part.exists():
        print(f"{part} exists; nothing to do")
        return

    model = load_bifrost_branch(args.vlm, width, args.branch, dev)
    # upstream inference casts the whole model to bf16; CPU (tests only) runs in fp32
    model = model.to(torch.bfloat16) if dev.type == "cuda" else model.float()
    conv = bb.make_conversation(args.processor)
    pipe = load_pipeline(args.flux, args.controlnet, width, dev, dtype)

    imgs = np.zeros((hi - lo, args.res, args.res, 3), dtype=np.uint8)
    for s in range(lo, hi, args.batch):
        e = min(s + args.batch, hi)
        lab = classes[s:e]
        bseed = args.seed * 1_000_003 + s
        tokens, revealed = mar_decode(model, conv, [names[int(c)] for c in lab],
                                      cfg_weight=args.cfg_weight, np_seed=bseed)
        if set(revealed) != {1}:
            raise RuntimeError(f"MAR revealed {sorted(set(revealed))} tokens per step; "
                               "upstream's guidance slicing needs exactly 1")
        idx = torch.from_numpy(lab.astype(np.int64))
        g = torch.Generator(device=dev).manual_seed(bseed)
        res = pipe(prompt_embeds=t5[idx].to(dev, dtype),
                   pooled_prompt_embeds=pooled[idx].to(dev, dtype),
                   control_image=tokens.to(dtype), num_inference_steps=args.steps,
                   controlnet_conditioning_scale=args.cn_scale, guidance_scale=args.guidance,
                   generator=g, height=args.res, width=args.res, output_type="np").images
        imgs[s - lo:e - lo] = (np.clip(res, 0, 1) * 255).round().astype(np.uint8)
        print(f"  {e - lo}/{hi - lo} images", flush=True)
    np.savez(part, arr_0=imgs, labels=classes[lo:hi])
    (out / f"shard{args.shard:02d}of{args.nshards:02d}.json").write_text(json.dumps(
        {**vars(args), "range": [lo, hi]}, indent=2))
    print(f"wrote {part}")


def merge(out_dir: str, nshards: int, n: int) -> pathlib.Path:
    """Concatenate shard npz files into the ADM-evaluator format (arr_0 uint8 NHWC)."""
    out = pathlib.Path(out_dir)
    parts = [np.load(out / f"shard{k:02d}of{nshards:02d}.npz") for k in range(nshards)]
    arr = np.concatenate([p["arr_0"] for p in parts])
    if arr.shape[0] != n:
        raise SystemExit(f"merged {arr.shape[0]} images, expected {n}")
    path = out / f"samples_{n}.npz"
    np.savez(path, arr_0=arr, labels=np.concatenate([p["labels"] for p in parts]))
    return path


if __name__ == "__main__":
    main()
