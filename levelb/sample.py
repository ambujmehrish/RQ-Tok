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
    return torch.scatter(
        masking,
        dim=-1,
        index=order[:, : int(mask_len)],
        src=torch.ones(bsz, seq_len, device=device),
    ).bool()


def _prompt_inputs(model, conv, neg: list[str], prompts: list[str], dev):
    """Embedded text and masks for the guidance pair: the negative half first, then the
    conditional half (upstream's layout)."""
    n_tok = bb.NUM_TOKENS
    input_ids, _, attn, img_pos, pos_ids = conv.t2i_prompt(
        list(neg) + list(prompts),
        img_h=bb.RESOLUTION,
        img_w=bb.RESOLUTION,
        num_visual_gen_tokens=n_tok,
    )
    if pos_ids.shape[1] == 3:
        pos_ids = rearrange(pos_ids, "bsz k c -> k bsz c")
    text = model.vision_language_model.model.embed_tokens(input_ids[:, :-n_tok].to(dev))
    return text, attn.to(dev), pos_ids.to(dev), img_pos.to(dev)


def random_other_classes(names: list[str], labels, rng: np.random.Generator) -> list[str]:
    """Independent condition guidance (arXiv 2407.02687): one random class per image, never
    the image's own. Called once per decoding step."""
    off = rng.integers(1, len(names), size=len(labels))
    return [names[(int(c) + int(o)) % len(names)] for c, o in zip(labels, off, strict=True)]


@torch.no_grad()
def mar_decode(
    model,
    conv,
    prompts: list[str],
    cfg_weight: float = 1.0,
    np_seed: int | None = None,
    neg_fn=None,
    token_noise: float = 0.0,
    generator=None,
) -> tuple[torch.Tensor, list[int]]:
    """Upstream's MAR loop. Returns tokens [B, N, d] and the number of tokens revealed per
    step (upstream's guidance slicing is only valid when it is 1 at every step).
    ``neg_fn`` (None = upstream's empty prompt, computed once): returns the negative half's
    prompts, called every step. ``token_noise``: std of Gaussian noise added to each revealed
    token (0 = upstream; the "regression + noise" baseline)."""
    add_upstream_to_path()
    from bifrost.models.mar.utils import sample_orders

    n_tok = bb.NUM_TOKENS
    bsz = len(prompts)
    dev = next(model.parameters()).device
    text, attn, pos_ids, img_pos = _prompt_inputs(model, conv, [""] * bsz, prompts, dev)

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
        if neg_fn is not None:
            text, attn, pos_ids, img_pos = _prompt_inputs(model, conv, neg_fn(), prompts, dev)
        emb = torch.cat((text, tokens), dim=1)
        out = model.vision_language_model(
            inputs_embeds=emb,
            use_cache=None,
            attention_mask=attn,
            past_key_values=None,
            return_dict=True,
            position_ids=pos_ids,
            image_position_mask=img_pos,
            t_emb=None,
        )
        pred = model.vision_head(out["hidden_states"][:, -n_tok:])
        ratio = np.cos(math.pi / 2.0 * (step + 1) / n_tok)
        mask_len = torch.tensor([np.floor(n_tok * ratio)], device=dev)
        mask_len = torch.maximum(
            torch.tensor([1.0], device=dev), torch.minimum(mask.sum(-1, keepdim=True) - 1, mask_len)
        )
        mask_next = _mask_by_order(mask_len[0].item(), orders, bsz, n_tok, dev)
        to_pred = (
            mask[:bsz].bool()
            if step >= n_tok - 1
            else torch.logical_xor(mask[:bsz].bool(), mask_next)
        )
        revealed.append(int(to_pred[0].sum()))
        mask = mask_next.float()
        to_pred2 = torch.cat([to_pred, to_pred], dim=0)
        sel = pred[to_pred2.nonzero(as_tuple=True)]
        sampled = sel[:bsz] + cfg_weight * (sel[bsz:] - sel[:bsz])
        if token_noise > 0:
            sampled = sampled + token_noise * torch.randn(
                sampled.shape, device=dev, generator=generator
            ).to(sampled.dtype)
        cur[to_pred.nonzero(as_tuple=True)] = sampled.to(cur.dtype)
        tokens = cur
    return tokens, revealed


@torch.no_grad()
def mar_decode_adarq(
    model,
    heads,
    conv,
    prompts: list[str],
    cfg_weight: float = 1.0,
    np_seed: int | None = None,
    flow_steps: int = 32,
    generator=None,
    code_temps=None,
    top_p: float | None = None,
    greedy: bool = False,
    residual_scale: float = 1.0,
    neg_fn=None,
    info: list | None = None,
    flow_cfg_weight: float | None = None,
    guide_halt: bool = True,
    anchor=None,
):
    """The same MAR loop as ``mar_decode`` (unconditional + conditional halves, random
    order, one token per step, cosine schedule); the revealed patch is produced by the
    AdaRQ heads instead of the linear head: codes depth by depth from guided logits, then
    the residual by Euler integration of the guided flow velocity. Returns tokens, the
    reveal count per step, and each patch's sampled depth. ``neg_fn`` as in ``mar_decode``.
    ``info``: a list that receives one dict per step with the code distributions'
    information sums per level (see ``AdaRQHeads.sample``). ``anchor`` = (Bifrost-1 model,
    beta): every revealed token becomes ``mu + beta * (sample - mu)``, with ``mu`` the
    regression prediction of that model for the same context (beta 0 = Bifrost-1's decoding,
    beta 1 = this arm's); a no-training probe of a mean-anchored bridge."""
    add_upstream_to_path()
    from bifrost.models.mar.utils import sample_orders

    n_tok = bb.NUM_TOKENS
    bsz = len(prompts)
    dev = next(model.parameters()).device
    text, attn, pos_ids, img_pos = _prompt_inputs(model, conv, [""] * bsz, prompts, dev)
    if np_seed is not None:
        np.random.seed(np_seed)
    mask = torch.ones(bsz, n_tok, device=dev)
    tokens = model.mask_token.repeat(bsz, n_tok, 1)
    orders = sample_orders(bsz, n_tok).to(dev)
    depths = torch.zeros(bsz, n_tok, dtype=torch.long, device=dev)
    revealed = []
    if anchor is not None:
        model_b, beta = anchor
        text_b, attn_b, pos_b, img_pos_b = _prompt_inputs(model_b, conv, [], prompts, dev)
    for step in range(n_tok):
        cur = tokens.clone()
        masked_now = mask.bool()
        tokens = torch.cat([tokens, tokens], dim=0) + model.learnable_pos_emb
        mask = torch.cat([mask, mask], dim=0)
        if neg_fn is not None:
            text, attn, pos_ids, img_pos = _prompt_inputs(model, conv, neg_fn(), prompts, dev)
        emb = torch.cat((text, tokens), dim=1)
        out = model.vision_language_model(
            inputs_embeds=emb,
            use_cache=None,
            attention_mask=attn,
            past_key_values=None,
            return_dict=True,
            position_ids=pos_ids,
            image_position_mask=img_pos,
            t_emb=None,
        )
        hid = out["hidden_states"][:, -n_tok:]
        ratio = np.cos(math.pi / 2.0 * (step + 1) / n_tok)
        mask_len = torch.tensor([np.floor(n_tok * ratio)], device=dev)
        mask_len = torch.maximum(
            torch.tensor([1.0], device=dev), torch.minimum(mask.sum(-1, keepdim=True) - 1, mask_len)
        )
        mask_next = _mask_by_order(mask_len[0].item(), orders, bsz, n_tok, dev)
        to_pred = (
            mask[:bsz].bool()
            if step >= n_tok - 1
            else torch.logical_xor(mask[:bsz].bool(), mask_next)
        )
        revealed.append(int(to_pred[0].sum()))
        mask = mask_next.float()
        where = to_pred.nonzero(as_tuple=True)
        stats = {} if info is not None else None
        lat, dep = heads.sample(
            hid[bsz:][where],
            hid[:bsz][where],
            cfg_weight=cfg_weight,
            code_temps=code_temps,
            top_p=top_p,
            greedy=greedy,
            residual_scale=residual_scale,
            flow_steps=flow_steps,
            generators=generator,
            stats=stats,
            flow_cfg_weight=flow_cfg_weight,
            guide_halt=guide_halt,
        )
        if info is not None:
            info.append(stats)
        if anchor is not None:
            # the other model reads the same tokens, with its own mask token and positions
            tok_b = torch.where(masked_now[..., None], model_b.mask_token.to(cur.dtype), cur)
            out_b = model_b.vision_language_model(
                inputs_embeds=torch.cat((text_b, tok_b + model_b.learnable_pos_emb), dim=1),
                use_cache=None,
                attention_mask=attn_b,
                past_key_values=None,
                return_dict=True,
                position_ids=pos_b,
                image_position_mask=img_pos_b,
                t_emb=None,
            )
            mu = model_b.vision_head(out_b["hidden_states"][:, -n_tok:])[where].float()
            lat = mu + beta * (lat.float() - mu)
        cur[where] = lat.to(cur.dtype)
        depths[where] = dep
        tokens = cur
    return tokens, revealed, depths


def load_adarq(
    vlm: str,
    width: int,
    branch_ckpt: str,
    heads_ckpt: str,
    tok_dir: str,
    device,
    flow_width: int,
    flow_depth: int,
):
    from .adarq_model import AdaRQHeads
    from .adarq_tokenizer import load_books

    model = bb.build_model(vlm, inner_dim=width)
    state = torch.load(branch_ckpt, map_location="cpu")
    named = dict(model.named_parameters())
    for n, t in state.items():
        if n not in named:
            raise KeyError(f"{branch_ckpt}: unknown parameter {n}")
        named[n].data.copy_(t.to(named[n].dtype))
    books, res_rms, _, _ = load_books(str(pathlib.Path(tok_dir) / "tokenizer.pt"))
    heads = AdaRQHeads(width, books, res_rms, flow_width, flow_depth)
    heads.load_state_dict(torch.load(heads_ckpt, map_location="cpu"))
    return model.to(device).eval(), heads.to(device).eval()


def load_bifrost_branch(vlm: str, width: int, ckpt: str, device):
    model = bb.build_model(vlm, inner_dim=width)
    state = torch.load(ckpt, map_location="cpu")
    named = dict(model.named_parameters())
    for n, t in state.items():
        if n not in named:
            raise KeyError(f"{ckpt}: unknown parameter {n}")
        named[n].data.copy_(t.to(named[n].dtype))
    expect = {
        n
        for n, _ in model.named_parameters()
        if "vision_branch_" in n
        or n.startswith(("vision_head.", "mask_token", "learnable_pos_emb"))
    }
    if set(state) != expect:
        raise KeyError(f"{ckpt} does not hold exactly the trainable branch tensors")
    return model.to(device).eval()


def load_flux_only(flux_id: str, device, dtype):
    """The backbone alone: diffusers' FluxPipeline. Text encoders are not loaded — the
    same precomputed class-prompt embeddings as the other arms are passed in."""
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        flux_id,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        torch_dtype=dtype,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def load_pipeline(flux_id: str, controlnet_ckpt: str, width: int, device, dtype):
    add_upstream_to_path()
    from bifrost.models.flux_cnet.pipeline_flux_controlnet import FluxControlNetPipeline
    from diffusers import FluxTransformer2DModel

    from .controlnet_build import build_controlnet

    tr = FluxTransformer2DModel.from_pretrained(flux_id, subfolder="transformer", torch_dtype=dtype)
    trained = json.loads((pathlib.Path(controlnet_ckpt).parent / "config.json").read_text())
    cn = build_controlnet(
        tr, width, num_double=int(trained["num_double"]), num_single=int(trained["num_single"])
    )
    cn.load_state_dict(torch.load(controlnet_ckpt, map_location="cpu"))
    pipe = FluxControlNetPipeline.from_pretrained(
        flux_id, controlnet=cn.to(dtype), transformer=tr, torch_dtype=dtype
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--arm",
        choices=["bifrost", "adarq", "flux"],
        required=True,
        help="flux: the backbone alone (paper Table 1, row 1): class prompt -> "
        "FLUX.1-dev, no MLLM, no ControlNet; a training-free protocol anchor",
    )
    ap.add_argument("--heads", default=None, help="adarq: final_heads.pt")
    ap.add_argument("--tok", default=None, help="adarq: Stage-0 dir (tokenizer.pt)")
    ap.add_argument("--flow-steps", type=int, default=32)
    ap.add_argument(
        "--code-temps",
        default="1",
        help="adarq: code temperature per level, e.g. 0.7,0.85,1,1 (one value = all "
        "levels; 0 = greedy at that level). Applied to the ordinary codes only; "
        "halting is gated on the original halt probability",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="adarq: nucleus filter on the ordinary codes (halting unaffected)",
    )
    ap.add_argument(
        "--greedy-codes",
        action="store_true",
        help="adarq: always take the most likely ordinary code (halting stays sampled)",
    )
    ap.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="adarq: initial-noise scale of the residual flow (1 = as trained)",
    )
    ap.add_argument(
        "--class-balanced",
        action="store_true",
        help="every class exactly n/1000 times, in a seeded random order",
    )
    ap.add_argument("--branch", default=None, help="final_trainable.pt of Stage 1")
    ap.add_argument("--controlnet", default=None, help="final_controlnet.pt of Stage 2")
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
    ap.add_argument(
        "--flow-cfg-weight",
        type=float,
        default=None,
        help="adarq: guidance weight of the residual flow (default: --cfg-weight)",
    )
    ap.add_argument(
        "--no-guide-halt",
        action="store_true",
        help="adarq: decide halting from the unguided conditional logits",
    )
    ap.add_argument(
        "--anchor-branch",
        default=None,
        help="adarq: Bifrost-1 final_trainable.pt; tokens become mu + beta * (sample - mu)",
    )
    ap.add_argument("--anchor-beta", type=float, default=1.0)
    ap.add_argument(
        "--neg-prompt",
        choices=["empty", "random-class"],
        default="empty",
        help="negative half of the guidance pair: upstream's empty prompt, or an "
        "independent random class redrawn every step (arXiv 2407.02687; "
        "guidance for a model trained without prompt dropout)",
    )
    ap.add_argument(
        "--token-noise",
        type=float,
        default=0.0,
        help="bifrost: std of Gaussian noise added to each revealed token "
        "(the regression + noise baseline; 0 = upstream)",
    )
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

    if args.class_balanced:
        if args.n % 1000:
            raise SystemExit("--class-balanced needs --n to be a multiple of 1000")
        classes = np.random.default_rng(args.seed).permutation(
            np.repeat(np.arange(1000), args.n // 1000)
        )
    else:
        classes = np.random.default_rng(args.seed).integers(0, 1000, args.n)
    code_temps = [float(t) for t in args.code_temps.split(",")]
    lo, hi = args.shard * args.n // args.nshards, (args.shard + 1) * args.n // args.nshards
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    part = out / f"shard{args.shard:02d}of{args.nshards:02d}.npz"
    if part.exists():
        print(f"{part} exists; nothing to do")
        return

    heads = model = conv = anchor_model = None
    if args.anchor_branch and args.arm != "adarq":
        raise SystemExit("--anchor-branch is for --arm adarq")
    if args.arm != "flux" and not (args.branch and args.controlnet):
        raise SystemExit(f"--arm {args.arm} needs --branch and --controlnet")
    if args.arm == "flux":
        pipe = load_flux_only(args.flux, dev, dtype)
    elif args.arm == "adarq":
        if not (args.heads and args.tok):
            raise SystemExit("--arm adarq needs --heads and --tok")
        trained = json.loads((pathlib.Path(args.heads).parent / "config.json").read_text())
        model, heads = load_adarq(
            args.vlm,
            width,
            args.branch,
            args.heads,
            args.tok,
            dev,
            int(trained["flow_width"]),
            int(trained["flow_depth"]),
        )
        if args.anchor_branch:
            anchor_model = load_bifrost_branch(args.vlm, width, args.anchor_branch, dev)
            anchor_model = (
                anchor_model.to(torch.bfloat16) if dev.type == "cuda" else anchor_model.float()
            )
    else:
        model = load_bifrost_branch(args.vlm, width, args.branch, dev)
    if model is not None:
        # upstream inference casts the whole model to bf16; CPU (tests only) runs in fp32.
        # The AdaRQ heads stay fp32 (they are small; sampling precision for the flow ODE).
        model = model.to(torch.bfloat16) if dev.type == "cuda" else model.float()
        conv = bb.make_conversation(args.processor)
        pipe = load_pipeline(args.flux, args.controlnet, width, dev, dtype)

    imgs = np.zeros((hi - lo, args.res, args.res, 3), dtype=np.uint8)
    depth_sum, depth_n = 0, 0
    depth_hist = np.zeros(5, dtype=np.int64)  # index = sampled depth (1..4)
    # information of the code distributions, summed over patches: [decoding step, level]
    n_lvl = heads.D if heads is not None else 0
    info_sum = {
        key: np.zeros((bb.NUM_TOKENS, n_lvl)) for key in ("kl", "ent_cond", "ent_used", "n")
    }
    for s in range(lo, hi, args.batch):
        e = min(s + args.batch, hi)
        lab = classes[s:e]
        bseed = args.seed * 1_000_003 + s
        names_b = [names[int(c)] for c in lab]
        idx = torch.from_numpy(lab.astype(np.int64))
        g = torch.Generator(device=dev).manual_seed(bseed)
        if args.arm == "flux":
            res = pipe(
                prompt_embeds=t5[idx].to(dev, dtype),
                pooled_prompt_embeds=pooled[idx].to(dev, dtype),
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=g,
                height=args.res,
                width=args.res,
                output_type="np",
            ).images
            imgs[s - lo : e - lo] = (np.clip(res, 0, 1) * 255).round().astype(np.uint8)
            print(f"  {e - lo}/{hi - lo} images", flush=True)
            continue
        neg_fn = None
        if args.neg_prompt == "random-class":
            neg_rng = np.random.default_rng(4 * bseed + 5)  # its own stream
            neg_fn = lambda lab=lab, r=neg_rng: random_other_classes(names, lab, r)  # noqa: E731
        if heads is None:
            tokens, revealed = mar_decode(
                model,
                conv,
                names_b,
                cfg_weight=args.cfg_weight,
                np_seed=bseed,
                neg_fn=neg_fn,
                token_noise=args.token_noise,
                generator=torch.Generator(device=dev).manual_seed(4 * bseed + 3),
            )
        else:
            # separate streams (codes, halting, residuals); MAR orders use NumPy (np_seed)
            # and the renderer its own generator, so settings change only what they target
            gens = {
                name: torch.Generator(device=dev).manual_seed(4 * bseed + off)
                for name, off in (("code", 1), ("halt", 2), ("res", 3))
            }
            info_steps: list = []
            tokens, revealed, depth = mar_decode_adarq(
                model,
                heads,
                conv,
                names_b,
                cfg_weight=args.cfg_weight,
                np_seed=bseed,
                flow_steps=args.flow_steps,
                generator=gens,
                code_temps=code_temps,
                top_p=args.top_p,
                greedy=args.greedy_codes,
                residual_scale=args.residual_scale,
                neg_fn=neg_fn,
                info=info_steps,
                flow_cfg_weight=args.flow_cfg_weight,
                guide_halt=not args.no_guide_halt,
                anchor=None if anchor_model is None else (anchor_model, args.anchor_beta),
            )
            for k_step, st in enumerate(info_steps):
                for key in info_sum:
                    info_sum[key][k_step] += st[key].cpu().numpy()
            depth_sum += int(depth.sum())
            depth_n += depth.numel()
            depth_hist += np.bincount(depth.flatten().cpu().numpy(), minlength=5)[:5]
        if set(revealed) != {1}:
            raise RuntimeError(
                f"MAR revealed {sorted(set(revealed))} tokens per step; "
                "upstream's guidance slicing needs exactly 1"
            )
        res = pipe(
            prompt_embeds=t5[idx].to(dev, dtype),
            pooled_prompt_embeds=pooled[idx].to(dev, dtype),
            control_image=tokens.to(dtype),
            num_inference_steps=args.steps,
            controlnet_conditioning_scale=args.cn_scale,
            guidance_scale=args.guidance,
            generator=g,
            height=args.res,
            width=args.res,
            output_type="np",
        ).images
        imgs[s - lo : e - lo] = (np.clip(res, 0, 1) * 255).round().astype(np.uint8)
        print(f"  {e - lo}/{hi - lo} images", flush=True)
    np.savez(part, arr_0=imgs, labels=classes[lo:hi])
    (out / f"shard{args.shard:02d}of{args.nshards:02d}.json").write_text(
        json.dumps(
            {
                **vars(args),
                "range": [lo, hi],
                "mean_depth": (depth_sum / depth_n) if depth_n else None,
                "depth_hist": depth_hist[1:].tolist() if depth_n else None,
                "code_info_nats": None
                if not depth_n
                else {
                    "doc": "per [decoding step][level], mean over live patches: kl = KL(cond "
                    "|| negative branch), ent_cond = entropy of the conditional code "
                    "distribution, ent_used = entropy of the guided one; n = patches",
                    **{
                        key: (info_sum[key] / np.maximum(info_sum["n"], 1)).round(4).tolist()
                        for key in ("kl", "ent_cond", "ent_used")
                    },
                    "n": info_sum["n"].astype(int).tolist(),
                },
                "revealed_per_step": "1 (asserted every batch)",
            },
            indent=2,
        )
    )
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
