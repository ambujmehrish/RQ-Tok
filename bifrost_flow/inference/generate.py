"""Inference entrypoint (smoke / demo).

    python -m bifrost_flow.inference.generate --preset tiny_cpu

Builds the full pipeline and runs one generation. With real backbones, pass text token
ids from the MLLM tokenizer and decode ``image_latents`` through the FLUX VAE.
"""

from __future__ import annotations

import argparse

import torch

from ..config import get_preset
from .pipeline import build_pipeline


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Bifrost-Flow inference")
    ap.add_argument("--preset", default="tiny_cpu")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--text-len", type=int, default=5)
    ap.add_argument("--cfg-scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer-ckpt", default=None)
    ap.add_argument("--branch-ckpt", default=None)
    ap.add_argument("--renderer-ckpt", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    cfg = get_preset(args.preset)
    pipe = build_pipeline(cfg)
    for ck in (args.tokenizer_ckpt, args.branch_ckpt, args.renderer_ckpt):
        if ck:
            print(f"loaded {pipe.load_stage(ck)} <- {ck}", flush=True)

    text = torch.randint(0, 256, (args.batch, args.text_len))
    g = torch.Generator().manual_seed(args.seed)
    out = pipe.generate(text, cfg_scale=args.cfg_scale, temperature=0.0, generator=g)
    print(f"codes {tuple(out.codes.shape)}  depths(mean)={out.depths.float().mean():.2f}  "
          f"latents {tuple(out.latents.shape)}  image_latents {tuple(out.image_latents.shape)}",
          flush=True)


if __name__ == "__main__":
    main()
