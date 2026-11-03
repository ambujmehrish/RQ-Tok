#!/usr/bin/env python3
"""Generate DPG-Bench images with Stable Diffusion v1.5 — scorer CALIBRATION only.

ELLA (arXiv 2403.05135, Table 4) reports SD v1.5 = 63.18 on DPG-Bench but does not state
its sampling settings, so the diffusers defaults for SD v1.5 are used (512 x 512, 50 steps,
guidance 7.5, fp16) with 4 images per prompt tiled into a 2 x 2 grid without padding, as
ELLA's README asks, named after the prompt file. Because the settings are not published,
the calibration tolerance is wider than GenEval's (see dpg_score.sbatch).
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompts_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from diffusers import StableDiffusionPipeline

    prompts = sorted(pathlib.Path(args.prompts_dir).glob("*.txt"))
    if len(prompts) != 1065:
        raise SystemExit(f"expected 1065 DPG-Bench prompts, found {len(prompts)}")
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for i, path in enumerate(prompts):
        target = out / (path.stem + ".png")
        if target.exists():
            continue
        prompt = path.read_text(encoding="utf-8").strip()
        g = torch.Generator(device="cuda").manual_seed(args.seed)
        imgs = pipe(prompt, num_images_per_prompt=4, height=512, width=512,
                    num_inference_steps=50, guidance_scale=7.5, generator=g).images
        grid = Image.new("RGB", (1024, 1024))
        for k, im in enumerate(imgs):
            grid.paste(im, ((k % 2) * 512, (k // 2) * 512))
        grid.save(target)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(prompts)}", flush=True)
    n = len(list(out.glob("*.png")))
    if n != 1065:
        raise SystemExit(f"{n} grids written, expected 1065")
    print(f"wrote 1065 SD v1.5 grids to {out}")


if __name__ == "__main__":
    main()
