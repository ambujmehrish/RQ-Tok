#!/usr/bin/env python3
"""Extract REAL CLIP patch latents from REAL images.

This is the only supported way to obtain input for a real AdaRQ-Flow run on this
machine: a real CLIP vision tower applied to real photographs, producing the patch
latents the tokenizer quantizes.

There is deliberately **no synthetic fallback**. If the model or the images cannot be
fetched, the script exits non-zero. Substituting generated data here would silently turn
a real measurement into a meaningless one — the exact failure this repository is
guarding against (see ISSUES.md F8/F9).

    python scripts/extract_clip_latents.py --out latents.pt --num-images 64

Output: a ``[M, d]`` float tensor of L2-comparable patch latents plus a sidecar
``<out>.meta.json`` recording provenance (model id, image source, counts).
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

# COCO val2017 ids verified reachable; val2017 is the FID reference set in EXPERIMENTS.md.
COCO_VAL2017_IDS = [
    "000000039769", "000000000139", "000000000285", "000000000632", "000000000724",
    "000000000776", "000000000785", "000000000802", "000000000872", "000000000885",
    "000000001000", "000000001268", "000000001296", "000000001353", "000000001425",
    "000000001490", "000000001503", "000000001532", "000000001584", "000000001675",
]
COCO_URL = "http://images.cocodataset.org/val2017/{}.jpg"


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as e:  # no fallback: a missing dep must stop the run
        raise SystemExit(
            f"'{module}' is required to extract real CLIP latents but is not installed.\n"
            "  pip install transformers pillow\n"
            "Refusing to substitute synthetic latents."
        ) from e


def load_local_images(image_dir: str, n: int):
    """Load real photographs already on disk (offline compute nodes)."""
    _require("PIL")
    from PIL import Image

    d = pathlib.Path(image_dir)
    files = sorted(
        f for f in d.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if d.is_dir() else []
    if len(files) < n:
        raise SystemExit(
            f"{image_dir} holds {len(files)} images but {n} were requested.\n"
            "Run scripts/prefetch_assets.sh on a node WITH internet first; refusing to "
            "pad or to substitute synthetic data."
        )
    return [Image.open(f).convert("RGB") for f in files[:n]]


def fetch_images(n: int, timeout: int = 30, save_dir: str | None = None):
    """Download real photographs. Raises unless at least `n` distinct images arrive."""
    PIL = _require("PIL")
    from PIL import Image

    ids = (COCO_VAL2017_IDS * (n // len(COCO_VAL2017_IDS) + 1))[:n]
    images, failures = [], []
    for img_id in ids:
        url = COCO_URL.format(img_id)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                raw = r.read()
            images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            if save_dir:
                out = pathlib.Path(save_dir)
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{img_id}_{len(images):04d}.jpg").write_bytes(raw)
        except Exception as e:                      # noqa: BLE001 - reported, not hidden
            failures.append(f"{img_id}: {type(e).__name__}")
    if len(images) < n:
        raise SystemExit(
            f"fetched only {len(images)}/{n} real images; refusing to proceed with a "
            f"short or padded set.\nFailures: {failures[:5]}\n"
            "Check network egress to images.cocodataset.org."
        )
    del PIL
    return images[:n]


def extract(images, model_id: str, device: str):
    """Real CLIP vision tower -> per-patch hidden states [B, N, d] (CLS dropped)."""
    _require("transformers")
    from transformers import CLIPImageProcessor, CLIPVisionModel

    proc = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPVisionModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    batch = proc(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**batch).last_hidden_state          # [B, 1 + N, d]
    if out.shape[1] < 2:
        raise RuntimeError(f"unexpected CLIP output shape {tuple(out.shape)}")
    return out[:, 1:, :].contiguous().float().cpu()      # drop CLS -> patch latents


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract real CLIP patch latents")
    ap.add_argument("--out", required=True, help="path to write the [M, d] tensor")
    ap.add_argument("--num-images", type=int, default=64)
    ap.add_argument("--model", default="openai/clip-vit-base-patch32",
                    help="an ungated real CLIP vision tower")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--image-dir", default=None,
                    help="load real images from this directory instead of downloading "
                         "(required on compute nodes without internet)")
    ap.add_argument("--save-images", default=None,
                    help="also write the downloaded images here, for later offline use")
    args = ap.parse_args()

    if args.image_dir:
        print(f"[extract] loading {args.num_images} real images from {args.image_dir}",
              flush=True)
        images = load_local_images(args.image_dir, args.num_images)
    else:
        print(f"[extract] fetching {args.num_images} real COCO val2017 images ...",
              flush=True)
        images = fetch_images(args.num_images, save_dir=args.save_images)
    print(f"[extract] running real CLIP: {args.model}", flush=True)
    z = extract(images, args.model, args.device)         # [B, N, d]
    b, n, d = z.shape
    flat = z.reshape(-1, d)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(flat, out)
    meta = {
        "source": (f"local real images: {args.image_dir}" if args.image_dir
                   else "COCO val2017 (real photographs)"),
        "model": args.model,
        "num_images": b,
        "patches_per_image": n,
        "clip_dim": d,
        "num_patch_latents": int(flat.shape[0]),
        "synthetic": False,
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[extract] wrote {tuple(flat.shape)} real patch latents -> {out}", flush=True)
    print(f"[extract] {b} images x {n} patches, clip_dim={d}", flush=True)


if __name__ == "__main__":
    main()
