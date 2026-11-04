#!/usr/bin/env python3
"""One pass over ImageNet train: 256-px crops + upstream Qwen2.5-VL visual latents.

    python -m levelb.precompute --imagenet $IMAGENET_ROOT/hf --out $LEVELB_ROOT/imagenet256

Writes (row i = the i-th image in the dataset's train order, parquet shard by shard):
    crops_train.npy    uint8  [N, 256, 256, 3]  resize-short-side 256 (bilinear) + centre crop
    latents_train.npy  uint16 [N, 64, d]        bf16 bit patterns of the post-merger latents
    labels_train.npy   int16  [N]
The latents come from upstream's frozen Qwen2.5-VL visual tower applied to the SAME crop
resized to 224 px — the tensors upstream's training forward would compute from
``pixel_values`` and accepts precomputed as ``image_clip_embs``. Precomputing them is exact
(the tower is frozen and deterministic) and lets both arms and the shared ControlNet use
identical inputs. Every parquet shard is marked done separately, so a job killed at the
24 h limit resumes. Nothing is skipped silently: a row that fails to decode raises.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image

from . import imagenet as inet
from .upstream import add_upstream_to_path, require_official_model

TOKENS = (inet.QWEN_RES // 28) ** 2     # 14-px patches, 2x2 merge: 224 px -> 8x8 = 64


def _decode_crop(blob: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(blob)) as im:
        return np.asarray(inet.center_crop(im), dtype=np.uint8)


def _shards(root: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(root, "data", "train-*.parquet")))
    if not files:
        raise SystemExit(f"no train parquet shards under {root}/data")
    return files


def _open(path: pathlib.Path, dtype, shape) -> np.memmap:
    if path.exists():
        arr = np.load(path, mmap_mode="r+")
        if arr.dtype != dtype or arr.shape != shape:
            raise SystemExit(f"{path} has {arr.dtype}{arr.shape}, expected {dtype}{shape}")
        return arr
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--imagenet", required=True, help="dir with data/train-*.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
    # Upstream's visual tower uses eager attention over ONE block-diagonal matrix for
    # the whole batch, so memory grows with batch^2 (128 images = 64 GiB per matrix).
    # Upstream ships window tables for exactly bsz 32 at 16x16 patches (224 px):
    # precomputed_tensors_bsz32_grid16x16.py. Use that batch.
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expect-total", type=int, default=inet.NUM_TRAIN,
                    help="total train rows the shards must contain")
    args = ap.parse_args()
    require_official_model(args.vlm)

    import pyarrow.parquet as pq

    add_upstream_to_path()
    from bifrost.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers import AutoProcessor

    files = _shards(args.imagenet)
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    total = sum(counts)
    if total != args.expect_total:
        raise SystemExit(f"train shards hold {total} rows, expected {args.expect_total}")
    offsets = np.concatenate([[0], np.cumsum(counts)])

    dev = torch.device(args.device)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.vlm, torch_dtype=torch.bfloat16)
    visual = vlm.visual.to(dev).eval()
    width = int(vlm.config.hidden_size)
    del vlm
    proc = AutoProcessor.from_pretrained(args.vlm).image_processor

    out = pathlib.Path(args.out)
    (out / "done").mkdir(parents=True, exist_ok=True)
    crops = _open(out / "crops_train.npy", np.uint8, (total, inet.VAE_RES, inet.VAE_RES, 3))
    lats = _open(out / "latents_train.npy", np.uint16, (total, TOKENS, width))
    labels = _open(out / "labels_train.npy", np.int16, (total,))

    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for k, f in enumerate(files):
            marker = out / "done" / f"{k:05d}.json"
            if marker.exists():
                continue
            a, b = int(offsets[k]), int(offsets[k + 1])
            table = pq.read_table(f, columns=["image", "label"])
            blobs = [r["bytes"] for r in table.column("image").to_pylist()]
            lab = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
            if len(blobs) != b - a or lab.min() < 0 or lab.max() >= inet.NUM_CLASSES:
                raise SystemExit(f"{f}: unexpected rows/labels")
            crops[a:b] = np.stack(list(pool.map(_decode_crop, blobs, chunksize=64)))
            labels[a:b] = lab.astype(np.int16)
            for s in range(a, b, args.batch):
                e = min(s + args.batch, b)
                views = [inet.qwen_view(Image.fromarray(crops[i])) for i in range(s, e)]
                px = proc(images=views, return_tensors="pt")
                grid = px["image_grid_thw"]
                if not torch.all(grid == torch.tensor([1, 16, 16])):
                    raise SystemExit(f"unexpected Qwen grid {grid[0].tolist()} (need 1x16x16)")
                z = visual(px["pixel_values"].to(dev, torch.bfloat16), grid_thw=grid.to(dev))
                z = z.reshape(e - s, TOKENS, width).to(torch.bfloat16).cpu()
                lats[s:e] = z.view(torch.int16).numpy().view(np.uint16)   # exact bf16 bits
            crops.flush()
            lats.flush()
            labels.flush()
            marker.write_text(json.dumps({"shard": os.path.basename(f), "start": a, "end": b}))
            done = len(list((out / "done").glob("*.json")))
            rate = (time.time() - t0) / max(1, k + 1)
            print(f"  shard {k + 1}/{len(files)} rows {a}-{b} | {done} shards done | "
                  f"{rate:.0f} s/shard", flush=True)

    if len(list((out / "done").glob("*.json"))) != len(files):
        raise SystemExit("not every shard is done")
    (out / "meta.json").write_text(json.dumps({
        "rows": total, "vlm": args.vlm, "latent_width": width, "latent_tokens": TOKENS,
        "latent_dtype": "bfloat16 stored as uint16 bits",
        "crop": "resize short side 256 (bilinear) + centre crop 256",
        "qwen_view": "same crop resized to 224 (bicubic) -> Qwen image processor",
    }, indent=2))
    print(f"done: {total} images -> {out}")


if __name__ == "__main__":
    main()


def read_latents(arr: np.ndarray) -> torch.Tensor:
    """uint16 bf16 bit patterns (as stored) -> bfloat16 tensor."""
    return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).view(torch.bfloat16)
