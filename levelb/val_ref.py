#!/usr/bin/env python3
"""Secondary test reference: the 50,000 ImageNet VALIDATION images as an ADM-style npz.

    python -m levelb.val_ref --imagenet $IMAGENET_ROOT/hf \
        --out $LEVELB_ROOT/imagenet256/val_ref_50000.npz

No stage of either arm ever trains on these images, so FID against them is a held-out
test set in the strict sense. Same 256-px crop as training (levelb.imagenet.center_crop);
written as ``arr_0`` uint8 [50000, 256, 256, 3], the format guided-diffusion's evaluator
reads (it computes the reference statistics itself). Reported separately from the
primary Bifrost-1 protocol (ADM reference batch).
"""

from __future__ import annotations

import argparse
import glob
import io
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from . import imagenet as inet


def _crop(blob: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(blob)) as im:
        return np.asarray(inet.center_crop(im), dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--imagenet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--expect", type=int, default=inet.NUM_VAL)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(args.imagenet, "data", "validation-*.parquet")))
    if not files:
        raise SystemExit("no validation parquet shards")
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if total != args.expect:
        raise SystemExit(f"validation shards hold {total} rows, expected {args.expect}")
    tmp = args.out + ".tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                    shape=(total, inet.VAE_RES, inet.VAE_RES, 3))
    pos = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for f in files:
            blobs = [r["bytes"] for r in pq.read_table(f, columns=["image"]).column(
                "image").to_pylist()]
            arr[pos:pos + len(blobs)] = np.stack(list(pool.map(_crop, blobs, chunksize=64)))
            pos += len(blobs)
            print(f"  {pos}/{total}", flush=True)
    arr.flush()
    np.savez(args.out, arr_0=arr)
    os.remove(tmp)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
