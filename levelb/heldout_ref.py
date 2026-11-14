#!/usr/bin/env python3
"""A 10,000-image reference from the held-out slice of ImageNet train (the training crop).

    python -m levelb.heldout_ref --data $LEVELB_ROOT/imagenet256 --out <npz>

Validation only: used to locate the paper's FID pipeline and, if ever needed, for
validation FID. Never a test reference. Written as ADM-format ``arr_0`` uint8 NHWC.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = pathlib.Path(args.data)
    crops = np.load(root / "crops_train.npy", mmap_mode="r")
    held = np.sort(np.load(root / "split_seed0.npz")["heldout"])
    arr = np.ascontiguousarray(crops[held])
    if arr.shape[1:] != (256, 256, 3) or arr.dtype != np.uint8:
        raise SystemExit(f"unexpected crops {arr.dtype}{arr.shape}")
    np.savez(args.out, arr_0=arr)
    print(f"wrote {args.out}: {arr.shape[0]} held-out images")


if __name__ == "__main__":
    main()
