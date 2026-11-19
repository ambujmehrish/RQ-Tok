#!/usr/bin/env python3
"""Side-by-side sample grid: the same classes, one row per arm, a real image on top.

    python -m levelb.sample_grid --root $LEVELB_ROOT --arms flux bifrost adarq adarq_tuned \
        --seed 0 --n 12 --out results/levelb/figures/samples_seed0.png

Every arm sampled with the same seed draws the same class sequence, so column i is one
class in every row. The top row is a held-out training image of that class (never seen
in training). Writes the PNG and a .txt listing the class of each column.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import zipfile

import numpy as np
from PIL import Image, ImageDraw

TILE, LABEL_W, PAD = 128, 150, 2


def read_rows(npz: pathlib.Path, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows ``idx`` of arr_0 (and labels) without loading the whole 2 GB array."""
    with zipfile.ZipFile(npz) as z, z.open("arr_0.npy") as f:
        ver = np.lib.format.read_magic(f)
        rd = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
              else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = rd(f)
        if fortran or dtype != np.uint8:
            raise SystemExit(f"{npz}: expected C-order uint8 images")
        per = int(np.prod(shape[1:]))
        start = f.tell()
        out = np.empty((len(idx),) + tuple(shape[1:]), dtype=np.uint8)
        for k, i in enumerate(idx):
            f.seek(start + int(i) * per)
            out[k] = np.frombuffer(f.read(per), dtype=np.uint8).reshape(shape[1:])
    with np.load(npz) as z:
        labels = np.asarray(z["labels"])[idx]
    return out, labels


def tile(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img).resize((TILE, TILE), Image.BICUBIC)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="LEVELB_ROOT")
    ap.add_argument("--arms", nargs="+", required=True, help="sample folder tags, e.g. bifrost")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    idx = np.arange(args.n)
    rows, classes = [], None
    for arm in args.arms:
        npz = root / "samples" / f"{arm}_seed{args.seed}" / "samples_10000.npz"
        if not npz.exists():
            raise SystemExit(f"{npz} missing (sampling or merge not done)")
        imgs, labels = read_rows(npz, idx)
        if classes is None:
            classes = labels
        elif not np.array_equal(labels, classes):
            raise SystemExit(f"{arm}: column classes differ from the first arm's")
        rows.append((arm, imgs))
    data = root / "imagenet256"
    names = json.loads((data / "class_names.json").read_text())
    lab = np.load(data / "labels_train.npy", mmap_mode="r")
    held = np.sort(np.load(data / "split_seed0.npz")["heldout"])
    held_lab = np.asarray(lab[held])
    crops = np.load(data / "crops_train.npy", mmap_mode="r")
    real = []
    for c in classes:
        hit = held[held_lab == c]
        real.append(np.asarray(crops[hit[0]]) if len(hit) else np.zeros((256, 256, 3), np.uint8))
    rows.insert(0, ("real (held-out)", np.stack(real)))

    w = LABEL_W + args.n * (TILE + PAD)
    h = len(rows) * (TILE + PAD) + 20
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    for j, c in enumerate(classes):
        draw.text((LABEL_W + j * (TILE + PAD) + 2, 4), names[int(c)][:20], fill="black")
    for r, (arm, imgs) in enumerate(rows):
        y = 20 + r * (TILE + PAD)
        draw.text((4, y + TILE // 2 - 6), arm, fill="black")
        for j in range(args.n):
            canvas.paste(tile(imgs[j]), (LABEL_W + j * (TILE + PAD), y))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    out.with_suffix(".txt").write_text(
        "\n".join(f"column {j}: class {int(c)} {names[int(c)]}" for j, c in enumerate(classes))
        + f"\nrows: real (held-out train image of the class), {', '.join(args.arms)}"
        + f" (seed {args.seed}, first {args.n} samples)\n")
    print(f"wrote {out} ({w}x{h})")


if __name__ == "__main__":
    main()
