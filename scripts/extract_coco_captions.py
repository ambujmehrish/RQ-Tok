#!/usr/bin/env python3
"""Extract (and verify) the real COCO captions file from the official annotations zip.

Kept as its own script, run with ``python3 -I``, because the archive is third-party data:
it is only ever read as a zip member, never executed, and nothing is imported from its
directory.

    python3 -I scripts/extract_coco_captions.py --zip ann.zip --out-dir annotations/
    python3 -I scripts/extract_coco_captions.py --verify annotations/captions_val2017.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import zipfile

MEMBER = "annotations/captions_val2017.json"


def extract(zip_path: str, out_dir: str) -> pathlib.Path:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        if MEMBER not in z.namelist():
            raise SystemExit(f"{zip_path} does not contain {MEMBER}")
        with z.open(MEMBER) as src:
            data = src.read()
    dest = out / pathlib.Path(MEMBER).name
    dest.write_bytes(data)
    print(f"[captions] wrote {dest} ({len(data) / 1e6:.1f} MB)")
    return dest


def verify(path: str) -> None:
    blob = json.loads(pathlib.Path(path).read_text())
    anns = blob.get("annotations") or []
    images = {a["image_id"] for a in anns if "image_id" in a}
    if len(images) < 1000:
        raise SystemExit(f"{path} covers only {len(images)} images; expected ~5000")
    print(f"[captions] verified {path}: {len(anns)} captions over {len(images)} images")


def main() -> None:
    ap = argparse.ArgumentParser(description="COCO captions extractor")
    ap.add_argument("--zip")
    ap.add_argument("--out-dir")
    ap.add_argument("--verify")
    args = ap.parse_args()
    if args.verify:
        verify(args.verify)
        return
    if not (args.zip and args.out_dir):
        raise SystemExit("need --zip and --out-dir, or --verify PATH")
    verify(str(extract(args.zip, args.out_dir)))


if __name__ == "__main__":
    main()
