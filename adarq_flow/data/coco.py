"""Real image-text datasets.

The development stand-in (`DummyImageTextDataset`) draws images and token ids from
independent generators, so there is **zero mutual information between image and text**
(`ISSUES.md` B4). On that data the branch stage's cross-entropy and the entire CFG
mechanism cannot express anything: a null result is guaranteed by construction, not
measured. This module supplies real pairs instead.

``CocoCaptionsDataset``
    real COCO photographs with their real human-written captions, read from the official
    ``captions_val2017.json``. COCO is already the FID reference set in
    `EXPERIMENTS.md`, so the same corpus serves training and evaluation.

There is no synthetic fallback: a missing annotation file or image directory raises.
Padding a short set or inventing captions would silently restore the B4 defect while
reporting success.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import Any

from torch import Tensor
from torch.utils.data import Dataset


def load_coco_captions(annotations: str) -> dict[int, list[str]]:
    """Parse ``captions_val2017.json`` -> ``{image_id: [caption, ...]}``.

    Reads only the fields it needs; the file is third-party data, never executed.
    """
    path = pathlib.Path(annotations)
    if not path.is_file():
        raise FileNotFoundError(
            f"{annotations} not found. Fetch the official COCO annotations with "
            "scripts/prefetch_assets.sh (or scripts/extract_coco_captions.py) — "
            "refusing to substitute synthetic captions, which would reintroduce the "
            "zero-mutual-information defect (B4)."
        )
    with path.open() as f:
        blob = json.load(f)
    anns = blob.get("annotations")
    if not isinstance(anns, list) or not anns:
        raise ValueError(f"{annotations} has no 'annotations' list; not a COCO captions file")
    out: dict[int, list[str]] = {}
    for a in anns:
        img_id, cap = a.get("image_id"), a.get("caption")
        if img_id is None or not isinstance(cap, str) or not cap.strip():
            continue
        out.setdefault(int(img_id), []).append(cap.strip())
    if not out:
        raise ValueError(f"{annotations} yielded no usable (image_id, caption) pairs")
    return out


def _image_id_from_name(name: str) -> int | None:
    """COCO filenames embed the image id (e.g. ``000000039769_0001.jpg``)."""
    for chunk in pathlib.Path(name).stem.split("_"):
        if chunk.isdigit():
            return int(chunk)
    return None


class CocoCaptionsDataset(Dataset):
    """Real COCO images paired with their real captions.

    Args:
        image_dir: directory of real photographs whose filenames carry COCO image ids
            (as written by ``scripts/extract_clip_latents.py --save-images``).
        annotations: path to ``captions_val2017.json``.
        tokenize: ``str -> LongTensor[T]``. Required: the text must be tokenized by the
            SAME tokenizer as the MLLM backbone, so no default is guessed.
        image_transform: ``PIL.Image -> Tensor``. Required for the same reason — the
            preprocessing must match the visual tower.
        caption_index: which caption to use when an image has several (COCO has ~5).
            Fixed rather than random so a run is reproducible.
    """

    def __init__(self, image_dir: str, annotations: str,
                 tokenize: Callable[[str], Tensor],
                 image_transform: Callable[[Any], Tensor],
                 caption_index: int = 0) -> None:
        if tokenize is None or image_transform is None:
            raise ValueError(
                "tokenize and image_transform are required: text must be tokenized by "
                "the MLLM's own tokenizer and images preprocessed by its visual tower's "
                "processor. Guessing either would silently change the data."
            )
        try:
            from PIL import Image  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "pillow is required to read real images (pip install pillow). "
                "Refusing to substitute synthetic data."
            ) from e

        d = pathlib.Path(image_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"image_dir {image_dir} is not a directory")
        captions = load_coco_captions(annotations)

        pairs: list[tuple[pathlib.Path, str]] = []
        unmatched: list[str] = []
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img_id = _image_id_from_name(f.name)
            caps = captions.get(img_id) if img_id is not None else None
            if not caps:
                unmatched.append(f.name)
                continue
            pairs.append((f, caps[min(caption_index, len(caps) - 1)]))

        if not pairs:
            raise ValueError(
                f"no image in {image_dir} could be matched to a caption "
                f"({len(unmatched)} unmatched). Filenames must embed the COCO image id. "
                "Refusing to proceed with unpaired data: image-text pairing is the whole "
                "point of this stage (B4)."
            )
        if unmatched:
            # Loud, not silent: a partially-paired set changes the effective dataset.
            raise ValueError(
                f"{len(unmatched)} of {len(unmatched) + len(pairs)} images in {image_dir} "
                f"have no caption (e.g. {unmatched[:3]}). Fix the image set or the "
                "annotations rather than training on a silently reduced corpus."
            )

        self.pairs = pairs
        self.tokenize = tokenize
        self.image_transform = image_transform
        self.annotations = annotations

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        from PIL import Image

        path, caption = self.pairs[idx]
        with Image.open(path) as im:
            img = self.image_transform(im.convert("RGB"))
        ids = self.tokenize(caption)
        if not isinstance(img, Tensor) or not isinstance(ids, Tensor):
            raise TypeError("image_transform and tokenize must return tensors")
        return img, ids

    def caption(self, idx: int) -> str:
        return self.pairs[idx][1]

    @staticmethod
    def mutual_information_sanity_check(ds: CocoCaptionsDataset,
                                        samples: int = 32) -> float:
        """Fraction of sampled captions that are distinct.

        A cheap guard against the B4 failure mode: if every image maps to the same (or a
        random, image-independent) caption, there is no signal for the branch to learn.
        """
        n = min(samples, len(ds))
        if n < 2:
            raise ValueError("need >= 2 samples to check caption diversity")
        caps = [ds.caption(i) for i in range(n)]
        return len(set(caps)) / n
