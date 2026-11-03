"""ImageNet-1k for Level B: class prompts, the held-out split, and the image transform.

Everything here is deterministic and checked: the class list must have exactly 1000
sorted synsets, the held-out split is a seeded permutation saved to disk, and the
transform is the one fixed in EXPERIMENTS.md (Level B spec).
"""

from __future__ import annotations

import ast
import json
import pathlib

import numpy as np
from PIL import Image

NUM_CLASSES = 1000
NUM_TRAIN = 1_281_167
NUM_VAL = 50_000
HELDOUT = 10_000          # slice of train used ONLY for checkpoint selection
VAE_RES = 256             # FLUX VAE input (ImageNet 256 x 256)
QWEN_RES = 224            # Qwen2.5-VL visual input -> 16x16 patches -> 8x8 = 64 tokens


def class_names(classes_py: str | pathlib.Path) -> list[str]:
    """First English name per class, in label order, parsed (never executed) from the
    dataset's ``classes.py`` (``IMAGENET2012_CLASSES = OrderedDict({synset: names})``)."""
    tree = ast.parse(pathlib.Path(classes_py).read_text(encoding="utf-8"))
    mapping = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "IMAGENET2012_CLASSES" for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call) and value.args:      # OrderedDict({...})
                value = value.args[0]
            mapping = ast.literal_eval(value)
    if not isinstance(mapping, dict):
        raise ValueError(f"{classes_py}: IMAGENET2012_CLASSES dict literal not found")
    synsets = list(mapping)
    if len(synsets) != NUM_CLASSES:
        raise ValueError(f"{classes_py}: {len(synsets)} classes, expected {NUM_CLASSES}")
    if synsets != sorted(synsets):
        raise ValueError(f"{classes_py}: synsets are not in sorted (label) order")
    names = [str(mapping[s]).split(",")[0].strip() for s in synsets]
    if any(not n for n in names):
        raise ValueError(f"{classes_py}: empty class name")
    return names


def heldout_split(out: str | pathlib.Path, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Seeded split of the train indices into (train, heldout); saved once, then reloaded."""
    path = pathlib.Path(out)
    if path.exists():
        blob = np.load(path)
        if int(blob["seed"]) != seed or len(blob["heldout"]) != HELDOUT:
            raise ValueError(f"{path} was made with a different seed/size; refusing to mix")
        return blob["train"], blob["heldout"]
    perm = np.random.default_rng(seed).permutation(NUM_TRAIN)
    heldout, train = np.sort(perm[:HELDOUT]), np.sort(perm[HELDOUT:])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train=train, heldout=heldout, seed=seed)
    return train, heldout


def center_crop(img: Image.Image, res: int = VAE_RES) -> Image.Image:
    """Resize the short side to ``res`` (bilinear) and centre-crop ``res x res``.

    Matches torchvision ``Resize(res, BILINEAR) + CenterCrop(res)`` as used by diffusers'
    ``train_controlnet_flux.py`` (the paper's ControlNet trainer).
    """
    import torchvision.transforms as T

    img = img.convert("RGB")
    img = T.Resize(res, interpolation=T.InterpolationMode.BILINEAR)(img)
    return T.CenterCrop(res)(img)


def qwen_view(crop256: Image.Image) -> Image.Image:
    """The Qwen input is the SAME crop at 224 px, so its 8x8 latent grid is spatially
    aligned with the FLUX 16x16 token grid the ControlNet steers."""
    if crop256.size != (VAE_RES, VAE_RES):
        raise ValueError(f"expected a {VAE_RES}px crop, got {crop256.size}")
    return crop256.resize((QWEN_RES, QWEN_RES), Image.BICUBIC)


def save_class_names(names: list[str], out: str | pathlib.Path) -> None:
    pathlib.Path(out).write_text(json.dumps(names, indent=0))
