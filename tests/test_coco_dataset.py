"""Real COCO image-caption dataset.

The property under test is the one the stand-in violates: real images paired with their
own real captions, so image and text actually carry mutual information (`ISSUES.md` B4).
Tests needing the annotation file skip when it is absent, so CI stays green.
"""

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from adarq_flow.config import get_preset
from adarq_flow.data import CocoCaptionsDataset, build_dataset, load_coco_captions


def _tok(s: str):
    return torch.tensor([ord(c) % 256 for c in s[:16]], dtype=torch.long)


def _tf(_im):
    return torch.zeros(3, 8, 8)


def _write_ann(path: pathlib.Path, mapping: dict[int, list[str]]) -> None:
    anns = [{"image_id": i, "caption": c} for i, caps in mapping.items() for c in caps]
    path.write_text(json.dumps({"annotations": anns}))


def _write_img(path: pathlib.Path) -> None:
    from PIL import Image

    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)


# -- caption parsing -------------------------------------------------------------------
def test_load_captions_groups_by_image(tmp_path):
    ann = tmp_path / "caps.json"
    _write_ann(ann, {1: ["a cat", "a feline"], 2: ["a dog"]})
    caps = load_coco_captions(str(ann))
    assert caps == {1: ["a cat", "a feline"], 2: ["a dog"]}


def test_missing_annotations_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="refusing to substitute"):
        load_coco_captions(str(tmp_path / "nope.json"))


def test_annotations_without_pairs_raises(tmp_path):
    ann = tmp_path / "empty.json"
    ann.write_text(json.dumps({"annotations": []}))
    with pytest.raises(ValueError, match="no 'annotations' list"):
        load_coco_captions(str(ann))


# -- pairing ---------------------------------------------------------------------------
def test_pairs_images_with_their_own_captions(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in (39769, 139):
        _write_img(img_dir / f"{i:012d}.jpg")
    ann = tmp_path / "caps.json"
    _write_ann(ann, {39769: ["two cats on a couch"], 139: ["a woman at a table"]})

    ds = CocoCaptionsDataset(str(img_dir), str(ann), tokenize=_tok, image_transform=_tf)
    assert len(ds) == 2
    captions = {ds.caption(i) for i in range(len(ds))}
    assert captions == {"two cats on a couch", "a woman at a table"}
    img, ids = ds[0]
    assert img.shape == (3, 8, 8) and ids.dtype == torch.long


def test_unmatched_images_raise_rather_than_shrinking_the_corpus(tmp_path):
    """A silently reduced dataset is a different experiment."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    _write_img(img_dir / f"{39769:012d}.jpg")
    _write_img(img_dir / f"{999999:012d}.jpg")      # no caption for this one
    ann = tmp_path / "caps.json"
    _write_ann(ann, {39769: ["two cats"]})
    with pytest.raises(ValueError, match="have no caption"):
        CocoCaptionsDataset(str(img_dir), str(ann), tokenize=_tok, image_transform=_tf)


def test_no_matches_at_all_raises(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    _write_img(img_dir / f"{111:012d}.jpg")
    ann = tmp_path / "caps.json"
    _write_ann(ann, {222: ["unrelated"]})
    with pytest.raises(ValueError, match="could be matched"):
        CocoCaptionsDataset(str(img_dir), str(ann), tokenize=_tok, image_transform=_tf)


def test_tokenizer_and_transform_are_required(tmp_path):
    """They must come from the MLLM; guessing would silently change the data."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    _write_img(img_dir / f"{39769:012d}.jpg")
    ann = tmp_path / "caps.json"
    _write_ann(ann, {39769: ["two cats"]})
    with pytest.raises(ValueError, match="required"):
        CocoCaptionsDataset(str(img_dir), str(ann), tokenize=None, image_transform=None)


# -- the B4 guard ----------------------------------------------------------------------
def test_caption_diversity_detects_signal(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    mapping = {}
    for i in range(4):
        _write_img(img_dir / f"{i:012d}.jpg")
        mapping[i] = [f"a distinct caption number {i}"]
    ann = tmp_path / "caps.json"
    _write_ann(ann, mapping)
    ds = CocoCaptionsDataset(str(img_dir), str(ann), tokenize=_tok, image_transform=_tf)
    assert CocoCaptionsDataset.mutual_information_sanity_check(ds) == 1.0


def test_degenerate_captions_are_detectable(tmp_path):
    """Every image sharing one caption == no image-text signal (the B4 failure)."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    mapping = {}
    for i in range(4):
        _write_img(img_dir / f"{i:012d}.jpg")
        mapping[i] = ["the same caption everywhere"]
    ann = tmp_path / "caps.json"
    _write_ann(ann, mapping)
    ds = CocoCaptionsDataset(str(img_dir), str(ann), tokenize=_tok, image_transform=_tf)
    assert CocoCaptionsDataset.mutual_information_sanity_check(ds) < 0.5


# -- factory ---------------------------------------------------------------------------
def test_coco_dataset_requires_its_inputs():
    import dataclasses

    cfg = get_preset("tiny_cpu")
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, dataset="coco"))
    with pytest.raises(ValueError, match="needs image_dir"):
        build_dataset(cfg)


def test_unknown_dataset_still_refuses():
    import dataclasses

    cfg = get_preset("tiny_cpu")
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, dataset="laion"))
    with pytest.raises(NotImplementedError, match="Refusing to substitute"):
        build_dataset(cfg)


def test_coco_is_not_a_stand_in():
    """'coco' must satisfy the experiment gate; 'dummy' must not."""
    import dataclasses

    cfg = get_preset("tiny_cpu")
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, dataset="coco"))
    assert "train.dataset" not in cfg.dummy_components()
    assert "train.dataset" in get_preset("tiny_cpu").dummy_components()
