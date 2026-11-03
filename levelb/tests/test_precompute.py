"""levelb.precompute on a tiny parquet in the HF ImageNet layout (code test only)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from levelb import imagenet as inet
from levelb.precompute import read_latents
from levelb.upstream import add_upstream_to_path

from .test_bifrost_build import tiny_vlm  # noqa: F401  (fixture)


def _jpeg(w: int, h: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, w)[None, :, None] * np.ones((h, 1, 3))
    x = (x + rng.integers(0, 40, size=(h, w, 3))).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(x).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture(scope="module")
def tiny_imagenet(tmp_path_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path_factory.mktemp("imagenet")
    (root / "data").mkdir()
    sizes = [(320, 240), (240, 320), (500, 375)]
    for shard, rows in enumerate([[0, 1], [2]]):
        blobs = [_jpeg(*sizes[i], seed=i) for i in rows]
        table = pa.table({
            "image": pa.array([{"bytes": b, "path": f"{i}.jpg"}
                               for i, b in zip(rows, blobs, strict=True)]),
            "label": pa.array([7 * i for i in rows], type=pa.int64())})
        pq.write_table(table, root / "data" / f"train-{shard:05d}-of-00002.parquet")
    return root, [_jpeg(*s, seed=i) for i, s in enumerate(sizes)]


@pytest.fixture(scope="module")
def tiny_vlm_with_processor(tiny_vlm):  # noqa: F811
    from transformers import AutoProcessor

    AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct").save_pretrained(tiny_vlm)
    return tiny_vlm


def _run(imagenet_root, out, vlm):
    env = dict(os.environ)
    subprocess.run([sys.executable, "-m", "levelb.precompute", "--imagenet", str(imagenet_root),
                    "--out", str(out), "--vlm", vlm, "--batch", "2", "--workers", "1",
                    "--device", "cpu", "--expect-total", "3"], check=True, env=env,
                   capture_output=True, text=True)


def test_precompute_matches_direct_computation(tiny_imagenet, tiny_vlm_with_processor, tmp_path):
    root, blobs = tiny_imagenet
    out = tmp_path / "pre"
    _run(root, out, tiny_vlm_with_processor)

    crops = np.load(out / "crops_train.npy")
    labels = np.load(out / "labels_train.npy")
    lats = read_latents(np.load(out / "latents_train.npy"))
    assert crops.shape == (3, 256, 256, 3) and labels.tolist() == [0, 7, 14]
    for i, b in enumerate(blobs):
        want = np.asarray(inet.center_crop(Image.open(io.BytesIO(b))), dtype=np.uint8)
        assert np.array_equal(crops[i], want)

    add_upstream_to_path()
    from bifrost.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers import AutoProcessor

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(tiny_vlm_with_processor,
                                                             torch_dtype=torch.bfloat16)
    proc = AutoProcessor.from_pretrained(tiny_vlm_with_processor).image_processor
    def direct(idx):
        views = [inet.qwen_view(Image.fromarray(crops[i])) for i in idx]
        px = proc(images=views, return_tensors="pt")
        with torch.no_grad():
            z = vlm.visual.eval()(px["pixel_values"].to(torch.bfloat16),
                                  grid_thw=px["image_grid_thw"])
        return z.reshape(len(idx), 64, -1).to(torch.bfloat16)

    # exact, batch for batch (shard 0 = rows 0,1 in one batch; shard 1 = row 2)
    assert torch.equal(lats[:2], direct([0, 1]))
    assert torch.equal(lats[2:], direct([2]))
    # and independent of batch composition up to bf16 rounding
    allz = direct([0, 1, 2]).float()
    assert (lats.float() - allz).abs().max() <= 2e-2 * allz.abs().max()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["rows"] == 3 and meta["latent_tokens"] == 64

    # resume: a rerun finds every shard done and leaves the arrays untouched
    before = (out / "latents_train.npy").stat().st_mtime_ns
    _run(root, out, tiny_vlm_with_processor)
    assert (out / "latents_train.npy").stat().st_mtime_ns == before


def test_wrong_total_is_refused(tiny_imagenet, tiny_vlm_with_processor, tmp_path):
    root, _ = tiny_imagenet
    r = subprocess.run([sys.executable, "-m", "levelb.precompute", "--imagenet", str(root),
                        "--out", str(tmp_path / "x"), "--vlm", tiny_vlm_with_processor,
                        "--device", "cpu", "--expect-total", "4"], capture_output=True, text=True)
    assert r.returncode != 0 and "expected 4" in (r.stderr + r.stdout)
