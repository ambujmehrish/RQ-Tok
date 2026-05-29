"""Phase 4 tests: datasets + decoupled training stages (torch, CPU)."""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from bifrost_flow.config import get_preset
from bifrost_flow.data import DummyImageTextDataset, build_dataset
from bifrost_flow.training import VALID_STAGES, Trainer


def _cfg(stage, max_steps=20, batch=4):
    cfg = get_preset("tiny_cpu")
    return dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, stage=stage, max_steps=max_steps,
                                       batch_size=batch, lr=1e-2, log_every=5))


def _trainer(stage, **kw):
    # setup_dist=False -> single-process CPU, no process group.
    return Trainer(_cfg(stage, **kw), dataset_length=64, setup_dist=False)


# -- data ------------------------------------------------------------------------------
def test_dummy_dataset_deterministic():
    ds = DummyImageTextDataset(length=10, image_size=8, text_len=5)
    img0, txt0 = ds[0]
    img0b, txt0b = ds[0]
    assert img0.shape == (3, 8, 8) and txt0.shape == (5,)
    assert torch.equal(img0, img0b) and torch.equal(txt0, txt0b)
    assert not torch.equal(ds[0][0], ds[1][0])


def test_real_dataset_raises():
    cfg = get_preset("base_gpu")  # dataset == "blip3o"
    with pytest.raises(NotImplementedError):
        build_dataset(cfg)


def test_valid_stages():
    assert set(VALID_STAGES) == {"tokenizer", "branch", "renderer"}


def test_bad_stage_raises():
    cfg = _cfg("nonsense")
    with pytest.raises(ValueError):
        Trainer(cfg, setup_dist=False)


# -- stages run and learn --------------------------------------------------------------
def test_stage_tokenizer_runs():
    t = _trainer("tokenizer", max_steps=30)
    rep = t.train()
    assert rep.stage == "tokenizer" and len(rep.loss_history) > 0
    assert all("recon" in s for s in rep.loss_history)
    # EMA reduces reconstruction over training.
    assert rep.loss_history[-1]["recon"] < rep.loss_history[0]["recon"]


def test_stage_branch_decreases():
    torch.manual_seed(0)
    t = _trainer("branch", max_steps=40)
    rep = t.train()
    assert rep.loss_history[-1]["total"] < rep.loss_history[0]["total"]


def test_stage_renderer_decreases():
    torch.manual_seed(0)
    t = _trainer("renderer", max_steps=40)
    rep = t.train()
    assert rep.loss_history[-1]["flow"] < rep.loss_history[0]["flow"]


# -- checkpointing ---------------------------------------------------------------------
def test_checkpoint_save_and_load(tmp_path):
    cfg = _cfg("branch", max_steps=10)
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train,
                                                             ckpt_dir=str(tmp_path)))
    t = Trainer(cfg, dataset_length=32, setup_dist=False)
    t.train()
    path = t.save_checkpoint(10)
    assert path is not None

    t2 = Trainer(cfg, dataset_length=32, setup_dist=False)
    step = t2.load_checkpoint(path)
    assert step == 10
    # loaded weights match
    a = dict(t.model.named_parameters())
    b = dict(t2.model.named_parameters())
    assert all(torch.allclose(a[k], b[k]) for k in a)


def test_checkpoint_stage_mismatch_raises(tmp_path):
    cfg_b = _cfg("branch", max_steps=4)
    cfg_b = dataclasses.replace(cfg_b, train=dataclasses.replace(cfg_b.train,
                                                                 ckpt_dir=str(tmp_path)))
    tb = Trainer(cfg_b, dataset_length=16, setup_dist=False)
    tb.train()
    path = tb.save_checkpoint(4)

    cfg_r = _cfg("renderer", max_steps=4)
    cfg_r = dataclasses.replace(cfg_r, train=dataclasses.replace(cfg_r.train,
                                                                 ckpt_dir=str(tmp_path)))
    tr = Trainer(cfg_r, dataset_length=16, setup_dist=False)
    with pytest.raises(ValueError):
        tr.load_checkpoint(path)
