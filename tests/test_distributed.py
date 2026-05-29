"""Tests for multi-GPU primitives (CPU/gloo; requires torch).

Includes a real 2-process gloo run that verifies the EMA codebook stays identical
across ranks (the core multi-GPU correctness property).
"""

import os
import socket

import pytest

torch = pytest.importorskip("torch")

from bifrost_flow.config import DistConfig, get_preset
from bifrost_flow.utils.distributed import (
    DistInfo,
    detect_launch_env,
    setup_distributed,
    wrap_model,
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# -- env detection ---------------------------------------------------------------------
def test_detect_single_process(monkeypatch):
    for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID", "SLURM_NTASKS"]:
        monkeypatch.delenv(k, raising=False)
    env = detect_launch_env()
    assert env["world_size"] == 1 and env["rank"] == 0 and env["source"] == "single"


def test_detect_torchrun(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "3")
    env = detect_launch_env()
    assert (env["world_size"], env["rank"], env["local_rank"], env["source"]) == (
        4, 3, 3, "torchrun")


def test_detect_slurm(monkeypatch):
    for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_LOCALID", "2")
    env = detect_launch_env()
    assert (env["world_size"], env["rank"], env["source"]) == (4, 2, "slurm")


# -- setup contracts (fail-loud) -------------------------------------------------------
def test_single_process_setup_no_group(monkeypatch):
    for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID", "SLURM_NTASKS"]:
        monkeypatch.delenv(k, raising=False)
    info = setup_distributed(backend="gloo")
    assert info.world_size == 1 and not info.is_distributed and info.is_main


def test_nccl_without_cuda_raises(monkeypatch):
    if torch.cuda.is_available():
        pytest.skip("CUDA available; this checks the no-CUDA failure path")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_port()))
    with pytest.raises(RuntimeError, match="NCCL"):
        setup_distributed(backend="nccl")


def test_missing_rendezvous_raises(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    with pytest.raises(RuntimeError, match="MASTER_ADDR"):
        setup_distributed(backend="gloo")


def test_bad_backend_raises():
    with pytest.raises(ValueError):
        setup_distributed(backend="mpi")


def test_wrap_model_noop_when_not_distributed():
    info = DistInfo(0, 1, 0, "gloo", torch.device("cpu"), is_distributed=False)
    model = torch.nn.Linear(4, 4)
    assert wrap_model(model, info, DistConfig()) is model


def test_config_has_dist_block():
    cfg = get_preset("base_gpu")
    assert cfg.dist.strategy == "ddp" and cfg.dist.backend == "nccl"
    # round-trips through (de)serialization
    from bifrost_flow.config import BifrostFlowConfig

    assert BifrostFlowConfig.from_dict(cfg.to_dict()).dist.strategy == "ddp"


# -- real 2-process gloo run: codebook stays identical across ranks --------------------
def _codebook_worker(rank: int, world_size: int, port: int):
    import torch
    import torch.distributed as dist

    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    from bifrost_flow.config import TokenizerConfig
    from bifrost_flow.tokenizer import AdaptiveResidualQuantizer
    from bifrost_flow.utils.distributed import cleanup, setup_distributed

    info = setup_distributed(backend="gloo")
    assert info.is_distributed and info.world_size == world_size

    cfg = TokenizerConfig(clip_dim=8, num_patches=4, codebook_size=16, max_depth=3)
    q = AdaptiveResidualQuantizer(cfg)
    q.train()

    # Per-rank-different data (real sharding); codebooks must still converge identically.
    torch.manual_seed(100 + rank)
    for _ in range(10):
        q(torch.randn(64, 8), update_codebook=True)

    embed = q.codebooks[0].embed
    ref = embed.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    ref /= world_size
    max_dev = (embed - ref).abs().max().item()
    cleanup()
    assert max_dev < 1e-6, f"rank {rank} codebook diverged: max_dev={max_dev}"


def test_codebook_identical_across_ranks():
    import torch.multiprocessing as mp

    world_size = 2
    port = _free_port()
    mp.spawn(_codebook_worker, args=(world_size, port), nprocs=world_size, join=True)
