"""Multi-GPU / distributed primitives (launcher-agnostic).

Works under both ``torchrun`` and SLURM ``srun`` (Cineca Leonardo uses SLURM).
Topology is read from the environment — never hard-coded:

  * torchrun sets ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT``;
  * SLURM (srun) sets ``SLURM_PROCID`` / ``SLURM_NTASKS`` / ``SLURM_LOCALID`` — the sbatch
    template additionally exports ``MASTER_ADDR`` / ``MASTER_PORT``.

Fail-loud, no silent fallbacks: a multi-rank world with a missing rendezvous address,
or an NCCL backend without CUDA, raises immediately rather than degrading.

This module imports torch; the ``adarq_flow.utils`` package stays torch-free, so
import it explicitly: ``from adarq_flow.utils.distributed import setup_distributed``.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: torch.device
    is_distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"environment variable {name}={v!r} is not an int") from e


def detect_launch_env() -> dict[str, int | str | None]:
    """Resolve (rank, world_size, local_rank) from torchrun or SLURM.

    torchrun env wins if present; otherwise SLURM. Returns a dict with the resolved
    values plus the rendezvous ``master_addr`` / ``master_port`` (may be ``None`` for
    a single-process run). Does not initialize anything.
    """
    rank = _env_int("RANK")
    world_size = _env_int("WORLD_SIZE")
    local_rank = _env_int("LOCAL_RANK")
    source = "torchrun"

    if world_size is None:  # not torchrun -> try SLURM
        rank = _env_int("SLURM_PROCID")
        world_size = _env_int("SLURM_NTASKS")
        local_rank = _env_int("SLURM_LOCALID")
        source = "slurm"

    if world_size is None:  # neither -> single process
        return {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "master_addr": None,
            "master_port": None,
            "source": "single",
        }

    return {
        "rank": rank if rank is not None else 0,
        "world_size": world_size,
        "local_rank": local_rank if local_rank is not None else 0,
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "source": source,
    }


def setup_distributed(backend: str = "nccl", init_timeout_min: int = 30) -> DistInfo:
    """Initialize the process group from the launcher environment.

    Single-process (``world_size == 1``): no process group is created and a CPU/GPU
    ``DistInfo`` is returned. Multi-rank: validates the rendezvous and CUDA/backend
    consistency, then calls ``init_process_group``.
    """
    if backend not in ("nccl", "gloo"):
        raise ValueError(f"backend must be 'nccl' or 'gloo', got {backend!r}")

    env = detect_launch_env()
    # rank/world_size/local_rank are always populated ints (see detect_launch_env).
    world_size = int(env["world_size"])  # type: ignore[arg-type]
    rank = int(env["rank"])  # type: ignore[arg-type]
    local_rank = int(env["local_rank"])  # type: ignore[arg-type]

    if world_size == 1:
        device = _select_device(backend, local_rank, allow_cpu=True)
        return DistInfo(0, 1, 0, backend, device, is_distributed=False)

    # --- multi-rank: validate before init (no fallbacks) ---
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError(
            "NCCL backend requested but CUDA is unavailable. Use backend='gloo' for "
            "CPU multi-process, or run on GPU nodes."
        )
    if not env.get("master_addr") or not env.get("master_port"):
        raise RuntimeError(
            "Distributed run (world_size>1) is missing MASTER_ADDR/MASTER_PORT. "
            "Under torchrun these are set automatically; under SLURM, export them in "
            "your sbatch script (see scripts/cineca_leonardo_4xA100.sbatch)."
        )

    device = _select_device(backend, local_rank, allow_cpu=(backend == "gloo"))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
            timeout=_dt.timedelta(minutes=init_timeout_min),
        )
    return DistInfo(rank, world_size, local_rank, backend, device, is_distributed=True)


def _select_device(backend: str, local_rank: int, allow_cpu: bool) -> torch.device:
    if backend == "nccl":
        if not torch.cuda.is_available():
            if allow_cpu:
                return torch.device("cpu")
            raise RuntimeError("NCCL backend requires CUDA, which is unavailable")
        return torch.device(f"cuda:{local_rank}")
    # gloo
    return torch.device("cpu")


# -- collectives / queries --------------------------------------------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """In-place mean all-reduce across ranks (no-op single-process)."""
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    return tensor


def reduce_dict(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Mean-reduce a dict of scalar tensors across ranks (for logging)."""
    if not is_dist():
        return values
    keys = sorted(values)
    packed = torch.stack([values[k].detach().float() for k in keys])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed /= get_world_size()
    return {k: packed[i] for i, k in enumerate(keys)}


def average_gradients(module: torch.nn.Module) -> None:
    """Mean-reduce ``.grad`` of every parameter across ranks (after backward).

    Used instead of a DDP wrapper because the stage models expose custom
    ``compute_loss`` methods (not ``forward``). No-op when not distributed.
    """
    if not is_dist():
        return
    ws = get_world_size()
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= ws


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# -- model / data wrapping --------------------------------------------------------------
def seed_everything(base_seed: int, rank: int = 0, seed_per_rank: bool = True) -> int:
    """Seed RNGs. With ``seed_per_rank`` the *data* RNG differs per rank while model
    init stays synchronized (DDP/FSDP broadcast parameters from rank 0 anyway).
    Returns the effective seed used."""
    seed = base_seed + (rank if seed_per_rank else 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def wrap_model(model: torch.nn.Module, info: DistInfo, dist_cfg) -> torch.nn.Module:
    """Wrap a model for the configured strategy. No-op if not distributed."""
    if not info.is_distributed or dist_cfg.strategy == "none":
        return model

    if dist_cfg.strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        device_ids = [info.local_rank] if info.device.type == "cuda" else None
        return DDP(
            model,
            device_ids=device_ids,
            output_device=info.local_rank if info.device.type == "cuda" else None,
            find_unused_parameters=dist_cfg.find_unused_parameters,
            bucket_cap_mb=dist_cfg.bucket_cap_mb,
            gradient_as_bucket_view=True,
        )

    if dist_cfg.strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        strat = {
            "full": ShardingStrategy.FULL_SHARD,
            "grad_op": ShardingStrategy.SHARD_GRAD_OP,
        }.get(dist_cfg.fsdp_sharding)
        if strat is None:
            raise ValueError(f"unknown fsdp_sharding {dist_cfg.fsdp_sharding!r}")
        mp = None
        if dist_cfg.fsdp_mixed_precision:
            from torch.distributed.fsdp import MixedPrecision

            mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        return FSDP(model, sharding_strategy=strat, mixed_precision=mp,
                    device_id=info.local_rank if info.device.type == "cuda" else None)

    raise ValueError(f"unknown dist strategy {dist_cfg.strategy!r}")


def make_sampler(dataset: Any, info: DistInfo, shuffle: bool = True, seed: int = 0):
    """A ``DistributedSampler`` when distributed, else ``None`` (let the loader decide)."""
    if not info.is_distributed:
        return None
    from torch.utils.data import DistributedSampler

    return DistributedSampler(
        dataset,
        num_replicas=info.world_size,
        rank=info.rank,
        shuffle=shuffle,
        seed=seed,
    )
