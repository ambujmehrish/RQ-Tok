"""Decoupled training driver (Stage 0 tokenizer -> A branch -> B renderer).

One :class:`Trainer` dispatches on ``cfg.train.stage``. It is distributed-aware
(reads topology from the launcher; averages gradients across ranks; the tokenizer's
EMA codebook all-reduces internally) and checkpoints from rank 0.

Stage semantics (DESIGN.md §5):
  * ``tokenizer`` — Stage 0: EMA codebook fit on CLIP latents (no optimizer).
  * ``branch``    — Stage A: MLLM vision branch, CE(codes+halt) + flow(residual),
    targets from the frozen tokenizer; MLLM backbone frozen.
  * ``renderer``  — Stage B: latent ControlNet, rectified-flow velocity loss on the
    dequantized code space (exposure-bias-aware); backbone frozen.

Fail-loud: unknown stage / dataset / shape mismatches raise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from ..config import BifrostFlowConfig
from ..data import build_dataloader, build_dataset
from ..mllm import build_vision_gen_model
from ..renderer import build_renderer
from ..tokenizer import build_tokenizer
from ..tokenizer.encoder import DummyCLIPEncoder
from ..utils.distributed import (
    DistInfo,
    average_gradients,
    barrier,
    cleanup,
    is_main_process,
    make_sampler,
    reduce_dict,
    seed_everything,
    setup_distributed,
)

VALID_STAGES = ("tokenizer", "branch", "renderer")


@dataclass
class TrainReport:
    stage: str
    steps: int
    loss_history: list[dict[str, float]] = field(default_factory=list)
    final_loss: float = float("nan")


class Trainer:
    def __init__(self, cfg: BifrostFlowConfig, dataset_length: int = 256,
                 setup_dist: bool = True) -> None:
        if cfg.train.stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}, got {cfg.train.stage!r}")
        self.cfg = cfg
        self.stage = cfg.train.stage

        if setup_dist:
            self.info = setup_distributed(
                backend=cfg.dist.backend, init_timeout_min=cfg.dist.init_timeout_min)
        else:
            self.info = DistInfo(0, 1, 0, "gloo", torch.device(cfg.train.device), False)
        self.device = self.info.device
        seed_everything(cfg.train.seed, self.info.rank, cfg.dist.seed_per_rank)

        # Frozen tokenizer is needed by every stage (produces latents / targets).
        self.tokenizer = build_tokenizer(cfg).to(self.device)
        self.N = cfg.tokenizer.num_patches
        self.D = cfg.tokenizer.max_depth
        self.d = cfg.tokenizer.clip_dim

        self.model: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.opt: torch.optim.Optimizer | None = None
        self._build_stage()

        dataset = build_dataset(cfg, length=dataset_length)
        self.sampler = make_sampler(dataset, self.info, shuffle=True, seed=cfg.train.seed)
        self.loader = build_dataloader(cfg, dataset, sampler=self.sampler)

    # -- stage construction ------------------------------------------------------------
    def _build_stage(self) -> None:
        cfg = self.cfg
        if self.stage == "tokenizer":
            self.tokenizer.train()
            return
        self.tokenizer.eval()  # frozen: produces targets only

        if self.stage == "branch":
            self.model = build_vision_gen_model(cfg).to(self.device)
        elif self.stage == "renderer":
            self.model = build_renderer(cfg).to(self.device)
            self.vae = DummyCLIPEncoder(
                cfg.renderer.latent_dim, cfg.renderer.num_image_tokens).to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(f"stage '{self.stage}' has no trainable parameters")
        self.opt = torch.optim.AdamW(
            params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # -- target extraction (frozen tokenizer) ------------------------------------------
    def _tokenize(self, images: Tensor):
        z = self.tokenizer.encode(images)                       # [B, N, d]
        qo = self.tokenizer(z, update_codebook=False)
        b = images.shape[0]
        codes = qo.codes.reshape(b, self.N, self.D)
        zhat = qo.zhat.reshape(b, self.N, self.d)
        res = qo.res.reshape(b, self.N, self.d)
        return z, codes, zhat, res

    # -- per-step loss -----------------------------------------------------------------
    def _step_tokenizer(self, images: Tensor) -> dict[str, Tensor]:
        z = self.tokenizer.encode(images)
        out = self.tokenizer(z, update_codebook=True)           # EMA (all-reduced)
        return {k: torch.as_tensor(v, device=self.device) for k, v in out.losses.item().items()}

    def _step_branch(self, images: Tensor, text: Tensor) -> Tensor:
        _, codes, zhat, res = self._tokenize(images)
        losses = self.model.compute_loss(text, codes, res, zhat)
        self._last = losses.item()
        return losses.total

    def _step_renderer(self, images: Tensor, text: Tensor) -> Tensor:
        _, _, zhat, res = self._tokenize(images)
        control = (zhat + res)                                   # dequantized + residual
        image_latents = self.vae(images)                         # [B, L, C] target
        loss = self.model.compute_loss(image_latents, control)
        self._last = loss.item()
        return loss.flow

    # -- main loop ---------------------------------------------------------------------
    def train(self) -> TrainReport:
        report = TrainReport(stage=self.stage, steps=self.cfg.train.max_steps)
        step = 0
        last: dict[str, float] = {}
        while step < self.cfg.train.max_steps:
            if self.sampler is not None:
                self.sampler.set_epoch(step)
            for images, text in self.loader:
                if step >= self.cfg.train.max_steps:
                    break
                images = images.to(self.device)
                text = text.to(self.device)

                if self.stage == "tokenizer":
                    metrics = self._step_tokenizer(images)
                    last = {k: float(v) for k, v in metrics.items()}
                else:
                    self.opt.zero_grad()
                    loss = (self._step_branch(images, text) if self.stage == "branch"
                            else self._step_renderer(images, text))
                    loss.backward()
                    average_gradients(self.model)
                    if self.cfg.train.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.cfg.train.grad_clip)
                    self.opt.step()
                    last = self._last

                if step % self.cfg.train.log_every == 0 or step == self.cfg.train.max_steps - 1:
                    snap = self._log_snapshot(step, last)
                    if is_main_process():
                        report.loss_history.append(snap)
                step += 1

        report.final_loss = last.get("total", last.get("flow", float("nan")))
        barrier()
        return report

    def _log_snapshot(self, step: int, last: dict[str, float]) -> dict[str, float]:
        reduced = reduce_dict(
            {k: torch.tensor(v, device=self.device) for k, v in last.items()})
        snap = {"step": float(step), **{k: float(v) for k, v in reduced.items()}}
        if is_main_process():
            msg = "  ".join(f"{k}={v:.4f}" for k, v in snap.items() if k != "step")
            print(f"[{self.stage}] step {step:6d}  {msg}", flush=True)
        return snap

    # -- checkpointing -----------------------------------------------------------------
    def _trainable_module(self) -> nn.Module:
        return self.tokenizer if self.stage == "tokenizer" else self.model

    def save_checkpoint(self, step: int) -> str | None:
        if not is_main_process():
            return None
        os.makedirs(self.cfg.train.ckpt_dir, exist_ok=True)
        path = os.path.join(self.cfg.train.ckpt_dir, f"{self.stage}_step{step}.pt")
        payload = {
            "stage": self.stage,
            "step": step,
            "config_name": self.cfg.name,
            "model": self._trainable_module().state_dict(),
            "optim": self.opt.state_dict() if self.opt is not None else None,
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload["stage"] != self.stage:
            raise ValueError(
                f"checkpoint stage {payload['stage']!r} != trainer stage {self.stage!r}")
        self._trainable_module().load_state_dict(payload["model"])
        if self.opt is not None and payload.get("optim") is not None:
            self.opt.load_state_dict(payload["optim"])
        return int(payload["step"])

    def close(self) -> None:
        cleanup()
