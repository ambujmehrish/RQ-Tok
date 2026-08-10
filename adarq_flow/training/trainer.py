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
from typing import cast

import torch
from torch import Tensor, nn

from ..config import AdaRQFlowConfig
from ..data import build_dataloader, build_dataset
from ..mllm import VisionGenModel, build_vision_gen_model
from ..renderer import FlowRenderer, build_image_latent_encoder, build_renderer
from ..tokenizer import build_tokenizer
from ..utils.distributed import (
    DistInfo,
    average_gradients,
    barrier,
    broadcast_module,
    cleanup,
    is_main_process,
    make_sampler,
    reduce_dict,
    seed_everything,
    setup_distributed,
)
from ..utils.logging import get_logger

VALID_STAGES = ("tokenizer", "branch", "renderer")


@dataclass
class TrainReport:
    stage: str
    steps: int
    loss_history: list[dict[str, float]] = field(default_factory=list)
    final_loss: float = float("nan")


class Trainer:
    def __init__(self, cfg: AdaRQFlowConfig, dataset_length: int = 256,
                 setup_dist: bool = True) -> None:
        if cfg.train.stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}, got {cfg.train.stage!r}")
        # Refuse to start a run whose components contradict its declared intent.
        cfg.validate_run_mode()
        self.cfg = cfg
        self.stage = cfg.train.stage
        self.log = get_logger("trainer")

        if setup_dist:
            self.info = setup_distributed(
                backend=cfg.dist.backend, init_timeout_min=cfg.dist.init_timeout_min)
        else:
            self.info = DistInfo(0, 1, 0, "gloo", torch.device(cfg.train.device), False)
        self.device = self.info.device
        # Model construction must be seeded IDENTICALLY on every rank; only the data
        # stream is decorrelated per rank (done after the modules are built).
        seed_everything(cfg.train.seed, rank=0, seed_per_rank=False)

        # Frozen tokenizer is needed by every stage (produces latents / targets).
        self.tokenizer = build_tokenizer(cfg).to(self.device)
        self.N = cfg.tokenizer.num_patches
        self.D = cfg.tokenizer.max_depth
        self.d = cfg.tokenizer.clip_dim

        self.model: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.opt: torch.optim.Optimizer | None = None
        self._build_stage()

        # Guarantee bit-identical replicas across ranks. We average gradients manually
        # (the stage models expose compute_loss, not forward), so nothing else would
        # synchronize the initial weights — including the frozen tokenizer, whose
        # codebook otherwise produces DIFFERENT training targets on every rank.
        broadcast_module(self.tokenizer)
        if self.model is not None:
            broadcast_module(self.model)
        if self.vae is not None:
            broadcast_module(self.vae)

        # Only now decorrelate the per-rank data stream.
        seed_everything(cfg.train.seed, self.info.rank, cfg.dist.seed_per_rank)

        dataset = build_dataset(cfg, length=dataset_length)
        self.sampler = make_sampler(dataset, self.info, shuffle=True, seed=cfg.train.seed)
        self.loader = build_dataloader(cfg, dataset, sampler=self.sampler)

    # Typed accessors: model/opt exist for the branch & renderer stages (built in
    # _build_stage). These assert the invariant and narrow the Optional for callers.
    @property
    def _model(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError(f"stage '{self.stage}' has no trainable model")
        return self.model

    @property
    def _opt(self) -> torch.optim.Optimizer:
        if self.opt is None:
            raise RuntimeError(f"stage '{self.stage}' has no optimizer")
        return self.opt

    # -- stage construction ------------------------------------------------------------
    def _build_stage(self) -> None:
        cfg = self.cfg
        if self.stage == "tokenizer":
            self.tokenizer.train()
            return
        self.tokenizer.eval()  # frozen: produces targets only
        if self.stage == "renderer" and cfg.renderer.scheduled_sampling_prob > 0.0:
            raise NotImplementedError(
                "renderer.scheduled_sampling_prob > 0 requires feeding branch-sampled\n"
                "residuals into Stage B, which is not implemented. Silently ignoring it\n"
                "would misreport the exposure-bias experiment; set it to 0.0."
            )

        if self.stage == "branch":
            self.model = build_vision_gen_model(cfg).to(self.device)
        elif self.stage == "renderer":
            self.model = build_renderer(cfg).to(self.device)
            self.vae = build_image_latent_encoder(cfg.renderer).to(self.device)

        params = [p for p in self._model.parameters() if p.requires_grad]
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
        model = cast(VisionGenModel, self._model)
        _, codes, zhat, res = self._tokenize(images)
        losses = model.compute_loss(text, codes, res, zhat)
        self._last = losses.item()
        return losses.total

    def _step_renderer(self, images: Tensor, text: Tensor) -> Tensor:
        model = cast(FlowRenderer, self._model)
        _, _, zhat, res = self._tokenize(images)
        # NOTE: res is defined as z - zhat, so `zhat + res` is EXACTLY the raw continuous
        # latent z. Conditioning on it would silently reproduce the very train/inference
        # mismatch this stage exists to remove, and would make the flag below inert.
        if self.cfg.renderer.train_on_dequantized:
            control = zhat            # finite-vocabulary code space (inference-side)
        else:
            control = zhat + res      # == z, ground-truth continuous latent (baseline)
        assert self.vae is not None                              # set for renderer stage
        image_latents = self.vae(images)                         # [B, L, C] target
        loss = model.compute_loss(image_latents, control)
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
                    self._opt.zero_grad()
                    loss = (self._step_branch(images, text) if self.stage == "branch"
                            else self._step_renderer(images, text))
                    loss.backward()
                    average_gradients(self._model)
                    if self.cfg.train.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            [p for p in self._model.parameters() if p.requires_grad],
                            self.cfg.train.grad_clip)
                    self._opt.step()
                    last = self._last

                if step % self.cfg.train.log_every == 0 or step == self.cfg.train.max_steps - 1:
                    snap = self._log_snapshot(step, last)
                    if is_main_process():
                        report.loss_history.append(snap)
                step += 1

        if not last:
            raise RuntimeError(
                "training loop completed without a single step; check max_steps and that "
                "the dataloader is non-empty (drop_last can empty it)")
        if "total" in last:
            report.final_loss = last["total"]
        elif "flow" in last:
            report.final_loss = last["flow"]
        else:
            raise RuntimeError(f"no recognised loss key in {sorted(last)}")
        barrier()
        return report

    def _log_snapshot(self, step: int, last: dict[str, float]) -> dict[str, float]:
        reduced = reduce_dict(
            {k: torch.tensor(v, device=self.device) for k, v in last.items()})
        snap = {"step": float(step), **{k: float(v) for k, v in reduced.items()}}
        msg = "  ".join(f"{k}={v:.4f}" for k, v in snap.items() if k != "step")
        self.log.info("[%s] step %6d  %s", self.stage, step, msg)
        return snap

    # -- checkpointing -----------------------------------------------------------------
    def _trainable_module(self) -> nn.Module:
        return self.tokenizer if self.stage == "tokenizer" else self._model

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
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload["stage"] != self.stage:
            raise ValueError(
                f"checkpoint stage {payload['stage']!r} != trainer stage {self.stage!r}")
        self._trainable_module().load_state_dict(payload["model"])
        if self.opt is not None and payload.get("optim") is not None:
            self.opt.load_state_dict(payload["optim"])
        return int(payload["step"])

    def close(self) -> None:
        cleanup()
