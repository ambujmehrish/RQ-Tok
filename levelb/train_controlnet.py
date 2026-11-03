#!/usr/bin/env python3
"""Level B Stage 2: train the latent ControlNet (upstream model, diffusers' recipe).

    python -m levelb.train_controlnet --data $LEVELB_ROOT/imagenet256 --out $LEVELB_ROOT/controlnet

The training step is diffusers' ``train_controlnet_flux.py`` (v0.33.1, the trainer the
paper names) with its defaults, on Bifrost-1's inputs: the control signal is the
ground-truth Qwen latents of the image (decoupled training, paper §3.3), entering through
upstream's ``NHWCUpsampleBlock``. Per step:

    x0      = pack((VAE.encode(img).latent_dist.sample() - shift) * scale)
    u       ~ logit-normal(0, 1);  t = scheduler.timesteps[int(u * 1000)];  sigma(t)
    x_t     = (1 - sigma) x0 + sigma * noise
    v       = FLUX(x_t, t, text, guidance=3.5, controlnet residuals(x_t, qwen latents))
    loss    = MSE(v, noise - x0)

AdamW lr 5e-6, betas (0.9, 0.999), wd 1e-2, eps 1e-8, grad clip 1.0, diffusers
``get_scheduler("constant")`` (which ignores the 500 warmup steps), batch 48, 2 epochs,
bf16 frozen FLUX/VAE with an fp32 ControlNet under bf16 autocast (accelerate's
mixed-precision behaviour). Text conditioning is precomputed (levelb.flux_text).
Resumable exactly; held-out flow loss is monitored with fixed noise and timesteps.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from .controlnet_build import build_controlnet
from .flux_text import from_bits
from .precompute import read_latents
from .upstream import add_upstream_to_path

_STOP = {"flag": False}


def _on_signal(signum, frame):  # noqa: ARG001
    _STOP["flag"] = True


class ControlData:
    """Crops, Qwen latents and precomputed FLUX text conditioning, by row index."""

    def __init__(self, root: str | pathlib.Path):
        root = pathlib.Path(root)
        meta = json.loads((root / "meta.json").read_text())
        self.crops = np.load(root / "crops_train.npy", mmap_mode="r")
        self.lat = np.load(root / "latents_train.npy", mmap_mode="r")
        self.lab = np.load(root / "labels_train.npy", mmap_mode="r")
        self.t5 = from_bits(np.load(root / "flux_t5.npy"))
        self.pooled = from_bits(np.load(root / "flux_pooled.npy"))
        split = np.load(root / "split_seed0.npz")
        self.train, self.heldout = split["train"], split["heldout"]
        self.width = int(meta["latent_width"])
        if not (self.crops.shape[0] == self.lat.shape[0] == self.lab.shape[0] == meta["rows"]):
            raise SystemExit(f"{root}: crops/latents/labels row counts differ")
        if self.t5.shape[0] != 1000 or self.pooled.shape[0] != 1000:
            raise SystemExit("run levelb.flux_text first: need 1000 class embeddings")

    def batch(self, idx: np.ndarray):
        idx = np.sort(idx)
        img = torch.from_numpy(np.ascontiguousarray(self.crops[idx])).permute(0, 3, 1, 2)
        img = img.float() / 127.5 - 1.0                              # ToTensor + Normalize(0.5)
        lab = torch.from_numpy(np.asarray(self.lab[idx], dtype=np.int64))
        return img, read_latents(self.lat[idx]), self.t5[lab], self.pooled[lab]


class Flux:
    """Frozen FLUX transformer + VAE + scheduler copy, with diffusers' helper functions."""

    def __init__(self, flux_id: str, device, dtype=torch.bfloat16):
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
        from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline

        self.vae = AutoencoderKL.from_pretrained(flux_id, subfolder="vae", torch_dtype=dtype)
        self.transformer = FluxTransformer2DModel.from_pretrained(
            flux_id, subfolder="transformer", torch_dtype=dtype)
        self.sched = FlowMatchEulerDiscreteScheduler.from_pretrained(flux_id, subfolder="scheduler")
        for m in (self.vae, self.transformer):
            m.requires_grad_(False)
            m.to(device)
        self.pack = FluxControlNetPipeline._pack_latents
        self.ids = FluxControlNetPipeline._prepare_latent_image_ids
        self.dtype, self.device = dtype, device

    def sigmas(self, timesteps: torch.Tensor, n_dim: int) -> torch.Tensor:
        sig = self.sched.sigmas.to(self.device, torch.float32)
        sched_t = self.sched.timesteps.to(self.device)
        idx = [(sched_t == t).nonzero().item() for t in timesteps.to(self.device)]
        s = sig[idx].flatten()
        while s.dim() < n_dim:
            s = s.unsqueeze(-1)
        return s

    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.vae.encode(img.to(self.device, self.dtype)).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        packed = self.pack(z, z.shape[0], z.shape[1], z.shape[2], z.shape[3])
        ids = self.ids(z.shape[0], z.shape[2] // 2, z.shape[3] // 2, self.device, self.dtype)
        return packed, ids


def flow_step(flux: Flux, cn, data_batch, generator: torch.Generator | None = None,
              guidance_scale: float = 3.5) -> torch.Tensor:
    """One diffusers-recipe flow-matching loss on a batch (mean over all elements)."""
    from diffusers.training_utils import compute_density_for_timestep_sampling

    img, qwen, t5, pooled = data_batch
    dev, dt = flux.device, flux.dtype
    x0, img_ids = flux.encode(img)
    side = int(math.isqrt(x0.shape[1]))
    if qwen.shape[1] * 4 != x0.shape[1] or side * side != x0.shape[1]:
        raise ValueError(f"{qwen.shape[1]} bridge tokens do not upsample 2x to the "
                         f"{x0.shape[1]}-token FLUX grid")
    # generator (CPU) only for the held-out evaluation; training uses the global RNG as
    # diffusers does
    noise = (torch.randn(x0.shape, generator=generator).to(dev, dt) if generator is not None
             else torch.randn_like(x0))
    u = compute_density_for_timestep_sampling(weighting_scheme="logit_normal",
                                              batch_size=x0.shape[0], logit_mean=0.0,
                                              logit_std=1.0, mode_scale=1.29,
                                              generator=generator)
    idx = (u * flux.sched.config.num_train_timesteps).long()
    timesteps = flux.sched.timesteps[idx].to(dev)
    sig = flux.sigmas(timesteps, x0.dim()).to(dt)      # diffusers: get_sigmas(dtype=latents)
    x_t = (1.0 - sig) * x0 + sig * noise
    guidance = (torch.full((x0.shape[0],), guidance_scale, device=dev, dtype=dt)
                if flux.transformer.config.guidance_embeds else None)
    txt = t5.to(dev, dt)
    txt_ids = torch.zeros(txt.shape[1], 3, device=dev, dtype=dt)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        blocks, singles = cn(hidden_states=x_t, controlnet_cond=qwen.to(dev, dt),
                             timestep=timesteps / 1000, guidance=guidance,
                             pooled_projections=pooled.to(dev, dt), encoder_hidden_states=txt,
                             txt_ids=txt_ids, img_ids=img_ids, return_dict=False)
    pred = flux.transformer(
        hidden_states=x_t, timestep=timesteps / 1000, guidance=guidance,
        pooled_projections=pooled.to(dev, dt), encoder_hidden_states=txt,
        controlnet_block_samples=None if blocks is None else [s.to(dt) for s in blocks],
        controlnet_single_block_samples=None if singles is None else [s.to(dt) for s in singles],
        txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
    return F.mse_loss(pred.float(), (noise - x0).float(), reduction="mean")


def _save(out, cn, opt, sched, step, epoch, pos):
    tmp = out / "ckpt.tmp"
    torch.save({"step": step, "epoch": epoch, "pos": pos, "controlnet": cn.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               tmp)
    tmp.replace(out / "ckpt.pt")


@torch.no_grad()
def heldout_loss(flux, cn, data: ControlData, n: int, batch: int) -> float:
    cn.eval()
    g = torch.Generator().manual_seed(1234)
    tot, cnt = 0.0, 0
    for s in range(0, n, batch):
        part = data.heldout[s:min(s + batch, n)]
        torch.manual_seed(1234 + s)                      # fixed VAE posterior samples
        loss = flow_step(flux, cn, data.batch(part), generator=g)
        tot += float(loss) * len(part)
        cnt += len(part)
    cn.train()
    return tot / cnt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flux", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--num-double", type=int, default=4)
    ap.add_argument("--num-single", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="split each batch for memory (gradient accumulation). NOTE: the "
                         "upsample block's BatchNorm then sees micro-batch statistics")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--grad-checkpointing", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--eval-n", type=int, default=480)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    from .upstream import require_official_flux
    require_official_flux(args.flux)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)
    add_upstream_to_path()
    from diffusers.optimization import get_scheduler

    dev = torch.device(args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = ControlData(args.data)
    flux = Flux(args.flux, dev, torch.bfloat16 if dev.type == "cuda" else torch.float32)
    cn = build_controlnet(flux.transformer, data.width, num_double=args.num_double,
                          num_single=args.num_single, seed=args.seed).to(dev).float()
    cn.train()
    if args.grad_checkpointing:
        flux.transformer.enable_gradient_checkpointing()
        cn.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(cn.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=1e-2, eps=1e-8)
    steps_per_epoch = len(data.train) // args.batch
    total = int(math.ceil(args.epochs * steps_per_epoch))
    sched = get_scheduler("constant", optimizer=opt, num_warmup_steps=500,
                          num_training_steps=total)
    micro = args.micro_batch or args.batch
    if args.batch % micro:
        raise SystemExit("--batch must be a multiple of --micro-batch")

    torch.manual_seed(args.seed)
    step, epoch, pos = 0, 0, 0
    if (out / "ckpt.pt").exists():
        st = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
        cn.load_state_dict(st["controlnet"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["torch_rng"])
        if st["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        step, epoch, pos = st["step"], st["epoch"], st["pos"]
        print(f"[resume] step {step} (epoch {epoch}, pos {pos})", flush=True)
    n_params = sum(p.numel() for p in cn.parameters())
    (out / "config.json").write_text(json.dumps(
        {**vars(args), "steps_total": total, "steps_per_epoch": steps_per_epoch,
         "controlnet_params": n_params, "bridge_width": data.width}, indent=2))
    print(f"[controlnet] {n_params:,} trainable | {total} steps ({args.epochs} epochs x "
          f"{steps_per_epoch}) | batch {args.batch} (micro {micro})", flush=True)

    log = (out / "metrics.jsonl").open("a")
    perm = np.random.default_rng(args.seed + epoch).permutation(data.train)
    started, t0 = step, time.time()
    params = list(cn.parameters())
    while step < total and not _STOP["flag"]:
        if pos + args.batch > len(perm):
            epoch, pos = epoch + 1, 0
            perm = np.random.default_rng(args.seed + epoch).permutation(data.train)
        part = perm[pos:pos + args.batch]
        pos += args.batch
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for s in range(0, len(part), micro):
            loss = flow_step(flux, cn, data.batch(part[s:s + micro])) * (micro / len(part))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            loss_sum += float(loss)
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        sched.step()
        step += 1
        if step % 25 == 0:
            rate = (step - started) / (time.time() - t0)
            log.write(json.dumps({"step": step, "epoch": epoch, "loss": loss_sum,
                                  "grad_norm": float(gnorm), "lr": sched.get_last_lr()[0],
                                  "steps_per_s": rate}) + "\n")
            log.flush()
            print(f"  step {step}/{total} loss {loss_sum:.5f} |g| {float(gnorm):.3f} "
                  f"{rate:.3f} it/s", flush=True)
        if step % args.eval_every == 0 or step == total:
            val = heldout_loss(flux, cn, data, args.eval_n, micro)
            log.write(json.dumps({"step": step, "heldout_flow_loss": val}) + "\n")
            log.flush()
            print(f"  [heldout] step {step} flow loss {val:.5f}", flush=True)
        if step % args.save_every == 0 or step == total:
            _save(out, cn, opt, sched, step, epoch, pos)
        if args.max_steps is not None and step - started >= args.max_steps:
            break
    _save(out, cn, opt, sched, step, epoch, pos)
    if step >= total:
        torch.save(cn.state_dict(), out / "final_controlnet.pt")
        print(f"[controlnet] finished {total} steps -> {out / 'final_controlnet.pt'}", flush=True)
    else:
        print(f"[controlnet] stopped at step {step}/{total}; resubmit to continue", flush=True)


if __name__ == "__main__":
    main()
