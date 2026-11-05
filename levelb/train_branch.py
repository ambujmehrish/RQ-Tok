#!/usr/bin/env python3
"""Level B Stage 1: train the Bifrost-1 vision branch (upstream model, our loop).

    python -m levelb.train_branch --data $LEVELB_ROOT/imagenet256 --out $LEVELB_ROOT/branch_bifrost

Upstream's ``MultiModalityCausalLM`` and its training ``forward`` (MSE over all 64
positions) on the precomputed Qwen latents, prompts = class names, MAR masks. Settings
(EXPERIMENTS.md Level B spec): batch 128, lr 5e-4 with 5k linear warmup [upstream path
name], 16 epochs [paper §5.1]; optimiser and schedule are the HF ``Trainer`` defaults
upstream's ``TrainingArguments`` implies (AdamW (0.9, 0.999), eps 1e-8, no weight decay,
linear decay, grad-norm clip 1.0, bf16 autocast).

Resumable: the trainable weights, optimiser, scheduler and RNG state are checkpointed
every ``--save-every`` steps and on SIGTERM/SIGUSR1 (SLURM's pre-kill signal), and a rerun
continues from the latest checkpoint. The held-out slice is evaluated every
``--eval-every`` steps for monitoring; the test uses the FINAL checkpoint at the fixed
budget, the same rule for both arms.
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

from . import bifrost_build as bb
from .precompute import read_latents
from .upstream import require_official_model

_STOP = {"flag": False}


def _on_signal(signum, frame):  # noqa: ARG001
    _STOP["flag"] = True


def isolated_rng(device: torch.device):
    """Held-out evaluation draws from its own seeds; this keeps it from moving or reseeding
    the training stream (CPU and CUDA), so training is the same with or without evals."""
    return torch.random.fork_rng(devices=[device] if device.type == "cuda" else [])


def rng_state() -> dict:
    return {"torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])


class LatentData:
    """Precomputed latents + class prompts, read by index from the memmapped arrays."""

    def __init__(self, root: str | pathlib.Path):
        root = pathlib.Path(root)
        meta = json.loads((root / "meta.json").read_text())
        self.lat = np.load(root / "latents_train.npy", mmap_mode="r")
        self.lab = np.load(root / "labels_train.npy", mmap_mode="r")
        self.names = json.loads((root / "class_names.json").read_text())
        split = np.load(root / "split_seed0.npz")
        self.train, self.heldout = split["train"], split["heldout"]
        if self.lat.shape[0] != meta["rows"] or len(self.names) != 1000:
            raise SystemExit(f"{root}: latents/rows/class-name mismatch")
        self.width = int(meta["latent_width"])

    def batch(self, idx: np.ndarray) -> tuple[list[str], torch.Tensor]:
        idx = np.sort(idx)                                   # sequential-ish memmap reads
        lat = read_latents(self.lat[idx])
        prompts = [self.names[int(c)] for c in self.lab[idx]]
        return prompts, lat


def lr_lambda(step: int, warmup: int, total: int) -> float:
    """HF ``get_linear_schedule_with_warmup``."""
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def _save(out: pathlib.Path, model, opt, sched, step: int, rng: np.random.Generator,
          epoch: int, pos: int) -> None:
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    tmp = out / "ckpt.tmp"
    torch.save({"step": step, "epoch": epoch, "pos": pos, "trainable": trainable,
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng": rng.bit_generator.state, **rng_state()}, tmp)
    tmp.replace(out / "ckpt.pt")


@torch.no_grad()
def evaluate(model, conv, data: LatentData, n: int, batch: int, device) -> float:
    """MSE on a FIXED subset of the held-out slice with FIXED masks (comparable over time).
    ``batch`` is the micro-batch: no-grad memory, not the training batch, sets it."""
    with isolated_rng(device):
        return _evaluate(model, conv, data, n, batch, device)


def _evaluate(model, conv, data: LatentData, n: int, batch: int, device) -> float:
    model.eval()
    rng = np.random.default_rng(1234)
    idx = data.heldout[:n]
    total, count = 0.0, 0
    for s in range(0, len(idx), batch):
        part = idx[s:s + batch]
        prompts, lat = data.batch(part)
        mask = bb.sample_mar_mask(len(part), bb.NUM_TOKENS, rng)
        flow = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in bb.t2i_flow(conv, prompts, lat.to(device), mask).items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(t2i_flow=flow, **bb.forward_kwargs(len(part)))["loss"]
        total += float(loss) * len(part)
        count += len(part)
    model.train()
    return total / count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--processor", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--epochs", type=float, default=16)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="gradient accumulation: split each batch into micro-batches of this "
                         "size. The batch is sorted and its masks drawn ONCE, then split, and "
                         "each part's mean loss is weighted by its share, so the update is the "
                         "full-batch update whatever the split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-n", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after this many steps in THIS run (resumable); the schedule "
                         "still spans the full budget")
    args = ap.parse_args()
    require_official_model(args.vlm)
    require_official_model(args.processor)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)
    device = torch.device(args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = LatentData(args.data)
    steps_per_epoch = len(data.train) // args.batch
    total = int(math.ceil(args.epochs * steps_per_epoch))
    # seed BEFORE building: upstream initialises the mask token, position embedding and
    # linear head randomly (and AdaRQ its heads), so --seed must fix the initial weights
    torch.manual_seed(args.seed)
    model = bb.build_model(args.vlm, inner_dim=data.width).to(device)
    if device.type == "cpu":          # tests only: CPU has no bf16 autocast for these ops
        model = model.float()
    counts = bb.set_trainable(model)
    conv = bb.make_conversation(args.processor)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, args.warmup, total))

    rng = np.random.default_rng(args.seed)
    step, epoch, pos = 0, 0, 0
    ckpt = out / "ckpt.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        named = dict(model.named_parameters())
        for n, t in state["trainable"].items():
            named[n].data.copy_(t.to(named[n].dtype))
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        rng.bit_generator.state = state["rng"]
        set_rng_state(state)
        step, epoch, pos = state["step"], state["epoch"], state["pos"]
        print(f"[resume] step {step} (epoch {epoch}, pos {pos})", flush=True)
    (out / "config.json").write_text(json.dumps(
        {**vars(args), "steps_total": total, "steps_per_epoch": steps_per_epoch, **counts},
        indent=2))
    print(f"[branch] {counts['trainable']:,} trainable / {counts['frozen']:,} frozen | "
          f"{total} steps ({args.epochs} epochs x {steps_per_epoch})", flush=True)

    log = (out / "metrics.jsonl").open("a")
    model.train()
    perm = np.random.default_rng(args.seed + epoch).permutation(data.train)
    started, t0 = step, time.time()
    while step < total and not _STOP["flag"]:
        if pos + args.batch > len(perm):
            epoch, pos = epoch + 1, 0
            perm = np.random.default_rng(args.seed + epoch).permutation(data.train)
        part = np.sort(perm[pos:pos + args.batch])
        pos += args.batch
        mask = bb.sample_mar_mask(len(part), bb.NUM_TOKENS, rng)
        micro = args.micro_batch or len(part)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for s in range(0, len(part), micro):
            sub_idx, sub_mask = part[s:s + micro], mask[s:s + micro]
            prompts, lat = data.batch(sub_idx)
            flow = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in bb.t2i_flow(conv, prompts, lat, sub_mask).items()}
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                sub_loss = model(t2i_flow=flow, **bb.forward_kwargs(len(sub_idx)))["loss"]
            if not torch.isfinite(sub_loss):
                raise FloatingPointError(f"non-finite loss {sub_loss.item()} at step {step}")
            (sub_loss * (len(sub_idx) / len(part))).backward()
            loss_sum += float(sub_loss) * len(sub_idx) / len(part)
        loss = torch.tensor(loss_sum)
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            rate = (step - started) / (time.time() - t0)
            rec = {"step": step, "epoch": epoch, "loss": float(loss), "grad_norm": float(gnorm),
                   "lr": sched.get_last_lr()[0], "steps_per_s": rate}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"  step {step}/{total} loss {float(loss):.5f} |g| {float(gnorm):.3f} "
                  f"lr {rec['lr']:.2e} {rate:.2f} it/s", flush=True)
        if step % args.eval_every == 0 or step == total:
            val = evaluate(model, conv, data, args.eval_n, args.micro_batch or args.batch,
                           device)
            log.write(json.dumps({"step": step, "heldout_mse": val}) + "\n")
            log.flush()
            print(f"  [heldout] step {step} mse {val:.5f}", flush=True)
        if step % args.save_every == 0 or step == total:
            _save(out, model, opt, sched, step, rng, epoch, pos)
        if args.max_steps is not None and step - started >= args.max_steps:
            break
    _save(out, model, opt, sched, step, rng, epoch, pos)
    if step >= total:
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   out / "final_trainable.pt")
        print(f"[branch] finished {total} steps -> {out / 'final_trainable.pt'}", flush=True)
    else:
        print(f"[branch] stopped at step {step}/{total}; resubmit to continue", flush=True)


if __name__ == "__main__":
    main()
