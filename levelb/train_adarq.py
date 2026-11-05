#!/usr/bin/env python3
"""Level B Stage 1, AdaRQ-Flow arm: same upstream backbone, branch and training loop as the
Bifrost-1 arm (levelb.train_branch); only the bridge head and loss differ.

    python -m levelb.train_adarq --data $R/imagenet256 --tok $R/adarq_tok --out $R/branch_adarq

Identical to the Bifrost-1 arm: upstream model and branch initialisation, MAR masks, data
order (same seed), batch 128, lr 5e-4 with 5k warmup, 16 epochs, AdamW (0.9, 0.999) eps
1e-8 no weight decay, linear decay, clip 1.0, bf16 autocast, held-out monitoring only,
final checkpoint at the fixed budget. Different: upstream's linear head is replaced by the
AdaRQ heads (levelb.adarq_model) trained on Stage-0 codes + flow-matched residuals.
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
from .adarq_model import AdaRQHeads, branch_hidden, set_trainable_adarq
from .adarq_tokenizer import load_books
from .train_branch import (_STOP, LatentData, _on_signal, isolated_rng, lr_lambda, rng_state,
                           set_rng_state)
from .upstream import require_official_model


class CodedData(LatentData):
    def __init__(self, root, tok_dir):
        super().__init__(root)
        self.codes = np.load(pathlib.Path(tok_dir) / "codes_train.npy", mmap_mode="r")
        if self.codes.shape[0] != self.lat.shape[0]:
            raise SystemExit("codes and latents have different row counts")

    def batch(self, idx):
        idx = np.sort(idx)
        prompts, lat = super().batch(idx)
        return prompts, lat, torch.from_numpy(np.asarray(self.codes[idx], dtype=np.int64))


def _step_loss(model, heads, conv, prompts, lat, codes, mask, device, totals=None):
    flow = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in bb.t2i_flow(conv, prompts, lat, mask).items()}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        h = branch_hidden(model, flow, len(prompts))
        return heads.loss(h, lat.to(device), codes.to(device), mask.to(device).bool(),
                          totals=totals)


@torch.no_grad()
def evaluate(model, heads, conv, data, n, batch, device):
    """Held-out losses on a FIXED subset with FIXED masks and flow noise; ``batch`` is the
    micro-batch. Runs on its own RNG stream, so the training stream is untouched."""
    with isolated_rng(device):
        torch.manual_seed(1234)
        return _evaluate(model, heads, conv, data, n, batch, device)


def _evaluate(model, heads, conv, data, n, batch, device):
    model.eval()
    heads.eval()
    rng = np.random.default_rng(1234)
    tot = {"loss": 0.0, "code_ce": 0.0, "flow": 0.0}
    cnt = 0
    for s in range(0, n, batch):
        part = data.heldout[s:min(s + batch, n)]
        prompts, lat, codes = data.batch(part)
        mask = bb.sample_mar_mask(len(part), bb.NUM_TOKENS, rng)
        out = _step_loss(model, heads, conv, prompts, lat, codes, mask, device)
        for k in tot:
            tot[k] += float(out[k]) * len(part)
        cnt += len(part)
    model.train()
    heads.train()
    return {f"heldout_{k}": v / cnt for k, v in tot.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--tok", required=True, help="Stage-0 output dir (tokenizer.pt, codes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--processor", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--epochs", type=float, default=16)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="gradient accumulation; the loss is normalised by the WHOLE batch's "
                         "target counts, so the update equals the full-batch update")
    ap.add_argument("--flow-width", type=int, default=1024)
    ap.add_argument("--flow-depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-n", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    require_official_model(args.vlm)
    require_official_model(args.processor)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)
    device = torch.device(args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = CodedData(args.data, args.tok)
    books, res_rms, tau, _ = load_books(str(pathlib.Path(args.tok) / "tokenizer.pt"))
    steps_per_epoch = len(data.train) // args.batch
    total = int(math.ceil(args.epochs * steps_per_epoch))
    # seed BEFORE building: upstream initialises the mask token, position embedding and
    # linear head randomly (and AdaRQ its heads), so --seed must fix the initial weights
    torch.manual_seed(args.seed)
    model = bb.build_model(args.vlm, inner_dim=data.width).to(device)
    heads = AdaRQHeads(data.width, books, res_rms, args.flow_width, args.flow_depth).to(device)
    if device.type == "cpu":
        model, heads = model.float(), heads.float()
    counts = set_trainable_adarq(model, heads)
    conv = bb.make_conversation(args.processor)
    params = [p for p in model.parameters() if p.requires_grad] + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, args.warmup, total))

    rng = np.random.default_rng(args.seed)
    step, epoch, pos = 0, 0, 0
    ckpt = out / "ckpt.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        named = dict(model.named_parameters())
        for n, t in st["trainable"].items():
            named[n].data.copy_(t.to(named[n].dtype))
        heads.load_state_dict(st["heads"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        set_rng_state(st)
        step, epoch, pos = st["step"], st["epoch"], st["pos"]
        print(f"[resume] step {step} (epoch {epoch}, pos {pos})", flush=True)

    def save():
        tmp = out / "ckpt.tmp"
        torch.save({"step": step, "epoch": epoch, "pos": pos,
                    "trainable": {n: p.detach().cpu() for n, p in model.named_parameters()
                                  if p.requires_grad},
                    "heads": heads.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                    **rng_state()}, tmp)
        tmp.replace(ckpt)

    (out / "config.json").write_text(json.dumps(
        {**vars(args), "steps_total": total, "steps_per_epoch": steps_per_epoch,
         "tau": tau, "res_rms": res_rms, **counts}, indent=2))
    print(f"[adarq] {counts['trainable']:,} trainable (heads {counts['adarq_heads']:,} vs "
          f"Bifrost-1 linear head {counts['bifrost_linear_head']:,}) | {total} steps",
          flush=True)

    log = (out / "metrics.jsonl").open("a")
    model.train()
    heads.train()
    perm = np.random.default_rng(args.seed + epoch).permutation(data.train)
    started, t0 = step, time.time()
    while step < total and not _STOP["flag"]:
        if pos + args.batch > len(perm):
            epoch, pos = epoch + 1, 0
            perm = np.random.default_rng(args.seed + epoch).permutation(data.train)
        part = np.sort(perm[pos:pos + args.batch])
        pos += args.batch
        mask = bb.sample_mar_mask(len(part), bb.NUM_TOKENS, rng)
        all_codes = torch.from_numpy(np.asarray(data.codes[part], dtype=np.int64))
        totals = heads.counts(all_codes, mask.bool())
        micro = args.micro_batch or len(part)
        opt.zero_grad(set_to_none=True)
        acc = {"loss": 0.0, "code_ce": 0.0, "flow": 0.0}
        for s in range(0, len(part), micro):
            prompts, lat, codes = data.batch(part[s:s + micro])
            sub_res = _step_loss(model, heads, conv, prompts, lat, codes, mask[s:s + micro],
                                 device, totals=totals)
            if not torch.isfinite(sub_res["loss"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
            sub_res["loss"].backward()
            for k in acc:
                acc[k] += float(sub_res[k])
        res = {k: torch.tensor(v) for k, v in acc.items()}
        loss = res["loss"]
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            rate = (step - started) / (time.time() - t0)
            rec = {"step": step, "epoch": epoch, "loss": float(loss),
                   "code_ce": float(res["code_ce"]), "flow": float(res["flow"]),
                   "grad_norm": float(gnorm), "lr": sched.get_last_lr()[0], "steps_per_s": rate}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"  step {step}/{total} loss {float(loss):.4f} (ce {rec['code_ce']:.4f} "
                  f"flow {rec['flow']:.4f}) {rate:.2f} it/s", flush=True)
        if step % args.eval_every == 0 or step == total:
            val = evaluate(model, heads, conv, data, args.eval_n, args.micro_batch or args.batch,
                           device)
            log.write(json.dumps({"step": step, **val}) + "\n")
            log.flush()
            print(f"  [heldout] step {step} {val}", flush=True)
        if step % args.save_every == 0 or step == total:
            save()
        if args.max_steps is not None and step - started >= args.max_steps:
            break
    save()
    if step >= total:
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   out / "final_trainable.pt")
        torch.save(heads.state_dict(), out / "final_heads.pt")
        print(f"[adarq] finished {total} steps -> {out}", flush=True)
    else:
        print(f"[adarq] stopped at step {step}/{total}; resubmit to continue", flush=True)


if __name__ == "__main__":
    main()
