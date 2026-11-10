#!/usr/bin/env python3
"""Why does the AdaRQ-Flow code head overfit?  (diagnostic; held-out slice + train subset)

    python -m levelb.adarq_diag --data $R/imagenet256 --tok $R/adarq_tok --run $R/branch_adarq

Two hypotheses, each with a measurement on the trained model (final weights, or the latest
checkpoint if the run has not finished):

H1  memorisation through the VISIBLE tokens. MAR leaves 0-30% of an image's latents
    unmasked; that can identify a training image, whose exact codes the head then recalls.
    Measure: code cross-entropy on TRAIN vs HELD-OUT images at fixed mask ratios
    0.70 / 0.80 / 0.90 / 1.00. H1 predicts a train/held-out gap that shrinks as fewer
    tokens are visible and is ~0 at 1.00 (nothing visible: only the class name).

H2  the hard labels are arbitrary near codeword ties. Measure on held-out: for each wrong
    top-1 prediction, its distance to the residual relative to the true code's
    (1.00 = equally good); the share of positions where the tokenizer's own choice is a
    near-tie (2nd-nearest / nearest distance < 1.05, 1.10); and the probability the head
    puts on codewords within 10% of the true code's distance. Also the median gap
    d2^2 - d1^2 per level on train: the natural temperature for distance-based soft targets.

A check that the diagnostic reproduces the tokenizer: the nearest codeword to each
reconstructed residual must equal the stored code (reported; < 0.99 raises).
Nothing here touches the test set or changes a model.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from . import bifrost_build as bb
from .adarq_model import AdaRQHeads, branch_hidden
from .adarq_tokenizer import load_books
from .train_adarq import CodedData
from .upstream import require_official_model

RATIOS = (0.70, 0.80, 0.90, 1.00)


def f3(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def fixed_ratio_mask(bsz: int, ratio: float, rng: np.random.Generator) -> torch.Tensor:
    k = int(math.ceil(bb.NUM_TOKENS * ratio))
    mask = torch.zeros(bsz, bb.NUM_TOKENS)
    for b in range(bsz):
        mask[b, rng.permutation(bb.NUM_TOKENS)[:k]] = 1.0
    return mask


def load_run(run: pathlib.Path, vlm: str, width: int, books, res_rms, device):
    cfg = json.loads((run / "config.json").read_text())
    model = bb.build_model(vlm, inner_dim=width)
    heads = AdaRQHeads(width, books, res_rms, int(cfg["flow_width"]), int(cfg["flow_depth"]))
    if (run / "final_trainable.pt").exists() and (run / "final_heads.pt").exists():
        trainable = torch.load(run / "final_trainable.pt", map_location="cpu")
        heads.load_state_dict(torch.load(run / "final_heads.pt", map_location="cpu"))
        step, which = int(cfg["steps_total"]), "final"
    else:
        st = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
        trainable, step, which = st["trainable"], int(st["step"]), "latest checkpoint"
        heads.load_state_dict(st["heads"])
    named = dict(model.named_parameters())
    for n, t in trainable.items():
        if n not in named:
            raise KeyError(f"unknown parameter {n}")
        named[n].data.copy_(t.to(named[n].dtype))
    return model.to(device).eval(), heads.to(device).eval(), step, which


class Acc:
    """Running sums for one (split, ratio) cell."""

    def __init__(self, depth: int) -> None:
        self.ce = np.zeros(depth)
        self.n = np.zeros(depth)
        self.ce_codes = 0.0          # code levels only (no <halt> targets)
        self.n_codes = 0
        self.top1 = 0
        self.wrong_ratio: list[np.ndarray] = []   # d(pred)/d(true) for wrong predictions
        self.pred_halt = 0                         # wrong predictions that were <halt>
        self.tie: list[np.ndarray] = []            # d2/d1 of the tokenizer's choice
        self.near_mass: list[np.ndarray] = []      # prob. on codewords within 10% of true
        self.gap_sq: dict[int, list[np.ndarray]] = {}
        self.argmin_ok = 0

    def summary(self) -> dict:
        wr = np.concatenate(self.wrong_ratio) if self.wrong_ratio else np.zeros(0)
        tie = np.concatenate(self.tie) if self.tie else np.zeros(0)
        nm = np.concatenate(self.near_mass) if self.near_mass else np.zeros(0)
        n_wrong = len(wr) + self.pred_halt
        return {
            "ce_per_level": (self.ce / np.maximum(self.n, 1)).tolist(),
            "targets_per_level": self.n.astype(int).tolist(),
            "ce_all": float(self.ce.sum() / max(self.n.sum(), 1)),
            "ce_code_levels": self.ce_codes / max(self.n_codes, 1),
            "top1_acc_code_levels": self.top1 / max(self.n_codes, 1),
            "wrong_predictions": int(n_wrong),
            "wrong_pred_halt_share": self.pred_halt / max(n_wrong, 1),
            "wrong_within_5pct": float((wr < 1.05).mean()) if len(wr) else None,
            "wrong_within_10pct": float((wr < 1.10).mean()) if len(wr) else None,
            "wrong_within_20pct": float((wr < 1.20).mean()) if len(wr) else None,
            "wrong_ratio_median": float(np.median(wr)) if len(wr) else None,
            "tokenizer_near_tie_5pct": float((tie < 1.05).mean()) if len(tie) else None,
            "tokenizer_near_tie_10pct": float((tie < 1.10).mean()) if len(tie) else None,
            "prob_mass_within_10pct": float(nm.mean()) if len(nm) else None,
            "argmin_matches_stored_code": self.argmin_ok / max(self.n_codes, 1),
            "median_gap_sq_per_level": {str(k): float(np.median(np.concatenate(v)))
                                        for k, v in sorted(self.gap_sq.items())},
        }


@torch.no_grad()
def measure(model, heads, conv, data, idx, ratio, batch, device, seed) -> dict:
    rng = np.random.default_rng(seed)
    acc = Acc(heads.D)
    books = heads.books.float()
    c_sq = books.pow(2).sum(-1)                                     # [D, K]
    for s in range(0, len(idx), batch):
        part = np.sort(idx[s:s + batch])
        prompts, lat, codes = data.batch(part)
        mask = fixed_ratio_mask(len(part), ratio, rng)
        flow = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in bb.t2i_flow(conv, prompts, lat, mask).items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = branch_hidden(model, flow, len(part))
        m = mask.bool().to(device)
        h, z = h[m].float(), lat.to(device)[m].float()
        cd = codes.to(device)[m].long()
        pre = heads.prefixes(cd)                                    # [M, D+1, d]
        depth = (cd != heads.K).sum(-1)
        for lvl in range(heads.D):
            valid = lvl <= depth
            if not valid.any():
                continue
            logits = heads.code_logits(h[valid], pre[valid, lvl], lvl)    # [V, K+1] fp32
            y = cd[valid, lvl]
            ce = F.cross_entropy(logits, y, reduction="none")
            acc.ce[lvl] += float(ce.sum())
            acc.n[lvl] += int(valid.sum())
            code = y != heads.K                                     # levels with a real code
            if not code.any():
                continue
            lg, yc = logits[code], y[code]
            acc.ce_codes += float(ce[code].sum())
            acc.n_codes += int(code.sum())
            r = (z[valid] - pre[valid, lvl])[code]                  # residual before this level
            d2 = (r.pow(2).sum(-1, keepdim=True) - 2 * r @ books[lvl].t() + c_sq[lvl]).clamp_min(0)
            two = d2.topk(2, dim=-1, largest=False)
            acc.argmin_ok += int((two.indices[:, 0] == yc).sum())
            d_true = d2.gather(1, yc[:, None]).squeeze(1)
            acc.tie.append((two.values[:, 1] / two.values[:, 0].clamp_min(1e-12)).sqrt().cpu().numpy())
            acc.gap_sq.setdefault(lvl, []).append((two.values[:, 1] - two.values[:, 0]).cpu().numpy())
            pred = lg.argmax(-1)
            acc.top1 += int((pred == yc).sum())
            wrong = pred != yc
            halt = wrong & (pred == heads.K)
            acc.pred_halt += int(halt.sum())
            wc = wrong & ~halt
            if wc.any():
                d_pred = d2[wc].gather(1, pred[wc][:, None]).squeeze(1)
                acc.wrong_ratio.append((d_pred / d_true[wc].clamp_min(1e-12)).sqrt().cpu().numpy())
            near = d2 <= (1.10 ** 2) * d_true[:, None]               # includes the true code
            prob = torch.softmax(lg, -1)[:, :heads.K]
            acc.near_mass.append((prob * near).sum(-1).cpu().numpy())
    return acc.summary()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--run", required=True, help="AdaRQ-Flow Stage-1 run folder")
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--processor", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--n", type=int, default=2048, help="images per split")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="default: <run>/diag/overfit_diag.json")
    args = ap.parse_args()
    require_official_model(args.vlm)
    require_official_model(args.processor)

    device = torch.device(args.device)
    run = pathlib.Path(args.run)
    data = CodedData(args.data, args.tok)
    books, res_rms, tau, _ = load_books(str(pathlib.Path(args.tok) / "tokenizer.pt"))
    model, heads, step, which = load_run(run, args.vlm, data.width, books, res_rms, device)
    if device.type == "cpu":
        model, heads = model.float(), heads.float()
    conv = bb.make_conversation(args.processor)
    n = min(args.n, len(data.heldout))
    splits = {"heldout": np.asarray(data.heldout[:n]),
              "train": np.random.default_rng(7).choice(data.train, n, replace=False)}
    print(f"[diag] {run} at step {step} ({which}); {n} images per split", flush=True)

    res = {"run": str(run), "step": step, "weights": which, "images_per_split": n,
           "ratios": list(RATIOS), "cells": {}}
    for ratio in RATIOS:
        for split, idx in splits.items():
            # the same mask draw for both splits at a given ratio
            cell = measure(model, heads, conv, data, idx, ratio, args.batch, device,
                           seed=int(ratio * 1000))
            res["cells"][f"{split}@{ratio:.2f}"] = cell
            print(f"  {split:8s} mask {ratio:.2f}: CE all {cell['ce_all']:.4f} "
                  f"codes {cell['ce_code_levels']:.4f} top1 {cell['top1_acc_code_levels']:.3f}",
                  flush=True)
            if cell["argmin_matches_stored_code"] < 0.99:
                raise RuntimeError(f"residual reconstruction disagrees with the stored codes "
                                   f"({cell['argmin_matches_stored_code']:.4f}); the near-miss "
                                   "numbers would be wrong")

    print("\nH1  train/held-out gap in code CE (nats), by mask ratio:")
    for ratio in RATIOS:
        tr, ho = res["cells"][f"train@{ratio:.2f}"], res["cells"][f"heldout@{ratio:.2f}"]
        gap = ho["ce_all"] - tr["ce_all"]
        res["cells"][f"gap@{ratio:.2f}"] = gap
        print(f"  mask {ratio:.2f}: train {tr['ce_all']:.4f}  held-out {ho['ce_all']:.4f}  "
              f"gap {gap:+.4f}")
    h = res["cells"]["heldout@0.90"]
    print("\nH2  held-out, mask 0.90:")
    print(f"  top-1 accuracy on code levels       {f3(h['top1_acc_code_levels'])}")
    print(f"  wrong predictions within 5/10/20% of the true code's distance: "
          f"{f3(h['wrong_within_5pct'])} / {f3(h['wrong_within_10pct'])} / "
          f"{f3(h['wrong_within_20pct'])} (median ratio {f3(h['wrong_ratio_median'])})")
    print(f"  tokenizer near-ties (2nd/1st < 1.05 / 1.10): {f3(h['tokenizer_near_tie_5pct'])} / "
          f"{f3(h['tokenizer_near_tie_10pct'])}")
    print(f"  head's probability on codewords within 10% of true: {f3(h['prob_mass_within_10pct'])}")
    print(f"  median d2^2 - d1^2 per level (train): "
          f"{res['cells']['train@0.90']['median_gap_sq_per_level']}")
    print(f"  stored codes reproduced: {h['argmin_matches_stored_code']:.4f}")

    out = pathlib.Path(args.out) if args.out else run / "diag" / "overfit_diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
