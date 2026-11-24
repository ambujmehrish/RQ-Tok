#!/usr/bin/env python3
"""Inception Score decomposition with the unmodified ADM evaluator's own Inception network.

    python -m levelb.is_decomp --samples <samples_10000.npz> [--expect <...adm.metrics.json>]
    python -m levelb.is_decomp --samples $ADM_DIR/VIRTUAL_imagenet256_labeled.npz --no-labels

Per split of 5,000 images (the evaluator's split), ln IS = mean_i KL(p_i || p_bar)
= H(p_bar) - mean_i H(p_i): class spread minus per-image uncertainty. Both terms are
reported, and IS is recomputed from the same softmax outputs and checked against the
evaluator's ``compute_inception_score`` (and, with --expect, against the score already on
record). Per requested class (sample label): mean KL to the split marginal (that class's
share of ln IS), mean per-image entropy, entropy of the class's mean prediction, and how
often its images share the class's most common Inception argmax. None of these needs a
mapping from Inception's 1008 outputs to ImageNet label order.

Runs in the ADM evaluator environment (tensorflow-cpu); numpy-only apart from ``main``.
Writes <samples stem>.isdecomp.metrics.json next to the samples.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

SPLIT = 5000  # guided-diffusion evaluator.compute_inception_score(split_size=5000)


def _plogp(p: np.ndarray) -> np.ndarray:
    """p * log p with the evaluator's arithmetic (no epsilon); raises on a non-finite term."""
    out = p * np.log(p)
    if not np.isfinite(out).all():
        raise ValueError("non-finite p log p: a softmax output underflowed to 0")
    return out


def decompose(preds: np.ndarray, labels: np.ndarray | None, split: int = SPLIT) -> dict:
    """IS and its two terms from softmax outputs [N, C]; per-class terms when labels given."""
    preds = np.asarray(preds)
    if preds.ndim != 2 or len(preds) % split:
        raise ValueError(f"preds must be [N, C] with N a multiple of {split}, got {preds.shape}")
    n = len(preds)
    kl_img = np.empty(n)  # KL(p_i || split marginal)
    h_img = -_plogp(preds).sum(1)  # H(p_i)
    splits = []
    for s in range(0, n, split):
        part = preds[s : s + split]
        marg = part.mean(0, keepdims=True)
        # the evaluator's expression, term for term
        kl = part * (np.log(part) - np.log(marg))
        kl_img[s : s + split] = kl.sum(1)
        mean_kl = float(np.mean(np.sum(kl, 1)))
        h_marg = float(-_plogp(marg[0]).sum())
        h_cond = float(h_img[s : s + split].mean())
        if abs(mean_kl - (h_marg - h_cond)) > 1e-3:
            raise ValueError(
                f"identity ln IS = H(marg) - H(cond) off by {mean_kl - h_marg + h_cond}"
            )
        splits.append(
            {"IS": float(np.exp(mean_kl)), "ln_IS": mean_kl, "H_marg": h_marg, "H_cond": h_cond}
        )
    pmax = preds.max(1)
    res = {
        "n": n,
        "classes": preds.shape[1],
        "IS": float(np.mean([s["IS"] for s in splits])),
        "ln_IS": float(np.mean([s["ln_IS"] for s in splits])),
        "H_marg": float(np.mean([s["H_marg"] for s in splits])),
        "H_cond": float(np.mean([s["H_cond"] for s in splits])),
        "eff_classes": float(np.mean([np.exp(s["H_marg"]) for s in splits])),
        "max_prob_mean": float(pmax.mean()),
        "frac_max_prob_gt_0.5": float((pmax > 0.5).mean()),
        "frac_max_prob_gt_0.9": float((pmax > 0.9).mean()),
        "splits": splits,
    }
    if labels is None:
        return res
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (n,):
        raise ValueError(f"labels must be [{n}], got {labels.shape}")
    arg = preds.argmax(1)
    per = {k: [] for k in ("label", "n", "kl", "H_cond", "H_class_marg", "mode_agree")}
    for c in np.unique(labels):
        m = labels == c
        cm = preds[m].mean(0)
        counts = np.bincount(arg[m], minlength=preds.shape[1])
        per["label"].append(int(c))
        per["n"].append(int(m.sum()))
        per["kl"].append(float(kl_img[m].mean()))
        per["H_cond"].append(float(h_img[m].mean()))
        per["H_class_marg"].append(float(-_plogp(cm).sum()))
        per["mode_agree"].append(float(counts.max() / m.sum()))
    res["per_class"] = per
    res["mode_agree_mean"] = float(np.average(per["mode_agree"], weights=per["n"]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", required=True, help="npz with arr_0 uint8 NHWC (+ labels)")
    ap.add_argument("--no-labels", action="store_true", help="no per-class terms (reference batch)")
    ap.add_argument("--expect", default=None, help="metrics JSON whose IS this must reproduce")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    adm = os.environ.get("ADM_DIR")
    if not adm:
        raise SystemExit("ADM_DIR not set: source $BENCH_ROOT/adm_env.sh")
    sys.path.insert(0, str(pathlib.Path(adm) / "guided-diffusion" / "evaluations"))
    import tensorflow.compat.v1 as tf
    from evaluator import Evaluator  # unmodified, at the pinned commit

    path = pathlib.Path(args.samples).resolve()  # main() changes directory below
    expect = pathlib.Path(args.expect).resolve() if args.expect else None
    out = (
        pathlib.Path(args.out).resolve()
        if args.out
        else path.with_name(path.stem + ".isdecomp.metrics.json")
    )
    labels = None
    if not args.no_labels:
        with np.load(path) as z:
            if "labels" not in z.files:
                raise SystemExit(f"{path} has no 'labels'; pass --no-labels for a reference batch")
            labels = np.asarray(z["labels"])

    os.chdir(adm)  # the evaluator reads classify_image_graph_def.pb from the working directory
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    ev = Evaluator(tf.Session(config=config))
    ev.warmup()
    acts = ev.read_activations(str(path))[0]
    preds = np.concatenate(
        [
            ev.sess.run(
                ev.softmax, feed_dict={ev.softmax_input: acts[i : i + ev.softmax_batch_size]}
            )
            for i in range(0, len(acts), ev.softmax_batch_size)
        ]
    )
    res = decompose(preds, labels)
    is_eval = ev.compute_inception_score(acts)
    if abs(res["IS"] - is_eval) > 1e-6 * is_eval:
        raise SystemExit(f"recomputed IS {res['IS']} != evaluator IS {is_eval}")
    res["IS_evaluator"] = is_eval
    if expect:
        on_record = json.loads(expect.read_text())["IS"]
        if abs(is_eval - on_record) > 1e-3 * on_record:
            raise SystemExit(f"IS {is_eval} does not reproduce the recorded {on_record} ({expect})")
        res["IS_on_record"] = on_record
    res["source"] = str(path)
    out.write_text(json.dumps(res, indent=1))
    print(
        f"IS {res['IS']:.3f} = exp({res['H_marg']:.4f} - {res['H_cond']:.4f})  "
        f"eff. classes {res['eff_classes']:.1f}  max-prob {res['max_prob_mean']:.3f}  -> {out}"
    )


if __name__ == "__main__":
    main()
