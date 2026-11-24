"""levelb.is_decomp: the IS identity and the per-class terms (the network is the evaluator's)."""

from __future__ import annotations

import numpy as np
import pytest

from levelb.class_accuracy import per_class_summary
from levelb.is_decomp import decompose


def _softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _evaluator_is(preds, split):
    """guided-diffusion evaluator.compute_inception_score, verbatim arithmetic."""
    scores = []
    for i in range(0, len(preds), split):
        part = preds[i : i + split]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        scores.append(np.exp(np.mean(np.sum(kl, 1))))
    return float(np.mean(scores))


def test_matches_evaluator_and_identity():
    rng = np.random.default_rng(0)
    preds = _softmax(3 * rng.standard_normal((40, 12)))
    labels = rng.integers(0, 5, 40)
    r = decompose(preds, labels, split=20)
    assert r["IS"] == pytest.approx(_evaluator_is(preds, 20), rel=1e-12)
    for s in r["splits"]:
        assert s["ln_IS"] == pytest.approx(s["H_marg"] - s["H_cond"], abs=1e-9)
    assert sum(r["per_class"]["n"]) == 40
    # per-class KL, weighted by class size, averages back to ln IS (equal-size splits)
    w = np.array(r["per_class"]["n"]) / 40
    assert float(w @ np.array(r["per_class"]["kl"])) == pytest.approx(r["ln_IS"], rel=1e-9)


def test_confident_distinct_images_reach_the_class_count():
    preds = np.full((10, 10), 1e-6)
    np.fill_diagonal(preds, 1.0)
    preds /= preds.sum(1, keepdims=True)
    r = decompose(preds, np.arange(10), split=10)
    assert r["IS"] == pytest.approx(10, rel=1e-3)
    assert r["mode_agree_mean"] == 1.0


def test_rejects_partial_split_and_zero_probabilities():
    with pytest.raises(ValueError):
        decompose(np.full((15, 3), 1 / 3), None, split=10)
    with pytest.raises(ValueError):
        decompose(np.array([[1.0, 0.0], [0.0, 1.0]]), None, split=2)


def test_per_class_summary():
    labels = np.array([3, 1, 3, 3, 1])
    top1 = np.array([True, False, False, True, True])
    s = per_class_summary(labels, top1, np.array([0.9, 0.1, 0.2, 0.7, 0.5]), np.ones(5))
    pc = s["per_class"]
    assert pc["label"] == [1, 3] and pc["n"] == [2, 3] and pc["hits"] == [1, 2]
    assert pc["p_req"] == pytest.approx([0.3, 0.6])
