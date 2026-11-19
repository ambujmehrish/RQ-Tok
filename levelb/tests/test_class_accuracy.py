"""levelb.class_accuracy: top-k counting (the classifier itself is torchvision's)."""

from __future__ import annotations

import torch

from levelb.class_accuracy import topk_hits


def test_topk_hits_counts_label_in_top_k():
    logits = torch.tensor(
        [
            [0.0, 5.0, 1.0, 2.0, 3.0, 4.0, -1.0],  # label 1: top-1
            [9.0, 0.0, 8.0, 7.0, 6.0, 5.0, 4.0],  # label 5: 5th
            [9.0, 0.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        ]
    )  # label 6: 6th (miss)
    labels = torch.tensor([1, 5, 6])
    assert topk_hits(logits, labels) == {1: 1, 5: 2}
