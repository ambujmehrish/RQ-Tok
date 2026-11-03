"""bifrost/models/mar/utils.py — MISSING from the official Bifrost-1 repository.

Upstream commit 3047de271b54dbd37228b038cc1f949899b0f1a8 imports
``from bifrost.models.mar.utils import sample_orders, random_masking`` in
``inference_geneval_dpgbench.py`` but does not ship the module, so the official script
cannot start. ``scripts/bifrost1/setup.sbatch`` copies this file into the checkout.

* ``sample_orders`` is the only one the inference script calls. It is reproduced from MAR
  (Li et al., 2024, github.com/LTH14/mar, ``MAR.sample_orders``), which Bifrost-1 cites for
  its decoding: one uniformly random permutation of the token positions per sample, drawn
  from NumPy's global RNG exactly as MAR does. The upstream script never seeds NumPy, so
  decoding orders differ between runs; that behaviour is kept, not "fixed".
* ``random_masking`` is imported but never called during inference. Its upstream signature
  is unknown, so it is NOT guessed: calling it raises.

Recorded as deviation A1 in EXPERIMENTS.md (Bifrost-1 reference reproduction).
"""

import numpy as np
import torch


def sample_orders(bsz, seq_len):
    """``[bsz, seq_len]`` LongTensor; each row is a random permutation (as in MAR)."""
    orders = []
    for _ in range(bsz):
        order = np.array(list(range(seq_len)))
        np.random.shuffle(order)
        orders.append(order)
    return torch.Tensor(np.array(orders)).long()


def random_masking(*args, **kwargs):
    raise NotImplementedError(
        "random_masking is not shipped by upstream Bifrost-1 and is not used by its "
        "inference script; its signature is unknown, so it is deliberately not guessed."
    )
