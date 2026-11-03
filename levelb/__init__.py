"""Level B: the controlled ImageNet-256 head-to-head (EXPERIMENTS.md, "Level B").

Runs in the ``bifrost1`` environment (upstream's pinned torch / transformers / diffusers)
and imports upstream's own modules from ``$BIFROST_CODE`` (HL-hanlin/Bifrost-1@3047de2),
so the Bifrost-1 baseline trains the authors' released model code. Only what upstream did
not release is written here: the ImageNet data pipeline and the training/sampling loops.
"""
