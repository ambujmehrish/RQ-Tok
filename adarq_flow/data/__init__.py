"""Data loaders: dummy (CPU dev/tests), ImageNet, BLIP3-o. Requires torch."""

from .dataset import (
    DummyImageTextDataset,
    build_dataloader,
    build_dataset,
)

__all__ = ["DummyImageTextDataset", "build_dataset", "build_dataloader"]
