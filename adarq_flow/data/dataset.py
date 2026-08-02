"""Datasets / loaders.

``DummyImageTextDataset`` is a deterministic synthetic (image, text) source for CPU
development and tests. Real datasets (ImageNet, BLIP3-o) raise ``NotImplementedError``
until wired — no silent fallback to dummy data.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..config import AdaRQFlowConfig


class DummyImageTextDataset(Dataset):
    """Synthetic, deterministic (image ``[3,H,W]``, text ``[T]``) pairs."""

    def __init__(self, length: int = 256, image_size: int = 8, channels: int = 3,
                 text_len: int = 8, vocab: int = 256, seed: int = 0) -> None:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        self.length = length
        self.image_size = image_size
        self.channels = channels
        self.text_len = text_len
        self.vocab = vocab
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        img = torch.randn(self.channels, self.image_size, self.image_size, generator=g)
        text = torch.randint(0, self.vocab, (self.text_len,), generator=g)
        return img, text


def build_dataset(cfg: AdaRQFlowConfig, length: int = 256) -> Dataset:
    """Construct the dataset named by ``cfg.train.dataset``."""
    name = cfg.train.dataset
    if name == "dummy":
        return DummyImageTextDataset(
            length=length,
            image_size=max(cfg.renderer.image_size // 8, 8) if cfg.renderer.image_size else 8,
        )
    raise NotImplementedError(
        f"dataset '{name}' is not yet wired. Implement an ImageNet/BLIP3-o loader that "
        "yields (image, text_ids); for CPU runs set train.dataset='dummy'."
    )


def build_dataloader(cfg: AdaRQFlowConfig, dataset: Dataset, sampler=None) -> DataLoader:
    """A DataLoader; pass a ``DistributedSampler`` for multi-GPU (shuffle off then)."""
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=0,
    )
