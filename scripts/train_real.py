#!/usr/bin/env python3
"""Real end-to-end run: real Qwen2.5-VL + real FLUX + real COCO image-caption pairs.

This is the entrypoint that composes every real component. It exists because the
tokenizer and the image transform must come from the MLLM itself — a generic trainer
cannot guess them, and guessing would silently change the data (`ISSUES.md` B4/F8).

    python scripts/train_real.py --config configs/experiment.yaml --stage branch \\
        --image-dir $ADARQ_IMAGES --annotations $ADARQ_CAPTIONS

Offline compute nodes: source $ADARQ_CACHE/adarq_env.sh first (see
scripts/prefetch_assets.sh); both paths are then already in the environment.

No synthetic fallback anywhere: a missing model, image set or caption file exits
non-zero rather than substituting a stand-in.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from adarq_flow.config import AdaRQFlowConfig
from adarq_flow.data import CocoCaptionsDataset, build_dataloader
from adarq_flow.training import Trainer


def build_mllm_io(model_id: str, max_text_len: int):
    """Tokenizer + image transform taken from the MLLM's own processor.

    Both must match the backbone exactly: the text tokenizer because the frozen stack
    only understands its own ids, and the image preprocessing because the visual tower
    was trained on it.
    """
    try:
        from transformers import AutoTokenizer, Qwen2VLImageProcessorPil
    except ImportError as e:
        raise SystemExit(
            'transformers is required (pip install -e ".[models]"). Refusing to '
            "substitute a development stand-in."
        ) from e

    tok = AutoTokenizer.from_pretrained(model_id)
    proc = Qwen2VLImageProcessorPil.from_pretrained(model_id)

    def tokenize(text: str) -> torch.Tensor:
        ids = tok(text, truncation=True, max_length=max_text_len,
                  return_tensors="pt")["input_ids"][0]
        if ids.numel() == 0:
            raise ValueError(f"caption tokenized to nothing: {text!r}")
        return ids

    def image_transform(img):
        out = proc(images=[img], return_tensors="pt")
        # Qwen packs patches flat and reports the grid; the encoder needs both.
        return out["pixel_values"][0]

    return tokenize, image_transform


def main() -> None:
    ap = argparse.ArgumentParser(description="AdaRQ-Flow real end-to-end run")
    ap.add_argument("--config", required=True, help="e.g. configs/experiment.yaml")
    ap.add_argument("--stage", required=True,
                    choices=["tokenizer", "branch", "renderer"])
    ap.add_argument("--image-dir", default=None,
                    help="real images (defaults to $ADARQ_IMAGES)")
    ap.add_argument("--annotations", default=None,
                    help="captions_val2017.json (defaults to $ADARQ_CAPTIONS)")
    ap.add_argument("--max-text-len", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    import os

    image_dir = args.image_dir or os.environ.get("ADARQ_IMAGES")
    annotations = args.annotations or os.environ.get("ADARQ_CAPTIONS")
    if not image_dir or not annotations:
        raise SystemExit(
            "need --image-dir and --annotations (or $ADARQ_IMAGES / $ADARQ_CAPTIONS from "
            "scripts/prefetch_assets.sh). Refusing to fall back to synthetic data, which "
            "has zero image-text mutual information (ISSUES.md B4)."
        )

    import dataclasses

    cfg = AdaRQFlowConfig.from_yaml(args.config)
    if args.max_steps is not None:
        cfg = dataclasses.replace(
            cfg, train=dataclasses.replace(cfg.train, max_steps=args.max_steps))
    cfg = dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, stage=args.stage, dataset="coco"))
    cfg.validate_run_mode()          # refuses any remaining stand-in

    print(f"[real] config={args.config} stage={args.stage}", flush=True)
    print(f"[real] mllm={cfg.mllm.backbone}", flush=True)
    print(f"[real] renderer={cfg.renderer.backbone} vae={cfg.renderer.vae}", flush=True)

    tokenize, image_transform = build_mllm_io(cfg.mllm.backbone, args.max_text_len)
    dataset = CocoCaptionsDataset(image_dir, annotations,
                                  tokenize=tokenize, image_transform=image_transform)
    diversity = CocoCaptionsDataset.mutual_information_sanity_check(dataset)
    print(f"[real] {len(dataset)} real (image, caption) pairs | "
          f"caption diversity {diversity:.0%}", flush=True)
    if diversity < 0.5:
        raise SystemExit(
            f"caption diversity {diversity:.0%} is too low: the text carries almost no "
            "image-specific signal, so the branch stage would be vacuous (B4)."
        )

    trainer = Trainer(cfg, setup_dist=True)
    trainer.loader = build_dataloader(cfg, dataset, sampler=trainer.sampler)
    try:
        report = trainer.train()
        path = trainer.save_checkpoint(report.steps)
        print(f"[real] {report.stage} done: final_loss={report.final_loss:.4f} "
              f"checkpoint={path}", flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
