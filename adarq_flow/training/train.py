"""Unified training entrypoint for all decoupled stages.

    python -m adarq_flow.training.train --preset tiny_cpu --stage tokenizer
    srun python -m adarq_flow.training.train --preset base_gpu --stage branch

Launch under SLURM/torchrun for multi-GPU (see scripts/cineca_leonardo_4xA100.sbatch).
"""

from __future__ import annotations

import argparse
import dataclasses

from ..config import AdaRQFlowConfig, get_preset
from .trainer import Trainer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="AdaRQ-Flow trainer")
    ap.add_argument("--preset", default="tiny_cpu")
    ap.add_argument("--config", default=None,
                    help="path to a YAML config (e.g. configs/ablations/*.yaml). "
                         "Takes precedence over --preset.")
    ap.add_argument("--stage", required=True, choices=["tokenizer", "branch", "renderer"])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--dataset-length", type=int, default=256)
    args = ap.parse_args(argv)

    # Without this, generated ablation YAMLs cannot be run at all.
    cfg = (AdaRQFlowConfig.from_yaml(args.config) if args.config
           else get_preset(args.preset))
    overrides = {"stage": args.stage}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **overrides))

    trainer = Trainer(cfg, dataset_length=args.dataset_length)
    try:
        report = trainer.train()
        path = trainer.save_checkpoint(report.steps)
        if path:
            print(f"[{report.stage}] done: final_loss={report.final_loss:.4f} "
                  f"checkpoint={path}", flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
