"""Former development entrypoint — now refuses to run.

It could only train development configs on the dummy dataset (random images and token
ids with zero image-text mutual information), so any number it printed was meaningless.
Real training runs only through ``scripts/train_real.py`` (real images + real captions,
``run_mode: experiment``, every component real). Unit tests drive ``Trainer`` directly.
"""

from __future__ import annotations


def main(argv=None) -> None:
    raise SystemExit(
        "python -m adarq_flow.training.train is disabled: it only ever ran on the dummy "
        "dataset. Use scripts/train_real.py --config configs/experiment.yaml (real data, "
        "run_mode='experiment')."
    )


if __name__ == "__main__":
    main()
