"""Former demo entrypoint — now refuses to run.

It generated from the all-dummy ``tiny_cpu`` preset with RANDOM token ids as the "prompt",
so its output was meaningless. Generation for experiments is done by the Level B sampler
(real Qwen2.5-VL + real FLUX.1-dev, real prompts). Unit tests drive the pipeline directly.
"""

from __future__ import annotations


def main(argv=None) -> None:
    raise SystemExit(
        "python -m adarq_flow.inference.generate is disabled: it only ran the dummy preset "
        "on random token ids. Use the Level B sampler (levelb/) with real models and prompts."
    )


if __name__ == "__main__":
    main()
