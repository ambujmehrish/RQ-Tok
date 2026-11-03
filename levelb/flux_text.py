#!/usr/bin/env python3
"""FLUX text conditioning for the 1000 ImageNet class prompts, computed once.

    python -m levelb.flux_text --data $LEVELB_ROOT/imagenet256

FLUX's own ``encode_prompt`` (CLIP pooled + T5 at max_sequence_length 512, the defaults of
diffusers' ``train_controlnet_flux.py``). With class-name prompts there are only 1000
distinct inputs and the text encoders are frozen, so encoding each once is exact and
removes T5-XXL from the training loop. Writes ``flux_t5.npy`` (uint16 bf16 bits
[1000, 512, D]) and ``flux_pooled.npy`` (same, [1000, P]) next to the latents.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

MAX_SEQ = 512


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def from_bits(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16)


@torch.no_grad()
def encode(flux_id: str, prompts: list[str], device: str = "cuda", batch: int = 32):
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(flux_id, transformer=None, vae=None,
                                        torch_dtype=torch.bfloat16).to(device)
    t5, pooled = [], []
    for s in range(0, len(prompts), batch):
        part = prompts[s:s + batch]
        pe, pp, _ = pipe.encode_prompt(prompt=part, prompt_2=part, device=torch.device(device),
                                       max_sequence_length=MAX_SEQ)
        t5.append(pe.cpu())
        pooled.append(pp.cpu())
    return torch.cat(t5), torch.cat(pooled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="dir holding class_names.json")
    ap.add_argument("--flux", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    root = pathlib.Path(args.data)
    names = json.loads((root / "class_names.json").read_text())
    t5, pooled = encode(args.flux, names, args.device)
    if t5.shape[:2] != (len(names), MAX_SEQ) or pooled.shape[0] != len(names):
        raise SystemExit(f"unexpected shapes {tuple(t5.shape)} / {tuple(pooled.shape)}")
    np.save(root / "flux_t5.npy", bf16_bits(t5))
    np.save(root / "flux_pooled.npy", bf16_bits(pooled))
    (root / "flux_text_meta.json").write_text(json.dumps(
        {"flux": args.flux, "max_sequence_length": MAX_SEQ, "prompts": len(names),
         "t5_shape": list(t5.shape), "pooled_shape": list(pooled.shape)}, indent=2))
    print(f"encoded {len(names)} class prompts: T5 {tuple(t5.shape)}, pooled {tuple(pooled.shape)}")


if __name__ == "__main__":
    main()
