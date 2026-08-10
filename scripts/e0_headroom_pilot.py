#!/usr/bin/env python3
"""E0 — headroom pilot (EXPERIMENTS.md §E0). Go / no-go for the rate-allocation thesis.

Measures how much distortion a *perfect* allocator could save over a uniform one at the
SAME mean rate, and whether that saving requires content-awareness (oracle vs random).

    python scripts/e0_headroom_pilot.py --regime both

Regimes
-------
`homogeneous`   i.i.d. latents — every patch is statistically identical. There is no
                content structure to exploit, so a correct harness MUST report ~zero
                headroom here. This is the harness's own negative control: if it shows
                a gain on homogeneous data, the measurement is broken, not the thesis.
`heterogeneous` patch complexity varies (a mixture of tight and diffuse clusters), which
                is the property real images are assumed to have.

The number that decides the project must come from **real CLIP latents** (`--latents`);
synthetic regimes only validate the instrument and bound what to expect.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from adarq_flow.config import get_preset
from adarq_flow.eval import headroom_pilot, marginal_returns_are_monotone, prefix_errors
from adarq_flow.tokenizer import (
    AdaptiveResidualQuantizer,
    build_tokenizer,
    fit_tokenizer,
)


def synth_homogeneous(m: int, d: int, g: torch.Generator) -> torch.Tensor:
    """Every patch drawn from one isotropic Gaussian: no per-patch complexity structure."""
    return torch.randn(m, d, generator=g)


def synth_heterogeneous(m: int, d: int, g: torch.Generator,
                        n_clusters: int = 16) -> torch.Tensor:
    """Patches differ in how well a codeword can describe them.

    Half the clusters are tight (a code captures them almost exactly -> extra depth is
    wasted), half are diffuse (extra depth keeps paying). That heterogeneity is exactly
    what a rate allocator is supposed to exploit.
    """
    centers = torch.randn(n_clusters, d, generator=g) * 3.0
    spread = torch.where(torch.arange(n_clusters) < n_clusters // 2,
                         torch.full((n_clusters,), 0.02),
                         torch.full((n_clusters,), 1.5))
    assign = torch.randint(n_clusters, (m,), generator=g)
    return centers[assign] + spread[assign].unsqueeze(1) * torch.randn(m, d, generator=g)


def run(name: str, latents: torch.Tensor, preset: str, fit_steps: int,
        mean_depth: float | None, seed: int, real_data: bool = False,
        zero_code: bool = True) -> None:
    import dataclasses
    cfg = get_preset(preset)
    cfg = dataclasses.replace(cfg, tokenizer=dataclasses.replace(
        cfg.tokenizer, include_zero_code=zero_code))
    torch.manual_seed(seed)
    if real_data:
        # Build ONLY the quantizer: real latents are supplied directly, so no image
        # encoder is needed and no development stand-in is constructed at all.
        q = AdaptiveResidualQuantizer(cfg.tokenizer)
        gen = torch.Generator(device=latents.device).manual_seed(seed)
        q.train()
        for _ in range(fit_steps):
            idx = torch.randint(latents.shape[0], (min(128, latents.shape[0]),),
                                generator=gen, device=latents.device)
            q(latents[idx], update_codebook=True)
        q.eval()
    else:
        tok = build_tokenizer(cfg)
        fit_tokenizer(tok, latents, steps=fit_steps, batch_size=128, seed=seed)
        q = tok.quantizer
    errs = prefix_errors(q, latents)
    monotone = marginal_returns_are_monotone(errs)

    rep = headroom_pilot(q, latents, mean_depth=mean_depth, seed=seed,
                         real_data=real_data)
    print(f"\n{'=' * 68}\nREGIME: {name}   (D_max={q.max_depth}, K={q.codebook_size}, "
          f"M={latents.shape[0]})\n{'=' * 68}")
    print(rep.summary())
    if not monotone:
        print("\n  NOTE: per-patch marginal returns are NOT non-increasing in depth, so "
              "\n  the RD curves are non-convex. The Lagrangian allocator solves the "
              "\n  CONVEXIFIED problem, so 'oracle' is a tight lower bound on the true "
              "\n  integer optimum, not necessarily the optimum itself.")


def main() -> None:
    ap = argparse.ArgumentParser(description="E0 headroom pilot")
    ap.add_argument("--regime", default="both",
                    choices=["homogeneous", "heterogeneous", "both"])
    ap.add_argument("--latents", default=None,
                    help="path to a .pt tensor of REAL CLIP patch latents [M, d] "
                         "(the only input whose verdict counts)")
    ap.add_argument("--preset", default="tiny_cpu")
    ap.add_argument("--patches", type=int, default=4096)
    ap.add_argument("--fit-steps", type=int, default=400)
    ap.add_argument("--mean-depth", type=float, default=None,
                    help="rate all policies are held to (default: the threshold rule's "
                         "own realized mean depth)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-zero-code", action="store_true",
                    help="disable the reserved zero codeword; RVQ then is NOT monotone "
                         "in depth and the headroom measurement is confounded by the "
                         "quantizer harming itself at deeper levels")
    args = ap.parse_args()

    dim = get_preset(args.preset).tokenizer.clip_dim
    g = torch.Generator().manual_seed(args.seed)

    if args.latents:
        z = torch.load(args.latents, map_location="cpu", weights_only=True)
        if z.dim() == 3:
            z = z.reshape(-1, z.size(-1))
        if z.size(-1) != dim:
            raise ValueError(f"latents have dim {z.size(-1)}, preset expects {dim}")
        run("real latents", z, args.preset, args.fit_steps, args.mean_depth,
            args.seed, real_data=True,
            zero_code=not args.no_zero_code)
        return

    if args.regime in ("homogeneous", "both"):
        run("homogeneous (harness control — expect ~0 headroom)",
            synth_homogeneous(args.patches, dim, g), args.preset, args.fit_steps,
            args.mean_depth, args.seed, zero_code=not args.no_zero_code)
    if args.regime in ("heterogeneous", "both"):
        run("heterogeneous (structured — headroom should appear)",
            synth_heterogeneous(args.patches, dim, g), args.preset, args.fit_steps,
            args.mean_depth, args.seed, zero_code=not args.no_zero_code)

    print("\nNOTE: synthetic regimes validate the instrument only. The go/no-go verdict "
          "\nrequires real CLIP patch latents via --latents.")


if __name__ == "__main__":
    main()
