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
from adarq_flow.eval import (
    ClipTowerDistortion,
    DownstreamReport,
    allocate_greedy_downstream,
    allocate_oracle,
    allocate_random,
    allocate_uniform,
    headroom_pilot,
    load_tower_context,
    marginal_returns_are_monotone,
    prefix_errors,
    prefix_zhat,
)
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
        zero_code: bool = True, json_out: str | None = None,
        eval_latents: torch.Tensor | None = None) -> None:
    """Fit the quantizer on ``latents`` and measure on ``eval_latents``.

    Real data REQUIRES a held-out ``eval_latents``: fitting and scoring on the same
    patches lets a 4 x 512-code RVQ nearly memorize a small set, which inflated an
    earlier 64-image result (ISSUES.md F17).
    """
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
    if real_data and eval_latents is None:
        raise ValueError("real-data E0 needs held-out eval_latents (ISSUES.md F17)")
    fit_set = latents
    latents = eval_latents if eval_latents is not None else latents
    print(f"  fit on {fit_set.shape[0]} patches, measured on {latents.shape[0]} "
          f"{'held-out ' if eval_latents is not None else ''}patches", flush=True)
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
    if json_out:
        import json
        pathlib.Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(json_out).write_text(json.dumps({
            "seed": seed, "num_patches": int(latents.shape[0]),
            "num_fit_patches": int(fit_set.shape[0]),
            "mean_depth_budget": rep.mean_depth_budget,
            "budget_was_substituted": rep.budget_was_substituted,
            "oracle_gain_vs_uniform": rep.oracle_gain_vs_uniform,
            "oracle_gain_vs_random": rep.oracle_gain_vs_random,
            "monotone": monotone, "verdict": rep.verdict}, indent=2))
        print(f"  wrote {json_out}", flush=True)


def run_downstream(latents_path: str, ctx_path: str, preset: str, fit_steps: int,
                   mean_depth: float, model_id: str, max_images: int, seed: int,
                   device: str = "cpu", json_out: str | None = None,
                   fit_fraction: float = 0.5) -> None:
    """E0 measured through a REAL frozen decoder (the CLIP tower suffix).

    Distortion here is non-separable across patches, so the Lagrangian oracle does not
    apply and a greedy search over the actual decoder output is used instead.
    """
    import dataclasses
    import statistics as st

    dev = torch.device(device)
    z = torch.load(latents_path, map_location=dev, weights_only=True)
    ctx = load_tower_context(ctx_path, device=device)
    n, d = ctx.patches_per_image, z.shape[-1]
    b_all = ctx.num_images
    if z.shape[0] != b_all * n:
        raise ValueError(f"{z.shape[0]} latents != {b_all} images x {n} patches")
    # Held-out split BY IMAGE: the quantizer never sees the images it is scored on.
    n_fit = int(b_all * fit_fraction)
    if not 0 < n_fit < b_all:
        raise ValueError(f"fit_fraction={fit_fraction} leaves no fit or no eval images")
    n_img = min(max_images, b_all - n_fit)
    z_fit = z[: n_fit * n]
    z_eval = z[n_fit * n:(n_fit + n_img) * n]
    cfg = dataclasses.replace(get_preset(preset).tokenizer, clip_dim=d)

    torch.manual_seed(seed)
    q = AdaptiveResidualQuantizer(cfg).to(dev)
    q.train()
    g = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(fit_steps):
        q(z_fit[torch.randint(z_fit.shape[0], (min(128, z_fit.shape[0]),), generator=g,
                              device=dev)], update_codebook=True)
    q.eval()
    print(f"  tokenizer fit on {n_fit} images ({fit_steps} steps, "
          f"~{fit_steps * 128 / z_fit.shape[0]:.1f} passes); scored on {n_img} "
          "HELD-OUT images", flush=True)

    dist = ClipTowerDistortion(model_id, ctx, device=device, metric="cosine")
    table = prefix_zhat(q, z_eval).reshape(n_img, n, cfg.max_depth + 1, d)
    errs = prefix_errors(q, z_eval).reshape(n_img, n, cfg.max_depth + 1)
    ref = z_eval.reshape(n_img, n, d)
    budget = int(round(mean_depth * n))
    idx = torch.arange(n, device=dev)

    acc: dict[str, list[float]] = {k: [] for k in
                                   ("uniform", "random", "oracle_l2", "oracle_down")}
    per_image: list[float] = []
    for i in range(n_img):
        cls = ctx.cls_tokens[n_fit + i:n_fit + i + 1]
        e_ref = dist.embed(ref[i:i + 1], cls=cls)

        def score(lat, e_ref=e_ref, cls=cls):
            return 1.0 - (dist.embed(lat, cls=cls) * e_ref).sum(-1)

        pol = {
            "uniform": allocate_uniform(n, budget, cfg.max_depth, z.device),
            "random": allocate_random(
                n, budget, cfg.max_depth, z.device,
                generator=torch.Generator(device=dev).manual_seed(seed + i)),
            "oracle_l2": allocate_oracle(errs[i], budget, cfg.max_depth),
            "oracle_down": allocate_greedy_downstream(table[i], budget, cfg.max_depth,
                                                      score),
        }
        for k, dep in pol.items():
            if int(dep.sum()) != budget:
                raise RuntimeError(f"{k} did not match the rate budget")
            acc[k].append(float(score(table[i][idx, dep].unsqueeze(0))))
        per_image.append((acc["uniform"][-1] - acc["oracle_down"][-1])
                         / max(acc["uniform"][-1], 1e-12))
        print(f"  image {i + 1}/{n_img}", flush=True)

    rep = DownstreamReport(
        num_images=n_img, mean_depth_budget=budget / n, layer=ctx.layer,
        uniform=st.mean(acc["uniform"]), random=st.mean(acc["random"]),
        oracle_latent_l2=st.mean(acc["oracle_l2"]),
        oracle_downstream=st.mean(acc["oracle_down"]),
        per_image_gain=per_image)
    print("\n" + "=" * 68)
    print(rep.summary())
    print("\n  NOTE: greedy search on a non-separable objective is not provably optimal, "
          "\n  so the reported headroom is a LOWER bound on the true optimum.")
    if json_out:
        import json
        pathlib.Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(json_out).write_text(json.dumps({
            "seed": seed, "num_images": rep.num_images, "num_fit_images": n_fit,
            "layer": rep.layer,
            "mean_depth_budget": rep.mean_depth_budget,
            "uniform": rep.uniform, "random": rep.random,
            "oracle_latent_l2": rep.oracle_latent_l2,
            "oracle_downstream": rep.oracle_downstream,
            "headroom": rep.headroom, "latent_l2_gain": rep.latent_l2_gain,
            "random_gain": rep.random_gain, "criterion_gap": rep.criterion_gap,
            "per_image_gain": rep.per_image_gain,
            "underpowered": rep.underpowered(),
        }, indent=2))
        print(f"  wrote {json_out}", flush=True)


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
    ap.add_argument("--downstream", action="store_true",
                    help="measure distortion through the REAL frozen CLIP tower suffix "
                         "instead of latent L2 (requires latents extracted with --layer)")
    ap.add_argument("--ctx", default=None,
                    help="tower-context sidecar; defaults to <latents>.ctx.pt")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--json-out", default=None,
                    help="write the result as JSON (for multi-seed aggregation)")
    ap.add_argument("--max-images", type=int, default=8,
                    help="cap on HELD-OUT images scored; the greedy downstream search "
                         "is O(budget x N) decoder passes")
    ap.add_argument("--fit-fraction", type=float, default=0.5,
                    help="share of images used ONLY to fit the quantizer; the rest are "
                         "held out and scored (ISSUES.md F17)")
    ap.add_argument("--no-zero-code", action="store_true",
                    help="disable the reserved zero codeword; RVQ then is NOT monotone "
                         "in depth and the headroom measurement is confounded by the "
                         "quantizer harming itself at deeper levels")
    args = ap.parse_args()

    dim = get_preset(args.preset).tokenizer.clip_dim
    g = torch.Generator().manual_seed(args.seed)

    if args.downstream:
        if not args.latents:
            raise SystemExit("--downstream requires --latents (extracted with --layer)")
        ctx_path = args.ctx or (args.latents + ".ctx.pt")
        if not pathlib.Path(ctx_path).exists():
            raise SystemExit(
                f"{ctx_path} not found. Re-extract with:\n"
                f"  python scripts/extract_clip_latents.py --out {args.latents} --layer 8\n"
                "Downstream distortion needs the tower context; refusing to fall back to "
                "latent L2, which measures a different objective.")
        run_downstream(args.latents, ctx_path, args.preset, args.fit_steps,
                       args.mean_depth if args.mean_depth else 2.5,
                       args.model, args.max_images, args.seed,
                       device=args.device, json_out=args.json_out,
                       fit_fraction=args.fit_fraction)
        return

    if args.latents:
        z = torch.load(args.latents, map_location="cpu", weights_only=True)
        if z.dim() == 3:
            z = z.reshape(-1, z.size(-1))
        if z.size(-1) != dim:
            raise ValueError(f"latents have dim {z.size(-1)}, preset expects {dim}")
        import json
        meta_path = pathlib.Path(args.latents + ".meta.json")
        if not meta_path.exists():
            raise SystemExit(f"{meta_path} missing: image boundaries are needed to hold "
                             "out whole images")
        meta = json.loads(meta_path.read_text())
        b, n = int(meta["num_images"]), int(meta["patches_per_image"])
        if z.shape[0] != b * n:
            raise ValueError(f"{z.shape[0]} latents != {b} images x {n} patches")
        n_fit = int(b * args.fit_fraction)
        if not 0 < n_fit < b:
            raise ValueError(f"--fit-fraction {args.fit_fraction} leaves no fit or eval set")
        print(f"  {n_fit} images fit the quantizer, {b - n_fit} held-out images are scored")
        run("real latents (held-out images)", z[: n_fit * n], args.preset,
            args.fit_steps, args.mean_depth, args.seed, real_data=True,
            zero_code=not args.no_zero_code, json_out=args.json_out,
            eval_latents=z[n_fit * n:])
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
