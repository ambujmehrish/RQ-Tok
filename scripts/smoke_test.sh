#!/usr/bin/env bash
# =====================================================================================
# AdaRQ-Flow end-to-end smoke test on REAL DATA.
#
#   bash scripts/smoke_test.sh
#
# Runs the parts of the pipeline that are scientifically meaningful today, on real
# photographs through a real CLIP vision tower:
#
#   1. quality gate            ruff + mypy + pytest
#   2. real data               download real COCO val2017 images
#   3. real CLIP latents       openai/clip-vit-base-patch32 patch hidden states
#   4. Stage-0 tokenizer       adaptive RVQ fitted on those real latents
#   5. E0 headroom pilot       the go/no-go number for the rate-allocation thesis
#   6. experiment-gate check   confirm dummy components cannot reach a real run
#
# NO SYNTHETIC FALLBACK. Every step exits non-zero on failure rather than substituting
# generated data — a smoke test that silently degrades to fake inputs is worse than no
# smoke test, because it reports success (see ISSUES.md F8/F9).
#
# OFFLINE COMPUTE NODES: prefetch once on a login node, then source the env file:
#   ADARQ_CACHE=$WORK/adarq-cache bash scripts/prefetch_assets.sh
#   source $ADARQ_CACHE/adarq_env.sh && bash scripts/smoke_test.sh
# With $ADARQ_IMAGES set, no network access is attempted at all.
#
# Requires (online only) egress to huggingface.co and images.cocodataset.org, plus:
#   pip install -e ".[dev,torch]" && pip install transformers pillow
# =====================================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

NUM_IMAGES="${NUM_IMAGES:-64}"
FIT_STEPS="${FIT_STEPS:-600}"
MEAN_DEPTH="${MEAN_DEPTH:-2.5}"
PRESET="${PRESET:-clip_b32}"
WORK="${WORK:-$REPO/.smoke}"
LATENTS="$WORK/real_clip_latents.pt"

mkdir -p "$WORK"
step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }
fail() { printf '\n\033[31mFAILED: %s\033[0m\n' "$1" >&2; exit 1; }

# -- 1. quality gate ------------------------------------------------------------------
step "1/6  quality gate (ruff + mypy + pytest)"
python3 -m ruff check .            || fail "ruff"
python3 -m mypy adarq_flow/        || fail "mypy"
python3 -m pytest -q               || fail "pytest"

# -- 2/3. real images -> real CLIP latents --------------------------------------------
step "2/6  real images  +  3/6  real CLIP patch latents"
python3 -c "import transformers, PIL" 2>/dev/null \
  || fail "transformers/pillow missing — pip install transformers pillow (no synthetic fallback)"
# Offline path: use the images prefetched by scripts/prefetch_assets.sh.
EXTRA=()
if [ -n "${ADARQ_IMAGES:-}" ]; then
  [ -d "$ADARQ_IMAGES" ] || fail "ADARQ_IMAGES=$ADARQ_IMAGES does not exist — run scripts/prefetch_assets.sh on a login node"
  EXTRA=(--image-dir "$ADARQ_IMAGES")
  echo "  offline mode: real images from $ADARQ_IMAGES"
fi
python3 scripts/extract_clip_latents.py \
    --out "$LATENTS" --num-images "$NUM_IMAGES" "${EXTRA[@]+"${EXTRA[@]}"}" \
    || fail "real CLIP latent extraction"

python3 - "$LATENTS" <<'PY' || fail "latent provenance check"
import json, pathlib, sys, torch
p = pathlib.Path(sys.argv[1])
meta = json.loads(p.with_suffix(p.suffix + ".meta.json").read_text())
if meta.get("synthetic") is not False:
    raise SystemExit("latents are not marked real — refusing to continue")
z = torch.load(p, map_location="cpu", weights_only=True)
if z.dim() != 2 or z.shape[0] < 256:
    raise SystemExit(f"expected >=256 patch latents [M, d], got {tuple(z.shape)}")
print(f"  provenance OK: {meta['num_images']} real images from {meta['source']}")
print(f"  {tuple(z.shape)} latents via {meta['model']}")
PY

# -- 4. Stage-0 tokenizer on real latents ---------------------------------------------
step "4/6  Stage-0 adaptive RVQ tokenizer fitted on REAL latents"
python3 - "$LATENTS" "$PRESET" "$FIT_STEPS" <<'PY' || fail "Stage-0 tokenizer fit"
import sys, torch
from adarq_flow.config import get_preset
from adarq_flow.tokenizer import AdaptiveResidualQuantizer
from adarq_flow.eval import prefix_errors

path, preset, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
z = torch.load(path, map_location="cpu", weights_only=True)
cfg = get_preset(preset)
if z.shape[-1] != cfg.tokenizer.clip_dim:
    raise SystemExit(f"latent dim {z.shape[-1]} != preset {preset} clip_dim "
                     f"{cfg.tokenizer.clip_dim}")

torch.manual_seed(0)
q = AdaptiveResidualQuantizer(cfg.tokenizer)      # no encoder, no stand-in
g = torch.Generator().manual_seed(0)
q.train()
first = None
for i in range(steps):
    idx = torch.randint(z.shape[0], (min(128, z.shape[0]),), generator=g)
    out = q(z[idx], update_codebook=True)
    if first is None:
        first = out.losses.item()["recon"]
q.eval()
final = q(z, update_codebook=False)

e = prefix_errors(q, z).mean(0)
print(f"  recon (first batch -> full set): {first:.2f} -> "
      f"{final.losses.item()['recon']:.2f}")
print(f"  mean adaptive depth: {final.depths.float().mean():.2f} / D_max="
      f"{cfg.tokenizer.max_depth}")
print("  error by depth: " + "  ".join(f"d{k}={e[k]:.1f}" for k in range(1, len(e))))
gain = 100.0 * (e[1] - e[-1]) / e[1]
print(f"  depth 1 -> {len(e)-1} reduces error by {gain:.1f}%")
if gain <= 0:
    raise SystemExit("extra codes do not reduce error — check include_zero_code / "
                     "shared_codebook (ISSUES.md D1/D2)")
# codebook usage measured on the EVAL data, not the saturated EMA statistic (B2)
used = torch.unique(final.codes[final.codes != q.codebook_size]).numel()
print(f"  distinct codes actually used on real data: {used} / {q.codebook_size}")
PY

# -- 5. E0 headroom pilot on real latents ---------------------------------------------
step "5/6  E0 headroom pilot on REAL latents (go / no-go)"
python3 scripts/e0_headroom_pilot.py \
    --latents "$LATENTS" --preset "$PRESET" \
    --fit-steps "$FIT_STEPS" --mean-depth "$MEAN_DEPTH" || fail "E0 pilot"

# -- 6. the experiment gate still holds ------------------------------------------------
step "6/6  experiment gate: dummy components must be unreachable"
python3 - <<'PY' || fail "experiment gate"
import dataclasses, itertools
from adarq_flow.config import AdaRQFlowConfig, get_preset

base = get_preset("tiny_cpu")
leaked = 0
for mb, rb, rv, ds in itertools.product(
        ["dummy", "Qwen/Qwen2.5-VL-7B-Instruct"], ["dummy", "black-forest-labs/FLUX.1-dev"],
        ["dummy", "black-forest-labs/FLUX.1-dev"], ["dummy", "blip3o"]):
    cfg = dataclasses.replace(
        base,
        mllm=dataclasses.replace(base.mllm, backbone=mb),
        renderer=dataclasses.replace(base.renderer, backbone=rb, vae=rv),
        train=dataclasses.replace(base.train, dataset=ds, run_mode="experiment"))
    try:
        cfg.validate_run_mode()
        if cfg.dummy_components():
            leaked += 1
    except ValueError:
        pass
if leaked:
    raise SystemExit(f"{leaked} configs leaked a dummy into experiment mode")
print("  0/16 component combinations leak a dummy into run_mode='experiment'")

exp = AdaRQFlowConfig.from_yaml("configs/experiment.yaml")
assert exp.dummy_components() == {}, exp.dummy_components()
print("  configs/experiment.yaml contains no development stand-ins")
PY

printf '\n\033[32mSMOKE TEST PASSED\033[0m — real COCO images -> real CLIP latents -> tokenizer -> E0.\n'
printf 'Artifacts in %s\n' "$WORK"
printf 'Reminder: Stage A/B remain blocked (ISSUES.md B1-B6); this exercises the\n'
printf 'tokenizer + allocation path, which is the part real data can validate today.\n'
