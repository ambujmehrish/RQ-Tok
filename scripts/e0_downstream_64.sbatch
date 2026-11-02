#!/bin/bash
# =====================================================================================
# E0 headroom on vhpc01 — 64 DISTINCT real COCO images x 3 seeds, on ONE H200.
#
#   sbatch scripts/e0_downstream_64.sbatch
#   sbatch --qos=gpulongjobs scripts/e0_downstream_64.sbatch   # if 'normal' is full
#   E0_IMAGES=$HOME/work/adarq-cache/coco_val2017_512 NUM_IMAGES=512 MEAN_DEPTH=1.5 \
#       sbatch --time=08:00:00 scripts/e0_downstream_64.sbatch  # bigger set / other rate
#
# vhpc01 has its H200s one per node (vhpc13 / vhpc14 / vhpc15, gpu:h200:1 each), so a
# single job can never hold more than one. This job asks for exactly one and runs the
# seeds SEQUENTIALLY on it: it schedules as soon as any one H200 is free, instead of
# waiting for all three.
#
# Two measurements, both on the same 64 distinct images:
#   A. latent-L2 E0 (final-layer latents) — re-measures the earlier "+34% PROCEED",
#      which was in fact computed on only 20 distinct images (ISSUES.md F15).
#   B. downstream E0 (layer-L latents through the real CLIP tower suffix) — the
#      objective that matters. The CPU pilots were UNDERPOWERED (6-8 images, headroom
#      swung +15.7% .. +45.4%); EXPERIMENTS.md R2 requires >= 3 seeds with mean +- std,
#      and the report refuses a verdict below 32 images.
#
# Cost: the downstream objective is non-separable across patches, so a greedy search is
# used: ~budget x N ~= 73 x 49 ~= 3.6k decoder passes per image, ~230k per 64-image
# seed. Seconds-to-minutes per seed on an H200; the 4 h limit is generous.
#
# PREREQUISITE: scripts/prefetch_assets.sh has run (it writes $ADARQ_CACHE/adarq_env.sh)
# and the conda env 'adarq' exists. This job is fully OFFLINE and fails if either is
# missing — it never downloads, substitutes, or falls back.
# =====================================================================================
#SBATCH --job-name=adarq-e0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
# one H200 only; plain gpu:1 could land on an RTX6000/A40 node
#SBATCH --cpus-per-task=4
# 4 CPUs fits both QoS caps (normal: 120 per group, gpulongjobs: 10 per user)
#SBATCH --mem=64G
# explicit: the cluster default is 200 GB, far more than CLIP-B/32 on 64 images needs
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs

NUM_IMAGES="${NUM_IMAGES:-64}"
LAYER="${LAYER:-8}"
FIT_STEPS="${FIT_STEPS:-600}"
MEAN_DEPTH="${MEAN_DEPTH:-2.5}"
PRESET="${PRESET:-clip_b32}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
ADARQ_CACHE="${ADARQ_CACHE:-$HOME/work/adarq-cache}"

die() { echo "FAILED: $*" >&2; exit 1; }

# --- env: project conda env + offline asset cache -------------------------------------
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || die "no conda at $CONDA_SH (run scripts/vhpc01_env.sbatch first)"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "${ENV_NAME:-adarq}"
[ -f "$ADARQ_CACHE/adarq_env.sh" ] || die "$ADARQ_CACHE/adarq_env.sh missing — run \
scripts/prefetch_assets.sh first (or set ADARQ_CACHE)"
# shellcheck disable=SC1091
source "$ADARQ_CACHE/adarq_env.sh"   # HF_HOME, ADARQ_IMAGES, HF_HUB_OFFLINE=1
[ "${HF_HUB_OFFLINE:-0}" = "1" ] || die "adarq_env.sh did not set HF_HUB_OFFLINE=1"
# E0_IMAGES points a run at a larger prefetched image set without editing adarq_env.sh
ADARQ_IMAGES="${E0_IMAGES:-${ADARQ_IMAGES:-}}"
[ -d "${ADARQ_IMAGES:-}" ] || die "ADARQ_IMAGES=${ADARQ_IMAGES:-<unset>} is not a directory"
n_img=$(find "$ADARQ_IMAGES" -maxdepth 1 -name '*.jpg' | wc -l)
[ "$n_img" -ge "$NUM_IMAGES" ] || die "only $n_img images in $ADARQ_IMAGES, need $NUM_IMAGES"

OUT="${OUT:-$ADARQ_CACHE/e0_${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT"

# --- the GPU must be an H200 -----------------------------------------------------------
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
echo "### host=$(hostname) gpu='${gpu:-none}' job=${SLURM_JOB_ID:-local}"
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
python - <<'PY' || die "torch cannot use the GPU"
import torch
assert torch.cuda.is_available(), "CUDA not available"
cap = torch.cuda.get_device_capability(0)
assert cap == (9, 0), f"expected Hopper sm_90, got sm_{cap[0]}{cap[1]}"
print(f"### torch {torch.__version__} on {torch.cuda.get_device_name(0)} (sm_90)")
PY

echo "=== E0: ${NUM_IMAGES} distinct images, layer ${LAYER}, seeds ${SEEDS[*]} -> $OUT ==="

# --- 1. real CLIP latents from the prefetched images -----------------------------------
FINAL="$OUT/final.pt"            # final-layer latents for the latent-L2 E0 (A)
LAYERED="$OUT/L${LAYER}.pt"      # layer-L latents + tower context for downstream (B)
python scripts/extract_clip_latents.py --out "$FINAL" --num-images "$NUM_IMAGES" \
    --image-dir "$ADARQ_IMAGES" --device cuda
python scripts/extract_clip_latents.py --out "$LAYERED" --num-images "$NUM_IMAGES" \
    --layer "$LAYER" --image-dir "$ADARQ_IMAGES" --device cuda
test -f "${LAYERED}.ctx.pt" || die "tower context ${LAYERED}.ctx.pt missing"
python - "$FINAL" "$NUM_IMAGES" <<'PY'
import json, pathlib, sys
p, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
meta = json.loads(p.with_suffix(p.suffix + ".meta.json").read_text())
if meta.get("synthetic") is not False:
    sys.exit("latents are not marked real — refusing to continue")
if int(meta["num_images"]) != n:
    sys.exit(f"extracted {meta['num_images']} images, expected {n}")
print(f"### provenance OK: {meta['num_images']} real images from {meta['source']}")
PY

# --- 2. seeds, one after another on this single H200 -----------------------------------
for seed in "${SEEDS[@]}"; do
  echo "--- seed $seed: (A) latent-L2 E0"
  python scripts/e0_headroom_pilot.py --latents "$FINAL" --preset "$PRESET" \
      --fit-steps "$FIT_STEPS" --mean-depth "$MEAN_DEPTH" --seed "$seed" \
      2>&1 | tee "$OUT/latent_l2_seed${seed}.log"
  echo "--- seed $seed: (B) downstream E0"
  python scripts/e0_headroom_pilot.py --latents "$LAYERED" --preset "$PRESET" \
      --downstream --device cuda --max-images "$NUM_IMAGES" \
      --fit-steps "$FIT_STEPS" --mean-depth "$MEAN_DEPTH" --seed "$seed" \
      --json-out "$OUT/seed${seed}.json" \
      2>&1 | tee "$OUT/downstream_seed${seed}.log"
done

# --- 3. aggregate the downstream runs and apply the pre-registered rules ---------------
python - "$OUT" "${#SEEDS[@]}" <<'PY'
import json, pathlib, statistics as st, sys

out, expected = pathlib.Path(sys.argv[1]), int(sys.argv[2])
runs = [json.loads(p.read_text()) for p in sorted(out.glob("seed*.json"))]
if len(runs) != expected:
    raise SystemExit(f"{len(runs)} seed results found, expected {expected}")
if len(runs) < 3:
    raise SystemExit(f"only {len(runs)} seeds; EXPERIMENTS.md R2 needs >= 3")

KEYS = ("headroom", "latent_l2_gain", "random_gain", "criterion_gap")
def ms(key):
    v = [r[key] for r in runs]
    return st.mean(v), st.stdev(v)

n_img = runs[0]["num_images"]
print("\n" + "=" * 72)
print(f"E0 DOWNSTREAM — {n_img} real images x {len(runs)} seeds, layer {runs[0]['layer']}, "
      f"mean depth {runs[0]['mean_depth_budget']:.2f}")
print("=" * 72)
print(f"{'metric':<26}{'mean':>10}{'std':>10}")
for k in KEYS:
    m, s = ms(k)
    print(f"{k:<26}{m:>9.1%}{s:>10.1%}")

h_m, h_s = ms("headroom")
r_m, _ = ms("random_gain")
g_m, _ = ms("criterion_gap")
FLOOR = 0.10

print()
if n_img < 32:
    print(f"VERDICT: UNDERPOWERED — {n_img} images (< 32).")
elif h_m < FLOOR:
    print(f"VERDICT: REFUTED on the correct objective — headroom {h_m:.1%} < {FLOOR:.0%}. "
          "Allocation does not matter downstream; do not proceed to C3.")
elif h_m - 2 * h_s < FLOOR:
    print(f"VERDICT: INCONCLUSIVE — headroom {h_m:.1%} +- {h_s:.1%}; the 2-sigma lower "
          f"bound ({h_m - 2*h_s:.1%}) falls below the {FLOOR:.0%} floor (R1).")
elif r_m >= h_m - 1e-9:
    print("VERDICT: CONTENT-AWARENESS REFUTED — random matches the oracle at equal rate.")
elif g_m < 0.10:
    print(f"VERDICT: PROCEED, but latent-L2 already captures {1-g_m:.0%} of the headroom "
          "— a downstream-aware allocator has little left to win; C3 would be thin.")
else:
    print(f"VERDICT: PROCEED — headroom {h_m:.1%} +- {h_s:.1%} on the correct objective, "
          f"and a latent-L2 allocator misses {g_m:.0%} of it. Optimizing reconstruction "
          "is measurably the wrong criterion (NOVELTY.md 6).")
print("\nNOTE: greedy search on a non-separable objective is not provably optimal, so "
      "headroom is a LOWER bound. The decoder is the real CLIP tower suffix, a SEMANTIC "
      "proxy — the eventual C3 claim needs the actual renderer.")
(out / "aggregate.json").write_text(json.dumps(
    {k: dict(zip(("mean", "std"), ms(k), strict=True)) for k in KEYS}, indent=2))
print(f"\nwrote {out/'aggregate.json'}")
PY

echo "=== done: $OUT  (latent-L2 verdicts per seed: $OUT/latent_l2_seed*.log) ==="
