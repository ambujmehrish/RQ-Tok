#!/bin/bash
# =====================================================================================
# E0 DOWNSTREAM headroom — 64 real images x 3 seeds, on one Leonardo Booster node.
#
#   sbatch scripts/e0_downstream_64.sbatch
#
# Settles the question the CPU runs could not: does content-adaptive allocation help on
# the objective that actually matters (real CLIP embedding space), and does a
# latent-L2 allocator capture it? The CPU pilots were UNDERPOWERED — point estimates
# swung between +15.7% and +45.4% headroom on 6-8 images, with the random control moving
# from -0.3% to +12.9%. EXPERIMENTS.md R2 requires >= 3 seeds with mean +- std, and the
# report refuses a verdict below 32 images.
#
# Cost: the downstream objective is NON-SEPARABLE across patches, so the Lagrangian
# oracle does not apply and a greedy search is used: ~O(budget x N) decoder passes per
# image ~= 73 x 49 ~= 3.6k, i.e. ~230k per 64-image seed. Trivial on an A100, hours on CPU.
#
# PREREQUISITE (login node, has internet):
#   ADARQ_CACHE=$WORK/adarq-cache bash scripts/prefetch_assets.sh
# This job then runs fully offline.
# =====================================================================================
#SBATCH --job-name=adarq-e0-down
#SBATCH --account=<YOUR_PROJECT_ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3                  # one GPU per seed, run concurrently
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

NUM_IMAGES="${NUM_IMAGES:-64}"
LAYER="${LAYER:-8}"
FIT_STEPS="${FIT_STEPS:-600}"
MEAN_DEPTH="${MEAN_DEPTH:-2.5}"
PRESET="${PRESET:-clip_b32}"
SEEDS=(${SEEDS:-0 1 2})

# --- offline asset cache (never $HOME) ------------------------------------------------
if [ -z "${ADARQ_CACHE:-}" ]; then
  echo "ADARQ_CACHE is not set. Run scripts/prefetch_assets.sh on a LOGIN node first:" >&2
  echo "  ADARQ_CACHE=\$WORK/adarq-cache bash scripts/prefetch_assets.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$ADARQ_CACHE/adarq_env.sh"     # sets HF_HOME + HF_HUB_OFFLINE=1 + ADARQ_IMAGES
OUT="${OUT:-$ADARQ_CACHE/e0_downstream_${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT"

# module load cuda/12.x
# source <PATH_TO_YOUR_VENV>/bin/activate

echo "=== E0 downstream: ${NUM_IMAGES} images, layer ${LAYER}, seeds ${SEEDS[*]} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# --- 1. real CLIP latents at an intermediate layer (offline, from prefetched images) ---
LATENTS="$OUT/L${LAYER}.pt"
if [ ! -f "$LATENTS" ]; then
  python3 scripts/extract_clip_latents.py \
      --out "$LATENTS" \
      --num-images "$NUM_IMAGES" \
      --layer "$LAYER" \
      --image-dir "$ADARQ_IMAGES" \
      --device cuda
fi
test -f "${LATENTS}.ctx.pt" || { echo "tower context missing" >&2; exit 1; }

# --- 2. one seed per GPU, concurrently ------------------------------------------------
pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  CUDA_VISIBLE_DEVICES="$i" python3 scripts/e0_headroom_pilot.py \
      --latents "$LATENTS" \
      --preset "$PRESET" \
      --downstream \
      --device cuda \
      --max-images "$NUM_IMAGES" \
      --fit-steps "$FIT_STEPS" \
      --mean-depth "$MEAN_DEPTH" \
      --seed "$seed" \
      --json-out "$OUT/seed${seed}.json" \
      > "$OUT/seed${seed}.log" 2>&1 &
  pids+=("$!")
  echo "  launched seed $seed on GPU $i (pid ${pids[-1]})"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "seed ${SEEDS[$i]} FAILED — see $OUT/seed${SEEDS[$i]}.log" >&2
    fail=1
  fi
done
[ "$fail" -eq 0 ] || exit 1

# --- 3. aggregate across seeds and apply the pre-registered rules ----------------------
python3 - "$OUT" <<'PY'
import json, pathlib, statistics as st, sys

out = pathlib.Path(sys.argv[1])
runs = [json.loads(p.read_text()) for p in sorted(out.glob("seed*.json"))]
if len(runs) < 3:
    raise SystemExit(f"only {len(runs)} seeds completed; EXPERIMENTS.md R2 needs >= 3")

def ms(key):
    v = [r[key] for r in runs]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

n_img = runs[0]["num_images"]
print("\n" + "=" * 72)
print(f"E0 DOWNSTREAM — {n_img} real images x {len(runs)} seeds, layer {runs[0]['layer']}, "
      f"mean depth {runs[0]['mean_depth_budget']:.2f}")
print("=" * 72)
print(f"{'metric':<26}{'mean':>10}{'std':>10}")
for k in ("headroom", "latent_l2_gain", "random_gain", "criterion_gap"):
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
    {k: dict(zip(("mean", "std"), ms(k), strict=True))
     for k in ("headroom", "latent_l2_gain", "random_gain", "criterion_gap")}, indent=2))
print(f"\nwrote {out/'aggregate.json'}")
PY

echo "=== done: $OUT ==="
