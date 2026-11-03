#!/bin/bash
# =====================================================================================
# GenEval scoring with the official evaluator, on one H200, offline.
#
#   # 1. calibrate the scorer once (deviation E1 changed its torch/mmcv build):
#   sbatch --qos=gpulongjobs --export=ALL,CALIBRATE=1 scripts/bench/geneval_score.sbatch
#   # 2. score a model's images (folder in GenEval layout: <idx>/metadata.jsonl, samples/):
#   sbatch --qos=gpulongjobs --export=ALL,IMAGES=<dir> scripts/bench/geneval_score.sbatch
#
# CALIBRATE=1 regenerates Stable Diffusion v1.5 with GenEval's own generator and defaults
# (4 images/prompt, 50 steps, scale 9.0, seed 42) and requires the published overall score
# 0.43 (GenEval README) within +-0.02. Outside that band the scorer is NOT trusted and
# the job fails; no model score is reported until calibration passes.
# =====================================================================================
#SBATCH --job-name=geneval-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }

BENCH_ROOT="${BENCH_ROOT:-$HOME/work/bench}"
[ -f "$BENCH_ROOT/geneval_env.sh" ] || die "run scripts/bench/geneval_setup.sbatch first"
# shellcheck disable=SC1091
source "$BENCH_ROOT/geneval_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$GENEVAL_ENV"
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
[ "$(git -C "$GENEVAL_CODE" rev-parse HEAD)" = "$GENEVAL_COMMIT" ] || die "geneval checkout moved"

CALIBRATE="${CALIBRATE:-0}"
if [ "$CALIBRATE" = "1" ]; then
  IMAGES="$BENCH_ROOT/geneval_calibration_sd15"
  if [ ! -f "$IMAGES/DONE" ]; then
    rm -rf "$IMAGES"
    python "$GENEVAL_CODE/generation/diffusers_generate.py" \
        "$GENEVAL_CODE/prompts/evaluation_metadata.jsonl" \
        --model "stable-diffusion-v1-5/stable-diffusion-v1-5" --outdir "$IMAGES"
    touch "$IMAGES/DONE"
  fi
fi
IMAGES="${IMAGES:?set IMAGES=<dir in GenEval layout> or CALIBRATE=1}"
if [ "$CALIBRATE" != "1" ] && [ ! -f "$BENCH_ROOT/GENEVAL_CALIBRATED" ]; then
  die "scorer not calibrated yet; run with CALIBRATE=1 first"
fi
[ -d "$IMAGES" ] || die "$IMAGES does not exist"

# the benchmark must be fully covered: 553 prompts x 4 images
n_prompts=$(find "$IMAGES" -mindepth 1 -maxdepth 1 -type d | wc -l)
n_imgs=$(find "$IMAGES" -path '*/samples/*.png' | wc -l)
[ "$n_prompts" -eq 553 ] || die "$IMAGES has $n_prompts prompt folders, expected 553"
[ "$n_imgs" -eq 2212 ] || die "$IMAGES has $n_imgs images, expected 2212"

OUT="${OUT:-$IMAGES/../$(basename "$IMAGES")_geneval_${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT"
python "$GENEVAL_CODE/evaluation/evaluate_images.py" "$IMAGES" \
    --outfile "$OUT/results.jsonl" --model-path "$GENEVAL_MODELS"
python "$GENEVAL_CODE/evaluation/summary_scores.py" "$OUT/results.jsonl" | tee "$OUT/summary.txt"
overall=$(sed -n 's/^Overall score (avg. over tasks): //p' "$OUT/summary.txt")
[ -n "$overall" ] || die "could not read the overall score from summary.txt"

if [ "$CALIBRATE" = "1" ]; then
  python - "$overall" <<'PY'
import sys
got, ref, tol = float(sys.argv[1]), 0.43, 0.02
print(f"\nCALIBRATION: SD v1.5 overall {got:.3f} vs published {ref:.2f} (tolerance +-{tol})")
if abs(got - ref) > tol:
    sys.exit("CALIBRATION FAILED — the rebuilt scorer does not reproduce GenEval; "
             "do not use it to report any model.")
print("CALIBRATION PASSED — the scorer reproduces the published SD v1.5 score.")
PY
  touch "$BENCH_ROOT/GENEVAL_CALIBRATED"
fi
echo "=== GenEval overall: $overall  ($OUT) ==="
