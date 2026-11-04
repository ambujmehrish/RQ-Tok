#!/bin/bash
# =====================================================================================
# DPG-Bench scoring with ELLA's official scorer (mPLUG-large judge), on one H200.
#
#   # 1. calibrate once:
#   sbatch --qos=gpulongjobs --export=ALL,CALIBRATE=1 scripts/bench/dpg_score.sbatch
#   # 2. score a folder of 2x2 grids named <prompt_file>.png:
#   sbatch --qos=gpulongjobs --export=ALL,IMAGES=<dir> scripts/bench/dpg_score.sbatch
#
# CALIBRATE=1 generates SD v1.5 grids (scripts/bench/dpg_generate_sd15.py) and requires the
# published 63.18 (ELLA Table 4) within +-2.0. The tolerance is wider than GenEval's
# because ELLA does not publish its SD v1.5 sampling settings (diffusers defaults are used).
# Coverage is enforced: ELLA's scorer skips images that raise and averages the rest, so the
# job fails unless all 1065 grids were scored.
# =====================================================================================
#SBATCH --job-name=dpg-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }
# multiprocessing workers cannot delete their temp dirs on NFS (".nfs..." busy);
# keep temporary files on the node's local disk.
export TMPDIR="/tmp/$USER-${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

BENCH_ROOT="${BENCH_ROOT:-$HOME/work/bench}"
[ -f "$BENCH_ROOT/dpg_env.sh" ] || die "run scripts/bench/dpg_setup.sbatch first"
# shellcheck disable=SC1091
source "$BENCH_ROOT/dpg_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u   # conda's activate.d/deactivate.d scripts read unset variables; nounset off for conda only
conda activate "$DPG_ENV"
set -u
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
[ "$(git -C "$ELLA_CODE" rev-parse HEAD)" = "$ELLA_COMMIT" ] || die "ELLA checkout moved"

CALIBRATE="${CALIBRATE:-0}"
RESOLUTION="${RESOLUTION:-512}"          # tile size of the 2x2 grid
if [ "$CALIBRATE" = "1" ]; then
  IMAGES="$BENCH_ROOT/dpg_calibration_sd15"
  python "$REPO/scripts/bench/dpg_generate_sd15.py" "$ELLA_CODE/dpg_bench/prompts" "$IMAGES"
elif [ ! -f "$BENCH_ROOT/DPG_CALIBRATED" ]; then
  die "scorer not calibrated yet; run with CALIBRATE=1 first"
fi
IMAGES="${IMAGES:?set IMAGES=<dir of 2x2 grids> or CALIBRATE=1}"
n=$(find "$IMAGES" -maxdepth 1 -name '*.png' | wc -l)
[ "$n" -eq 1065 ] || die "$IMAGES has $n grids, expected 1065"

OUT="${OUT:-$(dirname "$IMAGES")/$(basename "$IMAGES")_dpg_${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT"
RES="$OUT/results.txt"
( cd "$ELLA_CODE" && accelerate launch --num_machines 1 --num_processes 1 \
    --mixed_precision fp16 "$REPO/scripts/bench/dpg_compute.py" \
    --mplug-dir "$DPG_MPLUG" --ella-dir "$ELLA_CODE" -- \
    --image-root-path "$IMAGES" --resolution "$RESOLUTION" --pic-num 4 \
    --csv "$ELLA_CODE/dpg_bench/dpg_bench.csv" --res-path "$RES" ) 2>&1 | tee "$OUT/log.txt"

# ELLA's prepare_dpg_data skips CSV row 0 (`if i == 0: continue`), which is proposition 1 of
# partiprompts97; that prompt's dependency lookup then raises KeyError(1) for EVERY model,
# so the official script averages 1064 prompts. That one known exclusion is reproduced as
# is (ELLA's scorer is not patched); any other failure stops the job.
failed=$(sed -n 's/^Failed filename: \([^ ]*\).*/\1/p' "$OUT/log.txt" | sort -u | tr '\n' ' ')
[ "$failed" = "partiprompts97.png " ] || [ -z "$failed" ] \
  || die "unexpected scoring failures: $failed(see $OUT/log.txt)"
scored=$(grep -c '\.png, ' "$RES" || true)
want=$([ -z "$failed" ] && echo 1065 || echo 1064)
[ "$scored" -eq "$want" ] || die "only $scored grids scored, expected $want"
score=$(sed -n 's/^DPG-Bench score: //p' "$RES" | tail -1)
[ -n "$score" ] || die "no DPG-Bench score in $RES"

if [ "$CALIBRATE" = "1" ]; then
  python - "$score" <<'PY'
import sys
got, ref, tol = float(sys.argv[1]), 63.18, 2.0
print(f"\nCALIBRATION: SD v1.5 DPG-Bench {got:.2f} vs published {ref} (tolerance +-{tol})")
if abs(got - ref) > tol:
    sys.exit("CALIBRATION FAILED — do not use this scorer to report any model.")
print("CALIBRATION PASSED")
PY
  touch "$BENCH_ROOT/DPG_CALIBRATED"
fi
echo "=== DPG-Bench score: $score  ($scored grids scored; $OUT) ==="
