#!/bin/bash
# =====================================================================================
# Requested-class top-1/top-5 accuracy (torchvision ResNet-50 IMAGENET1K_V2) on one H200.
#
#   sbatch --qos=gpulongjobs --export=ALL,SAMPLES=<samples_N.npz> scripts/levelb/class_accuracy.sbatch
#   SAMPLES=<a.npz>,<b.npz> PER_CLASS=1 sbatch --qos=gpulongjobs --export=ALL scripts/levelb/class_accuracy.sbatch
#   sbatch --qos=gpulongjobs --export=ALL,HELDOUT_N=2000 scripts/levelb/class_accuracy.sbatch  # ceiling
# =====================================================================================
#SBATCH --job-name=class-acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }
LEVELB_ROOT="${LEVELB_ROOT:-$HOME/work/levelb}"
BIFROST_ROOT="${BIFROST_ROOT:-$HOME/work/bifrost1}"
# shellcheck disable=SC1091
source "$BIFROST_ROOT/bifrost1_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u; conda activate "$BIFROST_ENV"; set -u
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${HELDOUT_N:-}" ]; then
  python -m levelb.class_accuracy --heldout "$LEVELB_ROOT/imagenet256" --n "$HELDOUT_N" \
    --out "results/levelb/class_accuracy_heldout_${HELDOUT_N}.json"
else
  SAMPLES="${SAMPLES:?set SAMPLES=<samples_N.npz>[,...] (merged; run the scoring job first) or HELDOUT_N}"
  IFS=, read -r -a LIST <<< "$SAMPLES"
  for s in "${LIST[@]}"; do [ -f "$s" ] || die "$s missing"; done
  # PER_CLASS=1: per requested class as well (<stem>.clsclass.metrics.json)
  PC=(); [ "${PER_CLASS:-0}" = 1 ] && PC=(--per-class)
  for s in "${LIST[@]}"; do python -m levelb.class_accuracy --samples "$s" "${PC[@]+"${PC[@]}"}"; done
fi
