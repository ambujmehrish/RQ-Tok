#!/bin/bash
# =====================================================================================
# Level B Stage 2: train the latent ControlNet on one H200 (resumable).
#
#   sbatch --qos=gpulongjobs scripts/levelb/train_controlnet.sbatch
#
# First run also encodes the 1000 class prompts with FLUX's text encoders (levelb.flux_text).
# 2 epochs at batch 48 is ~53k steps through the frozen 12B FLUX, i.e. several 24 h jobs:
# it checkpoints every 1000 steps and on SLURM's 10-minute warning; resubmit the SAME
# command to continue. If batch 48 does not fit, set MICRO=24 (or 16); note the upsample
# block's BatchNorm then sees micro-batch statistics (recorded in config.json).
# Needs scripts/levelb/precompute.sbatch.
# =====================================================================================
#SBATCH --job-name=levelb-controlnet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }

LEVELB_ROOT="${LEVELB_ROOT:-$HOME/work/levelb}"
BIFROST_ROOT="${BIFROST_ROOT:-$HOME/work/bifrost1}"
DATA="$LEVELB_ROOT/imagenet256"
OUT="${OUT:-$LEVELB_ROOT/controlnet}"
[ -f "$DATA/meta.json" ] || die "run scripts/levelb/precompute.sbatch first"
# shellcheck disable=SC1091
source "$BIFROST_ROOT/bifrost1_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u   # conda's activate.d/deactivate.d scripts read unset variables; nounset off for conda only
conda activate "$BIFROST_ENV"
set -u
# diffusers 0.33.1 (pinned) queries the Hub for a sharded checkpoint's file list even when
# cached, so FLUX.1-dev cannot load offline; compute nodes have internet.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -f "$DATA/flux_t5.npy" ]; then
  echo "### encoding the 1000 class prompts with FLUX's text encoders"
  python -m levelb.flux_text --data "$DATA"
fi

EXTRA=()
[ -n "${MICRO:-}" ] && EXTRA=(--micro-batch "$MICRO")
python -m levelb.train_controlnet --data "$DATA" --out "$OUT" "${EXTRA[@]+"${EXTRA[@]}"}" &
PID=$!
trap 'echo "### USR1: asking trainer to checkpoint"; kill -USR1 "$PID"' USR1
set +e
wait "$PID"; rc=$?
[ "$rc" -gt 128 ] && { wait "$PID"; rc=$?; }
set -e
[ "$rc" -eq 0 ] || die "trainer exited with $rc"
[ -f "$OUT/final_controlnet.pt" ] && echo "=== Stage 2 complete: $OUT ===" \
  || echo "=== not finished yet: resubmit this job to continue ==="
