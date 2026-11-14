#!/bin/bash
# =====================================================================================
# Level B sampling: 10,000 class-conditional 256x256 images for one arm and one seed.
#
#   sbatch --qos=gpulongjobs --export=ALL,ARM=bifrost,SEED=0 scripts/levelb/sample.sbatch
#   # split across jobs if needed:
#   for k in 0 1; do sbatch --qos=gpulongjobs --export=ALL,ARM=bifrost,SEED=0,SHARD=$k,NSHARDS=2 \
#       scripts/levelb/sample.sbatch; done
#
# Needs the finished Stage 1 (branch) and Stage 2 (ControlNet) checkpoints. When the last
# shard of a (arm, seed) set is written, merge with:
#   python -c "from levelb.sample import merge; print(merge('<out>', NSHARDS, 10000))"
# =====================================================================================
#SBATCH --job-name=levelb-sample
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }

ARM="${ARM:?set ARM=bifrost, adarq or flux}"
SEED="${SEED:?set SEED (0, 1, 2)}"
SHARD="${SHARD:-0}"; NSHARDS="${NSHARDS:-1}"
LEVELB_ROOT="${LEVELB_ROOT:-$HOME/work/levelb}"
BIFROST_ROOT="${BIFROST_ROOT:-$HOME/work/bifrost1}"
RUN="${RUN:-branch_${ARM}}"   # Stage-1 run folder, e.g. RUN=branch_adarq_tuned
TAG="${TAG:-$ARM}"            # sample folder name, e.g. TAG=adarq_tuned
BRANCH="${BRANCH:-$LEVELB_ROOT/$RUN/final_trainable.pt}"
CN="${CN:-$LEVELB_ROOT/controlnet/final_controlnet.pt}"
OUT="${OUT:-$LEVELB_ROOT/samples/${TAG}_seed${SEED}}"
if [ "$ARM" != "flux" ]; then   # flux: the backbone alone, no trained parts
  [ -f "$BRANCH" ] || die "$BRANCH missing: Stage 1 has not finished"
  [ -f "$CN" ] || die "$CN missing: Stage 2 has not finished"
fi

# shellcheck disable=SC1091
source "$BIFROST_ROOT/bifrost1_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u   # conda's activate.d/deactivate.d scripts read unset variables; nounset off for conda only
conda activate "$BIFROST_ENV"
set -u
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0   # diffusers 0.33.1 sharded-load Hub lookup
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

EXTRA=()
if [ "$ARM" = "adarq" ]; then
  HEADS="$LEVELB_ROOT/$RUN/final_heads.pt"
  [ -f "$HEADS" ] || die "$HEADS missing: AdaRQ-Flow Stage 1 has not finished"
  EXTRA=(--heads "$HEADS" --tok "$LEVELB_ROOT/adarq_tok")
fi
if [ "$ARM" != "flux" ]; then EXTRA+=(--branch "$BRANCH" --controlnet "$CN"); fi
python -m levelb.sample --arm "$ARM" "${EXTRA[@]+"${EXTRA[@]}"}" \
    --data "$LEVELB_ROOT/imagenet256" --seed "$SEED" --n 10000 \
    --shard "$SHARD" --nshards "$NSHARDS" --out "$OUT"
echo "=== shard $SHARD/$NSHARDS of $ARM seed $SEED done: $OUT ==="
