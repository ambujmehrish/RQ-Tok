#!/bin/bash
# =====================================================================================
# Level B Stage 1, AdaRQ-Flow arm: Stage 0 tokenizer (once) + branch with AdaRQ heads (resumable).
#
#   sbatch --qos=gpulongjobs scripts/levelb/train_adarq.sbatch
#
# 16 epochs at batch 128 is ~159k steps, more than one 24 h job. The script checkpoints
# every 2000 steps and again when SLURM warns 10 minutes before the limit; resubmitting
# the SAME command continues from the latest checkpoint. Needs scripts/levelb/precompute.sbatch.
# =====================================================================================
#SBATCH --job-name=levelb-adarq
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
OUT="${OUT:-$LEVELB_ROOT/branch_adarq}"
TOK="${TOK:-$LEVELB_ROOT/adarq_tok}"
[ -f "$DATA/meta.json" ] || die "run scripts/levelb/precompute.sbatch first"
# shellcheck disable=SC1091
source "$BIFROST_ROOT/bifrost1_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u   # conda's activate.d/deactivate.d scripts read unset variables; nounset off for conda only
conda activate "$BIFROST_ENV"
set -u
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# SLURM signals the batch shell (B:USR1); forward it so Python checkpoints and exits.
if [ ! -f "$TOK/tokenizer.pt" ]; then
  echo "### Stage 0: fitting the adaptive RVQ tokenizer on the train split"
  python -m levelb.adarq_tokenizer --data "$DATA" --out "$TOK"
fi
python -m levelb.train_adarq --data "$DATA" --tok "$TOK" --out "$OUT" &
PID=$!
trap 'echo "### USR1: asking trainer to checkpoint"; kill -USR1 "$PID"' USR1
set +e
wait "$PID"; rc=$?
[ "$rc" -gt 128 ] && { wait "$PID"; rc=$?; }      # wait was interrupted by the trap
set -e
[ "$rc" -eq 0 ] || die "trainer exited with $rc"
grep -q '"heldout_mse"' "$OUT/metrics.jsonl" || echo "note: no held-out eval logged yet"
[ -f "$OUT/final_trainable.pt" ] && echo "=== AdaRQ-Flow Stage 1 complete: $OUT ===" \
  || echo "=== not finished yet: resubmit this job to continue ==="
