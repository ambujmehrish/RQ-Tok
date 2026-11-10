#!/bin/bash
# =====================================================================================
# Why does the AdaRQ-Flow code head overfit? (diagnostic; held-out slice + train subset)
#
#   sbatch --qos=gpulongjobs scripts/levelb/adarq_diag.sbatch
#   # another run folder:  --export=ALL,RUN=branch_adarq_tuned
#
# Uses the run's final weights (or its latest checkpoint if unfinished); ~1 h on one H200.
# Writes <run>/diag/overfit_diag.json. Never touches the test set or a model.
# =====================================================================================
#SBATCH --job-name=adarq-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }

LEVELB_ROOT="${LEVELB_ROOT:-$HOME/work/levelb}"
BIFROST_ROOT="${BIFROST_ROOT:-$HOME/work/bifrost1}"
RUN="$LEVELB_ROOT/${RUN:-branch_adarq}"
[ -f "$RUN/config.json" ] || die "$RUN has no config.json"
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
python -m levelb.adarq_diag --data "$LEVELB_ROOT/imagenet256" --tok "$LEVELB_ROOT/adarq_tok" \
    --run "$RUN"
