#!/bin/bash
# =====================================================================================
# Diagnose the ADM evaluator's FID failure on vhpc01 (CPU only: the evaluator runs on
# tensorflow-cpu, so no GPU is requested and the job does not wait behind training).
#
#   sbatch --qos=gpulongjobs scripts/bench/adm_diag.sbatch
#
# Recomputes the calibration batch's statistics with the evaluator's own code (~1 h),
# then replays its FID under several OpenBLAS thread/kernel settings. Diagnostic only.
# =====================================================================================
#SBATCH --job-name=adm-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }

BENCH_ROOT="${BENCH_ROOT:-$HOME/work/bench}"
[ -f "$BENCH_ROOT/adm_env.sh" ] || die "run scripts/bench/adm_setup.sbatch first"
# shellcheck disable=SC1091
source "$BENCH_ROOT/adm_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u   # conda's activate.d/deactivate.d scripts read unset variables; nounset off for conda only
conda activate "$ADM_ENV"
set -u
[ "$(git -C "$ADM_DIR/guided-diffusion" rev-parse HEAD)" = "$ADM_COMMIT" ] || die "evaluator checkout moved"
export TMPDIR="/tmp/$USER-${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
sha256sum -c "$ADM_DIR/SHA256SUMS" || die "downloaded batches changed since setup"
python "$REPO/scripts/bench/adm_diag.py" "$ADM_DIR" "$BENCH_ROOT/adm_diag"
