#!/bin/bash
# =====================================================================================
# Inception Score decomposition (levelb/is_decomp.py) with the unmodified ADM evaluator's
# Inception network: ln IS = H(marginal) - mean H(p(y|x)), plus per requested class.
#
#   SAMPLES=<a.npz>,<b.npz> sbatch --qos=gpulongjobs --export=ALL scripts/bench/is_decomp.sbatch  # list: no --export
#   sbatch --qos=gpulongjobs --export=ALL,REAL=1 scripts/bench/is_decomp.sbatch   # ADM reference batch
#
# Each sample npz must already have its ADM score (<stem>.adm.metrics.json), whose IS this
# must reproduce. Same environment, node type and CPU checks as adm_score.sbatch.
# =====================================================================================
#SBATCH --job-name=is-decomp
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
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
set +u; conda activate "$ADM_ENV"; set -u
[ "$(git -C "$ADM_DIR/guided-diffusion" rev-parse HEAD)" = "$ADM_COMMIT" ] || die "evaluator checkout moved"
export TMPDIR="/tmp/$USER-${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
NCPU="${SLURM_CPUS_PER_TASK:?run through sbatch}"
export OMP_NUM_THREADS="$NCPU" OPENBLAS_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
CPU_MODEL=$(lscpu | sed -n 's/^Model name: *//p' | head -1)
case "$CPU_MODEL" in
  *"Xeon(R) 6760P"*) export OPENBLAS_CORETYPE=SkylakeX ;;
esac
[ -f "$BENCH_ROOT/ADM_CALIBRATED" ] || die "evaluator not calibrated yet"
CAL_CPU=$(sed 's/^[^|]*| //' "$BENCH_ROOT/ADM_CALIBRATED")
[ "$CAL_CPU" = "$CPU_MODEL" ] || die "calibrated on '$CAL_CPU', this node is '$CPU_MODEL'"
echo "  node $(hostname) | $CPU_MODEL | $NCPU threads | OpenBLAS core ${OPENBLAS_CORETYPE:-auto}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [ "${REAL:-0}" = 1 ]; then
  python -m levelb.is_decomp --samples "$ADM_DIR/VIRTUAL_imagenet256_labeled.npz" --no-labels
else
  SAMPLES="${SAMPLES:?set SAMPLES=<samples_10000.npz>[,...] or REAL=1}"
  IFS=, read -r -a LIST <<< "$SAMPLES"
  for s in "${LIST[@]}"; do
    [ -f "$s" ] || die "$s missing"
    [ -f "${s%.npz}.adm.metrics.json" ] || die "${s%.npz}.adm.metrics.json missing: score it first"
  done
  for s in "${LIST[@]}"; do
    python -m levelb.is_decomp --samples "$s" --expect "${s%.npz}.adm.metrics.json"
  done
fi
echo "=== is_decomp done ==="
