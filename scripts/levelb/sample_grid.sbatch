#!/bin/bash
# =====================================================================================
# Side-by-side sample grid (H200 partition; CPU work): same classes, one row per arm, a real held-out
# image on top. Writes results/levelb/figures/samples_seed${SEED}.png (+ .txt), which the
# job queue's collector pushes to git.
#
#   sbatch --qos=gpulongjobs --export=ALL,ARMS="flux bifrost adarq",SEED=0,N=16 \
#       scripts/levelb/sample_grid.sbatch
# =====================================================================================
#SBATCH --job-name=sample-grid
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
LEVELB_ROOT="${LEVELB_ROOT:-$HOME/work/levelb}"
BIFROST_ROOT="${BIFROST_ROOT:-$HOME/work/bifrost1}"
# shellcheck disable=SC1091
source "$BIFROST_ROOT/bifrost1_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u; conda activate "$BIFROST_ENV"; set -u
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
SEED="${SEED:-0}"
# shellcheck disable=SC2086
python -m levelb.sample_grid --root "$LEVELB_ROOT" --arms ${ARMS:-flux bifrost adarq} \
    --seed "$SEED" --n "${N:-16}" --out "results/levelb/figures/samples_seed${SEED}.png"
