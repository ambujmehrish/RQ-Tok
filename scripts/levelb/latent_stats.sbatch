#!/bin/bash
# =====================================================================================
# Latent statistics (levelb/latent_stats.py): energy shares, spectrum, and a linear probe of
# class information in the true latents vs the Stage-0 code-only reconstructions.
# Validation statistics only; no generation.
#
#   sbatch --qos=gpulongjobs --export=ALL scripts/levelb/latent_stats.sbatch
# =====================================================================================
#SBATCH --job-name=latent-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
TOK="${TOK:-$LEVELB_ROOT/adarq_tok}"
[ -f "$TOK/tokenizer.pt" ] || die "$TOK/tokenizer.pt missing: Stage 0 has not finished"
# shellcheck disable=SC1091
source "$BIFROST_ROOT/bifrost1_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u; conda activate "$BIFROST_ENV"; set -u
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python -m levelb.latent_stats --data "$LEVELB_ROOT/imagenet256" --tok "$TOK" \
    --out "results/levelb/latent_stats.json" --per-class "${PER_CLASS:-50}"
echo "=== latent stats done: results/levelb/latent_stats.json ==="
