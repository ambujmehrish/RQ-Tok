#!/bin/bash
# =====================================================================================
# Oracle renders (levelb/oracle_render.py): real held-out images' latents, or their
# Stage-0 codes with the residual dropped, through the same latent ControlNet + FLUX.
# No MLLM, no trained head. Validation only.
#
#   sbatch --qos=gpulongjobs --export=ALL,MODE=latent scripts/levelb/oracle_render.sbatch
#   MODE: latent | adaptive | depth1 .. depth4      N (default 2000), SEED (default 100)
# =====================================================================================
#SBATCH --job-name=oracle-render
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }

MODE="${MODE:?set MODE=latent, adaptive or depthK}"
SEED="${SEED:-100}"; N="${N:-2000}"
LEVELB_ROOT="${LEVELB_ROOT:-$HOME/work/levelb}"
BIFROST_ROOT="${BIFROST_ROOT:-$HOME/work/bifrost1}"
CN="${CN:-$LEVELB_ROOT/controlnet/final_controlnet.pt}"
TOK="${TOK:-$LEVELB_ROOT/adarq_tok}"
OUT="${OUT:-$LEVELB_ROOT/samples/oracle_${MODE}_seed${SEED}}"
[ -f "$CN" ] || die "$CN missing: Stage 2 has not finished"
[ -f "$TOK/tokenizer.pt" ] && [ -f "$TOK/codes_train.npy" ] || die "$TOK incomplete: Stage 0 has not finished"

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

python -m levelb.oracle_render --mode "$MODE" --tok "$TOK" --controlnet "$CN" \
    --data "$LEVELB_ROOT/imagenet256" --n "$N" --seed "$SEED" --out "$OUT" \
    --batch "${BATCH:-16}"
echo "=== oracle render $MODE done: $OUT ==="
