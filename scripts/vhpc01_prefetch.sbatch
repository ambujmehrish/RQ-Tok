#!/bin/bash
# =====================================================================================
# STEP 3 of the vhpc01 setup: fetch models + real data into $ADARQ_CACHE.
#
#   ADARQ_CACHE=$HOME/work/adarq-cache sbatch scripts/vhpc01_prefetch.sbatch
#   ADARQ_CACHE=... sbatch --export=ALL,GATED=1 scripts/vhpc01_prefetch.sbatch
#
# Only useful if the STEP 1 survey showed that COMPUTE nodes have outbound internet.
# If they do not, this job fails fast at the first download — it does not hang, and it
# never substitutes anything.
#
# Qwen2.5-VL-3B/7B are always pulled (open weights). GATED=1 adds FLUX.1-dev, which
# needs a Hugging Face token visible to the job (HF_TOKEN, or `hf auth login`
# beforehand) AND its licence accepted on huggingface.co.
# =====================================================================================
#SBATCH --job-name=adarq-prefetch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
# project rule: jobs run on H200 nodes only (the download itself needs no GPU)
#SBATCH --mem=16G
#SBATCH --time=04:00:00
# FLUX.1-dev alone is ~24 GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${ADARQ_CACHE:?set ADARQ_CACHE, e.g. ADARQ_CACHE=\$HOME/work/adarq-cache sbatch ...}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-adarq}"

# Fail in seconds, not after a long timeout, if this node has no egress.
code=$(curl -sS -o /dev/null -m 20 -w '%{http_code}' https://huggingface.co 2>/dev/null || true)
if [ "$code" != "200" ]; then
  echo "no egress to huggingface.co from $(hostname) (got '${code:-nothing}')." >&2
  echo "Compute nodes are offline: fetch on a machine with internet, copy into" >&2
  echo "\$ADARQ_CACHE, then run jobs offline via \$ADARQ_CACHE/adarq_env.sh." >&2
  exit 1
fi

if [ "${GATED:-0}" = "1" ]; then
  bash scripts/prefetch_assets.sh --gated
else
  bash scripts/prefetch_assets.sh
fi
echo "### done — later jobs: source $ADARQ_CACHE/adarq_env.sh"
