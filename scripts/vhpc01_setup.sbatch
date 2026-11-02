#!/bin/bash
# =====================================================================================
# STEP 1 of the vhpc01 setup: survey the cluster from a GPU compute node.
#
#   sbatch scripts/vhpc01_setup.sbatch
#
# If SLURM rejects it because a partition or account is required, add them — the error
# message names what is missing, and `sinfo -s` / `sacctmgr show assoc user=$USER` list
# the valid values:
#   sbatch -p <partition> -A <account> scripts/vhpc01_setup.sbatch
#
# Deliberately NO hardcoded #SBATCH --partition / --account / --qos: those are exactly
# the facts this job exists to discover, and a wrong guess just produces a confusing
# rejection. Everything else is sized small so it schedules quickly.
#
# It installs nothing and writes nothing outside logs/. Output lands in
# logs/adarq-setup-<jobid>.out — paste that back to continue.
# =====================================================================================
#SBATCH --job-name=adarq-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
# (gpus partition also holds Tesla/A40 nodes vhpc2/vhpc3; vhpc23 is the only H200 node)
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out        # one file: easier to paste back

set -uo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "### AdaRQ-Flow vhpc01 setup survey"
echo "### submitted from: ${SLURM_SUBMIT_DIR:-?}"
date

bash scripts/vhpc01_discover.sh

echo
echo "===== what happens next ====="
cat <<'EOF'
Depending on the egress result above:

  COMPUTE NODE HAS INTERNET
    the model/data prefetch can run as an ordinary job:
      ADARQ_CACHE=$HOME/work/adarq-cache sbatch scripts/vhpc01_prefetch.sbatch

  COMPUTE NODE HAS NO INTERNET
    the weights must be fetched wherever egress exists (login node, or copied in),
    then every run goes fully offline via $ADARQ_CACHE/adarq_env.sh (HF_HUB_OFFLINE=1).

Either way, FLUX.1-dev and Qwen2.5-VL are GATED: a token must be present
(hf auth login, or HF_TOKEN) on whichever machine does the downloading, and the model
licences accepted on huggingface.co, or the fetch returns 401.
EOF
echo "### done"
