#!/usr/bin/env bash
# =====================================================================================
# Discover what this cluster actually offers, so the sbatch files stop guessing.
#
# Submit it (this cluster is submit-only):
#   sbatch scripts/vhpc01_setup.sbatch                 # runs this on a GPU node
#   sbatch -p <partition> -A <account> scripts/vhpc01_setup.sbatch
#
# Read-only: it submits nothing and installs nothing. Paste the output back and the
# sbatch templates can be filled in with real partition / account / GPU facts.
#
# It reports WHERE it ran, because the critical unknown is whether COMPUTE nodes have
# outbound internet. If they do, the model prefetch can be a job like everything else.
# If they do not, the weights must reach $ADARQ_CACHE another way, and every run has to
# be fully offline (HF_HUB_OFFLINE=1).
# =====================================================================================
set -uo pipefail          # no -e: a missing command should not abort the whole survey

have() { command -v "$1" >/dev/null 2>&1; }
sec()  { printf '\n===== %s =====\n' "$1"; }

sec "where am I running"
hostname; uname -sr
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "context: SLURM job ${SLURM_JOB_ID} on node(s) ${SLURM_JOB_NODELIST:-?}"
  echo "         partition=${SLURM_JOB_PARTITION:-?} gpus=${SLURM_GPUS_ON_NODE:-?}"
  echo "  -> egress results below are for a COMPUTE node (what the prefetch job needs)"
else
  echo "context: LOGIN node (not a SLURM job)"
  echo "  -> egress results below are for the login node only; compute nodes may differ"
fi
[ -r /etc/os-release ] && grep -E '^(PRETTY_NAME|VERSION)=' /etc/os-release

sec "storage (free space decides where the model cache goes)"
df -h "$HOME" /scratch /tmp 2>/dev/null
echo "--- quota (empty output usually means no quota enforced) ---"
have quota && quota -s 2>/dev/null | head -5
have lfs && lfs quota -h "$HOME" 2>/dev/null | head -5

sec "SLURM: partitions, limits, accounts"
if have sinfo; then
  sinfo -o '%20P %5a %10l %6D %8G %N' 2>/dev/null
  echo "--- GPU gres per node ---"
  sinfo -o '%20N %10G %20f' -h 2>/dev/null | sort -u
  echo "--- your accounts / QoS ---"
  have sacctmgr && sacctmgr -nP show assoc user="$USER" format=Account,Partition,QOS 2>/dev/null
  echo "--- default account ---"
  have sacctmgr && sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null
  echo "--- current queue ---"
  squeue -o '%8i %12P %10u %8T %10M %6D %R' 2>/dev/null | head -12
else
  echo "sinfo not found: is SLURM on the PATH on this node?"
fi

sec "GPUs visible from this (login) node"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  nvidia-smi | grep -i cuda | head -2
else
  echo "no nvidia-smi on the login node (normal: GPUs usually live on compute nodes)"
fi

sec "modules"
if have module; then
  module avail 2>&1 | grep -iE 'cuda|python|gcc|nccl' | head -20
else
  echo "no module system (miniforge-only environment is fine)"
fi

sec "python / conda"
have python3 && python3 -VV
have conda && { conda --version; echo "envs:"; conda env list 2>/dev/null | head -10; }
have mamba && mamba --version | head -1
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,
'avail',torch.cuda.is_available())" 2>/dev/null || echo "torch not importable in base env"

sec "network egress FROM HERE (decides whether prefetch can be a SLURM job)"
for host in https://huggingface.co http://images.cocodataset.org https://pypi.org; do
  code=$(curl -sS -o /dev/null -m 20 -w '%{http_code}' "$host" 2>/dev/null)
  printf '  %-34s -> %s\n' "$host" "${code:-FAILED}"
done
echo "--- proxy vars ---"
env | grep -iE '^(https?_proxy|no_proxy)=' || echo "  (none set)"

sec "huggingface auth (FLUX.1-dev + Qwen are gated)"
echo "HF_HOME=${HF_HOME:-<unset>}  HF_TOKEN=${HF_TOKEN:+<set>}${HF_TOKEN:-<unset>}"
ls -d "$HOME/hf_cache" 2>/dev/null && du -sh "$HOME/hf_cache" 2>/dev/null
have hf && hf auth whoami 2>/dev/null | head -3
[ -f "$HOME/.cache/huggingface/token" ] && echo "token file present"

sec "done"
echo "Paste this output back to continue the setup."
