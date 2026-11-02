#!/bin/bash
# =====================================================================================
# STEP 2 of the vhpc01 setup: build the conda env on a GPU node and VERIFY it.
#
#   sbatch scripts/vhpc01_env.sbatch
#   sbatch -p <partition> -A <account> scripts/vhpc01_env.sbatch
#
# Why a GPU job rather than the login node: the only check that matters is whether torch
# can actually see and use an H200, and that can only be answered on a node that has
# one. A CPU-only "pip install succeeded" tells you nothing.
#
# Idempotent: re-running updates the existing env instead of rebuilding it.
# Output: logs/adarq-env-<jobid>.out
# =====================================================================================
#SBATCH --job-name=adarq-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

ENV_NAME="${ENV_NAME:-adarq}"
PY_VER="${PY_VER:-3.11}"
# H200 is Hopper (sm_90) and needs a CUDA 12.x build. cu124 wheels cover 12.4+ drivers;
# override if the survey showed an older driver.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "### env=$ENV_NAME python=$PY_VER torch-index=$TORCH_INDEX"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

# --- conda from the user's existing miniforge --------------------------------------
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || { echo "no conda.sh at $CONDA_SH" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "### env '$ENV_NAME' exists — updating in place"
else
  echo "### creating env '$ENV_NAME'"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"
python -VV

# --- deps --------------------------------------------------------------------------
python -m pip install --upgrade pip
echo "### torch (CUDA build for H200)"
python -m pip install --index-url "$TORCH_INDEX" torch
echo "### project + model/data deps"
python -m pip install -e ".[dev,torch,models]"
python -m pip install pillow            # real image loading

# --- VERIFY: the whole point of running this on a GPU node -------------------------
echo
echo "===== verification ====="
python - <<'PY'
import sys
import torch

print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("FAIL: torch cannot see a GPU on a node that has one — wrong wheel?")
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory/1e9:.0f} GB  sm_{p.major}{p.minor}")
cap = torch.cuda.get_device_capability(0)
if cap[0] < 9:
    print(f"  NOTE: sm_{cap[0]}{cap[1]} is not Hopper; expected sm_90 for H200")
# A real matmul in bf16: catches a wheel that imports but cannot execute.
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
assert torch.isfinite(x @ x).all(), "bf16 matmul produced non-finite values"
print(f"  bf16 matmul OK on {n} visible GPU(s)")
PY

echo
echo "===== project test suite (in the new env) ====="
python -m pytest -q 2>&1 | tail -5
python -m ruff check . 2>&1 | tail -2
python -m mypy adarq_flow/ 2>&1 | tail -2

cat <<EOF

===== activate it in later jobs =====
  source \$HOME/miniforge3/etc/profile.d/conda.sh
  conda activate $ENV_NAME

Next: fetch the models/data (needs egress + a HF token for the gated repos):
  ADARQ_CACHE=\$HOME/work/adarq-cache sbatch scripts/vhpc01_prefetch.sbatch
EOF
echo "### done"
