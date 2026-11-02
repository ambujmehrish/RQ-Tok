#!/bin/bash
# =====================================================================================
# STEP 2 of the vhpc01 setup: build the conda env and VERIFY it on an H200.
#
#   sbatch scripts/vhpc01_env.sbatch                       # full: build + H200 check
#   GPU_CHECK=defer bash scripts/vhpc01_env.sbatch         # login node, GPUs down
#
# The check that matters is whether torch can see and USE an H200, which only a node
# with one can answer. On the login node (e.g. while vhpc23 is in maintenance) the env
# can be built ahead of time, but the GPU check must be explicitly deferred with
# GPU_CHECK=defer, and the job re-submitted later. Without that flag a host with no GPU
# FAILS, so a CPU-only "install succeeded" is never mistaken for a verified env.
#
# Idempotent: re-running updates the existing env instead of rebuilding it.
# Output: logs/adarq-env-<jobid>.out
# =====================================================================================
#SBATCH --job-name=adarq-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
# (plain gpu:1 could land on the A40/Tesla nodes and "verify" a GPU we never train on)
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
mkdir -p logs

GPU_CHECK="${GPU_CHECK:-required}"           # required | defer
if nvidia-smi -L >/dev/null 2>&1; then
  HAVE_GPU=1
elif [ "$GPU_CHECK" = "defer" ]; then
  HAVE_GPU=0
  echo "### no GPU on $(hostname): building the env, GPU verification DEFERRED"
else
  echo "no GPU visible on $(hostname). Submit with sbatch to verify on an H200, or set" >&2
  echo "GPU_CHECK=defer to build the env now and verify later." >&2
  exit 1
fi

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
# Pin torch to the build just installed: otherwise a dependency asking for a newer
# torch makes pip swap in a PyPI build compiled for a different CUDA, silently undoing
# the H200-specific choice above.
TORCH_PIN="$(python -c 'import torch; print(torch.__version__)')"
echo "### project + model/data deps (torch pinned to $TORCH_PIN)"
python -m pip install -e ".[dev,torch,models]" -c <(echo "torch==$TORCH_PIN")
python -m pip install pillow            # real image loading

# --- VERIFY: the whole point of running this on a GPU node -------------------------
echo
echo "===== verification ====="
if [ "$HAVE_GPU" = "0" ]; then
  python -c "import torch; print('torch', torch.__version__, '| CUDA build', torch.version.cuda)"
  echo "GPU check DEFERRED: re-run 'sbatch scripts/vhpc01_env.sbatch' once vhpc23 is up."
else
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
name = torch.cuda.get_device_name(0)
if cap != (9, 0) or "H200" not in name:
    sys.exit(f"FAIL: this job asked for an H200 but got {name} (sm_{cap[0]}{cap[1]}). "
             "A pass here would certify a GPU we never train on.")
# A real matmul in bf16: catches a wheel that imports but cannot execute.
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
assert torch.isfinite(x @ x).all(), "bf16 matmul produced non-finite values"
print(f"  bf16 matmul OK on {n} visible GPU(s)")
PY
fi

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
