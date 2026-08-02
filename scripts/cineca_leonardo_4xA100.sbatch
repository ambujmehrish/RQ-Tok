#!/bin/bash
# ======================================================================================
# AdaRQ-Flow — Cineca Leonardo (Booster) — 1 node x 4x A100-64GB, DDP.
#
# Submit with:  sbatch scripts/cineca_leonardo_4xA100.sbatch
# Fill in the <PLACEHOLDERS> (account, venv path, module names) before submitting.
# ======================================================================================
#SBATCH --job-name=adarq-flow
#SBATCH --account=<YOUR_PROJECT_ACCOUNT>      # e.g. saturday/your Cineca project
#SBATCH --partition=boost_usr_prod            # Leonardo Booster GPU partition
#SBATCH --qos=normal                          # or boost_qos_dbg for short debug jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4                   # 1 task per GPU (rank == task)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8                      # 32 cores / 4 tasks on a Booster node
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# --- environment ----------------------------------------------------------------------
# module load cuda/12.x                       # match your installed PyTorch CUDA
# source <PATH_TO_YOUR_VENV>/bin/activate     # venv/conda with: pip install -e ".[torch]"

# --- rendezvous (SLURM does NOT set these; we must) -----------------------------------
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS                # consumed by setup_distributed()
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- NCCL on Leonardo's fabric --------------------------------------------------------
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
# export NCCL_IB_HCA=mlx5                      # uncomment/tune if InfiniBand needs it
# export NCCL_SOCKET_IFNAME=ib0

# Each srun task becomes one rank; setup_distributed() reads SLURM_PROCID/LOCALID.
srun python scripts/train_tokenizer_ddp.py \
    --preset base_gpu \
    --steps 1000 \
    --batch-size 256

# For real Phase-4 training, swap the entrypoint for the stage trainer (same launch),
# e.g.:  srun python -m adarq_flow.training.train --preset base_gpu --stage tokenizer
