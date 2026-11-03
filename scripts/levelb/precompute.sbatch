#!/bin/bash
# =====================================================================================
# Level B step 1: ImageNet train -> 256-px crops + Qwen2.5-VL-3B visual latents (once).
#
#   sbatch --qos=gpulongjobs scripts/levelb/precompute.sbatch
#
# Needs: scripts/imagenet/prefetch.sbatch (ImageNet parquet) and scripts/bifrost1/setup.sbatch
# (the bifrost1 env + upstream code). Runs upstream's frozen Qwen2.5-VL visual tower, offline.
# ~590 GB written (crops 251 GB + latents 336 GB). Resumable shard by shard: if the 24 h
# limit hits, resubmit the same command.
# =====================================================================================
#SBATCH --job-name=levelb-precompute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }

IMAGENET_ROOT="${IMAGENET_ROOT:-$HOME/work/imagenet}"
LEVELB_ROOT="${LEVELB_ROOT:-$HOME/work/levelb}"
BIFROST_ROOT="${BIFROST_ROOT:-$HOME/work/bifrost1}"
[ -f "$IMAGENET_ROOT/PROVENANCE.json" ] || die "run scripts/imagenet/prefetch.sbatch first"
[ -f "$BIFROST_ROOT/bifrost1_env.sh" ] || die "run scripts/bifrost1/setup.sbatch first"
# shellcheck disable=SC1091
source "$BIFROST_ROOT/bifrost1_env.sh"        # BIFROST_CODE, HF_HOME, HF_HUB_OFFLINE=1
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u   # conda's activate.d/deactivate.d scripts read unset variables; nounset off for conda only
conda activate "$BIFROST_ENV"
set -u
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"

OUT="$LEVELB_ROOT/imagenet256"
mkdir -p "$OUT"
free=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
[ -f "$OUT/crops_train.npy" ] || [ "$free" -ge 650 ] || die "$OUT has ${free}G free; need >= 650G"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python -m levelb.precompute --imagenet "$IMAGENET_ROOT/hf" --out "$OUT" \
    --vlm Qwen/Qwen2.5-VL-3B-Instruct --batch 128

# fixed held-out slice (checkpoint selection only) and the class prompts
python - "$OUT" "$IMAGENET_ROOT/hf/classes.py" <<'PY'
import sys
from levelb import imagenet as inet
out, classes = sys.argv[1:]
train, held = inet.heldout_split(f"{out}/split_seed0.npz", seed=0)
names = inet.class_names(classes)
inet.save_class_names(names, f"{out}/class_names.json")
print(f"split: {len(train)} train / {len(held)} held-out | {len(names)} class prompts, "
      f"e.g. {names[:3]}")
PY
cp "$IMAGENET_ROOT/PROVENANCE.json" "$OUT/imagenet_provenance.json"
echo "=== done: $OUT ==="
