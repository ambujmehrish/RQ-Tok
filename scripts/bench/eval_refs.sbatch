#!/bin/bash
# =====================================================================================
# Reference sets for locating the Bifrost-1 paper's ImageNet FID pipeline (CPU only).
#
#   sbatch --qos=gpulongjobs scripts/bench/eval_refs.sbatch
#
# Downloads OpenAI's ADM reference batches at 64 / 128 / 512 px (byte-range verified) and
# builds two from our data: ImageNet-val squashed to 256 (no crop) and the 10,000-image
# held-out train slice. None of these is a test reference of our protocol.
# =====================================================================================
#SBATCH --job-name=eval-refs
#SBATCH --partition=vhpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }

BENCH_ROOT="${BENCH_ROOT:-$HOME/work/bench}"
LEVELB_ROOT="${LEVELB_ROOT:-$HOME/work/levelb}"
BIFROST_ROOT="${BIFROST_ROOT:-$HOME/work/bifrost1}"
IMAGENET_ROOT="${IMAGENET_ROOT:-$HOME/work/imagenet}"
# shellcheck disable=SC1091
source "$BENCH_ROOT/adm_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
REFS="$LEVELB_ROOT/imagenet256/refs"
mkdir -p "$REFS"

BASE="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet"
set +u; conda activate "$ADM_ENV"; set -u
for r in 64/VIRTUAL_imagenet64_labeled 128/VIRTUAL_imagenet128_labeled 512/VIRTUAL_imagenet512; do
  dst="$ADM_DIR/$(basename $r).npz"
  [ -f "$dst" ] || python "$REPO/scripts/bench/fetch_ranged.py" "$BASE/$r.npz" "$dst" \
    || die "could not download $r (resubmit to resume)"
  python -c "import numpy as np,sys; z=np.load(sys.argv[1]); print(sys.argv[1], z['arr_0'].shape, sorted(z.files))" "$dst"
done
sha256sum "$ADM_DIR"/VIRTUAL_imagenet{64_labeled,128_labeled,512}.npz > "$REFS/ADM_EXTRA_SHA256SUMS"

source "$BIFROST_ROOT/bifrost1_env.sh"
set +u; conda activate "$BIFROST_ENV"; set -u
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$REFS/heldout_10000.npz" ] || python -m levelb.heldout_ref --data "$LEVELB_ROOT/imagenet256" \
  --out "$REFS/heldout_10000.npz"
[ -f "$REFS/val_squash_50000.npz" ] || python -m levelb.val_ref --imagenet "$IMAGENET_ROOT/hf" \
  --mode squash --out "$REFS/val_squash_50000.npz"
ls -la "$REFS"
echo "=== reference sets ready ==="
