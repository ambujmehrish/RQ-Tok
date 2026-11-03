#!/bin/bash
# =====================================================================================
# Level B data: ImageNet-1k (ILSVRC 2012) + the ADM ImageNet-256 FID reference batch.
#
#   sbatch --qos=gpulongjobs scripts/imagenet/prefetch.sbatch
#
# Bifrost-1 §5.1 trains on the ImageNet train split and reports FID / sFID / IS on 10K
# generated 256x256 images, following MAR and xAR, which score against the ADM reference
# batch. Both are fetched here, verified, and recorded with their exact revisions:
#   * ILSVRC/imagenet-1k (Hugging Face, GATED: the account must be approved first)
#     train 1,281,167 + validation 50,000 images, parquet shards (~150 GB)
#   * VIRTUAL_imagenet256_labeled.npz (openaipublic, guided-diffusion evaluations)
# The test split (no labels) is not downloaded. Needs internet on the compute node.
# =====================================================================================
#SBATCH --job-name=imagenet-prefetch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
# project rule: jobs run on H200 nodes only (the download itself needs no GPU)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }
step() { printf '\n=== %s ===\n' "$1"; }

IMAGENET_ROOT="${IMAGENET_ROOT:-$HOME/work/imagenet}"
REF_URL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
MIN_FREE_GB="${MIN_FREE_GB:-250}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=0

# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u   # conda's activate.d/deactivate.d scripts read unset variables; nounset off for conda only
conda activate "${ENV_NAME:-adarq}"
set -u
python -c "import huggingface_hub, pyarrow" 2>/dev/null \
  || die "the adarq env needs pyarrow: pip install -e \".[imagenet]\""
code=$(curl -sS -o /dev/null -m 20 -w '%{http_code}' https://huggingface.co || true)
[ "$code" = "200" ] || die "no internet from $(hostname)"

mkdir -p "$IMAGENET_ROOT"
free=$(df -BG --output=avail "$IMAGENET_ROOT" | tail -1 | tr -dc '0-9')
[ "$free" -ge "$MIN_FREE_GB" ] || die "$IMAGENET_ROOT has ${free}G free; need >= ${MIN_FREE_GB}G"
echo "  $IMAGENET_ROOT: ${free}G free"

step "ILSVRC/imagenet-1k (train + validation parquet)"
HF_DIR="$IMAGENET_ROOT/hf"
python - "$HF_DIR" > "$IMAGENET_ROOT/.revision" <<'PY'
import sys
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError
repo, out = "ILSVRC/imagenet-1k", sys.argv[1]
try:
    sha = HfApi().dataset_info(repo).sha
    snapshot_download(repo, repo_type="dataset", revision=sha, local_dir=out,
                      allow_patterns=["data/train-*", "data/validation-*",
                                      "classes.py", "README.md"])
except GatedRepoError as e:
    sys.exit(f"{repo} is gated and this account is not approved yet.\n"
             f"  request access at https://huggingface.co/datasets/{repo}\n{e}")
print(sha)
PY
REV=$(cat "$IMAGENET_ROOT/.revision")
echo "  revision $REV"

step "verify split sizes and labels from parquet metadata"
python - "$HF_DIR" <<'PY'
import glob, sys
import pyarrow.compute as pc
import pyarrow.parquet as pq
root = sys.argv[1]
want = {"train": 1_281_167, "validation": 50_000}
for split, n_want in want.items():
    files = sorted(glob.glob(f"{root}/data/{split}-*.parquet"))
    if not files:
        sys.exit(f"no {split} shards found")
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    labels = set()
    for f in files:
        labels.update(pc.unique(pq.read_table(f, columns=["label"])["label"]).to_pylist())
    print(f"  {split}: {len(files)} shards, {n:,} images, {len(labels)} classes "
          f"(label range {min(labels)}..{max(labels)})")
    if n != n_want:
        sys.exit(f"{split}: {n} images, expected {n_want}")
    if labels != set(range(1000)):
        sys.exit(f"{split}: labels are not exactly 0..999")
PY

step "ADM ImageNet-256 reference batch (FID / sFID / IS)"
REF="$IMAGENET_ROOT/VIRTUAL_imagenet256_labeled.npz"
if [ ! -f "$REF" ]; then
  curl -sS -L --fail -o "$REF.part" "$REF_URL" || die "could not download $REF_URL"
  mv "$REF.part" "$REF"
fi
python - "$REF" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1]) as z:
    keys = sorted(z.files)
    print(f"  keys: {keys}")
    for k in ("mu", "sigma", "mu_s", "sigma_s"):
        if k not in z.files:
            sys.exit(f"reference batch is missing {k!r}")
    print(f"  pool mu {z['mu'].shape}, spatial mu_s {z['mu_s'].shape}")
PY
REF_SHA=$(sha256sum "$REF" | cut -d' ' -f1)

cat > "$IMAGENET_ROOT/PROVENANCE.json" <<EOF
{
  "imagenet": "ILSVRC/imagenet-1k",
  "imagenet_revision": "$REV",
  "splits": {"train": 1281167, "validation": 50000},
  "adm_reference_batch": "$REF_URL",
  "adm_reference_sha256": "$REF_SHA"
}
EOF
step "done"
cat "$IMAGENET_ROOT/PROVENANCE.json"
