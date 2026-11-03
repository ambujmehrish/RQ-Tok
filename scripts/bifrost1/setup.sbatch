#!/bin/bash
# =====================================================================================
# Bifrost-1 REFERENCE (Level A), step 1: official code + env + checkpoint, on one H200.
#
#   sbatch --qos=gpulongjobs scripts/bifrost1/setup.sbatch
#
# Everything is the authors' own artefact, pinned:
#   code        github.com/HL-hanlin/Bifrost-1 @ 3047de271b54dbd37228b038cc1f949899b0f1a8
#   weights     huggingface.co/hanlincs/Bifrost-1  (revision recorded at download time)
#   env         their requirements.txt (torch 2.5.1, transformers 4.49.0, diffusers 0.33.1)
#               in a SEPARATE conda env 'bifrost1' — never mixed with the adarq env
#   DPG-Bench   prompts + dpg_bench.csv from the benchmark's own repo (TencentQQGYLab/ELLA),
#               because upstream Bifrost-1 does not ship them (deviation A2)
# plus the one file upstream forgot, bifrost/models/mar/utils.py (deviation A1).
#
# Needs internet on the compute node (the vhpc01 survey showed it has it).
# Output: $BIFROST_ROOT/{code,ckpt,bifrost1_env.sh,PROVENANCE.json}
# =====================================================================================
#SBATCH --job-name=bifrost1-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs

UPSTREAM_URL="https://github.com/HL-hanlin/Bifrost-1"
UPSTREAM_COMMIT="3047de271b54dbd37228b038cc1f949899b0f1a8"
ELLA_URL="https://github.com/TencentQQGYLab/ELLA"
CKPT_REPO="hanlincs/Bifrost-1"
BIFROST_ROOT="${BIFROST_ROOT:-$HOME/work/bifrost1}"
ENV_NAME="${BIFROST_ENV:-bifrost1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"          # already holds Qwen2.5-VL-7B + FLUX.1-dev
export HF_HUB_OFFLINE=0

die() { echo "FAILED: $*" >&2; exit 1; }
step() { printf '\n=== %s ===\n' "$1"; }

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
echo "### host=$(hostname) gpu='${gpu:-none}'"
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
code=$(curl -sS -o /dev/null -m 20 -w '%{http_code}' https://huggingface.co || true)
[ "$code" = "200" ] || die "no internet from $(hostname) (huggingface.co -> '${code:-nothing}')"
mkdir -p "$BIFROST_ROOT"

# --- 1. official code at the pinned commit --------------------------------------------
step "official code @ $UPSTREAM_COMMIT"
CODE="$BIFROST_ROOT/code"
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$UPSTREAM_URL" "$CODE"
fi
git -C "$CODE" fetch -q origin
git -C "$CODE" checkout -q "$UPSTREAM_COMMIT"
[ "$(git -C "$CODE" rev-parse HEAD)" = "$UPSTREAM_COMMIT" ] || die "checkout is not $UPSTREAM_COMMIT"
# A1: the module upstream imports but does not ship
mkdir -p "$CODE/bifrost/models/mar"
: > "$CODE/bifrost/models/mar/__init__.py"
cp "$REPO/scripts/bifrost1/mar_utils.py" "$CODE/bifrost/models/mar/utils.py"

# --- 2. DPG-Bench prompts from the benchmark's own repo (A2) ---------------------------
step "DPG-Bench prompts (ELLA)"
ELLA="$BIFROST_ROOT/ella"
[ -d "$ELLA/.git" ] || git clone -q --depth 1 "$ELLA_URL" "$ELLA"
ELLA_COMMIT=$(git -C "$ELLA" rev-parse HEAD)
DPG="$CODE/evaluation/DPGBench/dpg_bench"
mkdir -p "$DPG"
rm -rf "$DPG/prompts"
cp -r "$ELLA/dpg_bench/prompts" "$DPG/prompts"
cp "$ELLA/dpg_bench/dpg_bench.csv" "$DPG/dpg_bench.csv"
n_dpg=$(find "$DPG/prompts" -name '*.txt' | wc -l)
n_gen=$(grep -c . "$CODE/evaluation/GenEval/prompts/generation_prompts.txt")
echo "  DPG-Bench prompts: $n_dpg   GenEval prompts: $n_gen"
[ "$n_dpg" -gt 1000 ] || die "expected ~1065 DPG-Bench prompts, found $n_dpg"
[ "$n_gen" -eq 553 ] || die "expected 553 GenEval prompts, found $n_gen"

# --- 3. the authors' environment, separate from adarq ---------------------------------
step "conda env '$ENV_NAME' from upstream requirements.txt"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" \
  || conda create -y -q -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"
python -m pip install -q --upgrade pip
# torch 2.5.1 as pinned upstream, from the CUDA 12.4 index (H200 = sm_90)
python -m pip install -q --index-url "$TORCH_INDEX" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
# A3: pins that cannot be installed are replaced by the nearest installable release,
# listed here explicitly (never silently). av 14.3.0 has no files left on PyPI and 14.4.0
# ships no Python 3.11 wheel (it would compile against a system FFmpeg the nodes lack);
# 14.2.0 is the nearest release with a cp311 manylinux wheel. No upstream code imports
# av. A4: protobuf is UNPINNED upstream and today resolves to a release whose runtime
# rejects the generated code in the pinned sentencepiece 0.1.99 ("Descriptors cannot be
# created directly"), which breaks FLUX's T5 tokenizer; protobuf 3.20.3 is pinned (allowed
# by every Google package in the list). The also-unpinned google-api-core and
# googleapis-common-protos are pinned to their March-2025 releases (2.24.2 / 1.69.2), the
# upstream requirements' era, which accept protobuf 3.20.3 — newer ones require >= 4.25.8.
# Any OTHER uninstallable pin still fails the job.
REQ="$BIFROST_ROOT/requirements.resolved.txt"
sed -e 's/^av==14\.3\.0$/av==14.2.0/' -e 's/^protobuf$/protobuf==3.20.3/' \
    -e 's/^google-api-core$/google-api-core==2.24.2/' \
    -e 's/^googleapis-common-protos$/googleapis-common-protos==1.69.2/' \
    "$CODE/requirements.txt" > "$REQ"
diff "$CODE/requirements.txt" "$REQ" || true          # print every substitution made
[ "$(diff "$CODE/requirements.txt" "$REQ" | grep -c '^>')" -eq 4 ] \
  || die "expected exactly four substituted pins (A3/A4); requirements.txt changed upstream?"
python -m pip install -q --only-binary=av -r "$REQ" \
    -c <(printf 'torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\n')
python - <<'PY'
import torch, transformers, diffusers
assert torch.__version__.startswith("2.5.1"), torch.__version__
assert transformers.__version__ == "4.49.0", transformers.__version__
assert diffusers.__version__ == "0.33.1", diffusers.__version__
assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)
print(f"  torch {torch.__version__} / transformers {transformers.__version__} / "
      f"diffusers {diffusers.__version__} on {torch.cuda.get_device_name(0)}")
PY

# --- 4. the released checkpoint --------------------------------------------------------
step "checkpoint $CKPT_REPO"
CKPT="$BIFROST_ROOT/ckpt"
CKPT_SHA=$(python - "$CKPT_REPO" "$CKPT" <<'PY'
import sys
from huggingface_hub import HfApi, snapshot_download
repo, out = sys.argv[1:]
sha = HfApi().model_info(repo).sha
snapshot_download(repo, revision=sha, local_dir=out)
print(sha)
PY
)
echo "  $CKPT_REPO @ $CKPT_SHA -> $CKPT"
# the backbones the script loads by name must already be in the shared HF cache
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen2.5-VL-7B-Instruct", "black-forest-labs/FLUX.1-dev"):
    snapshot_download(repo, local_files_only=True)      # raises if not cached
    print(f"  cached: {repo}")
PY

# --- 5. the official modules import cleanly -------------------------------------------
step "import check (offline)"
( cd "$CODE" && HF_HUB_OFFLINE=1 python - <<'PY'
from bifrost.models.modeling_bifrost import MultiModalityCausalLM  # noqa: F401
from bifrost.models.mar.utils import sample_orders
from bifrost.conversation import Conversation  # noqa: F401
o = sample_orders(2, 256)
assert o.shape == (2, 256) and sorted(o[0].tolist()) == list(range(256))
print("  official modules import; sample_orders returns permutations")
PY
)

# --- 6. env file + provenance -----------------------------------------------------------
cat > "$BIFROST_ROOT/bifrost1_env.sh" <<EOF
# Generated by scripts/bifrost1/setup.sbatch — source in OFFLINE Bifrost-1 jobs.
export BIFROST_ROOT="$BIFROST_ROOT"
export BIFROST_CODE="$CODE"
export BIFROST_CKPT="$CKPT"
export BIFROST_ENV="$ENV_NAME"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
EOF
python - "$BIFROST_ROOT/PROVENANCE.json" <<PY
import json, sys
json.dump({
    "upstream": "$UPSTREAM_URL", "upstream_commit": "$UPSTREAM_COMMIT",
    "checkpoint": "$CKPT_REPO", "checkpoint_sha": "$CKPT_SHA",
    "dpg_bench_source": "$ELLA_URL", "dpg_bench_commit": "$ELLA_COMMIT",
    "deviations": {
        "A1": "bifrost/models/mar/utils.py missing upstream; sample_orders reproduced from MAR",
        "A2": "DPG-Bench prompts missing upstream; taken from TencentQQGYLab/ELLA",
        "A3": "av==14.3.0 removed from PyPI; av==14.2.0 (nearest cp311 wheel) installed; av is not imported upstream",
        "A4": "protobuf unpinned upstream; pinned to 3.20.3 (newer protobuf rejects sentencepiece 0.1.99 pb2, breaking the T5 tokenizer); google-api-core 2.24.2 and googleapis-common-protos 1.69.2 pinned to match",
    },
}, open(sys.argv[1], "w"), indent=2)
PY
step "done"
cat "$BIFROST_ROOT/PROVENANCE.json"
echo "next: sbatch --qos=gpulongjobs --export=ALL,BENCH=geneval scripts/bifrost1/generate.sbatch"
