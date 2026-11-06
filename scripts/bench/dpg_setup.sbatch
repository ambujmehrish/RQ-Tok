#!/bin/bash
# =====================================================================================
# DPG-Bench scorer: ELLA's official compute_dpg_bench.py + mPLUG-large, on one H200.
#
#   sbatch --qos=gpulongjobs scripts/bench/dpg_setup.sbatch
#
# Official: github.com/TencentQQGYLab/ELLA @ 3c228f1 (prompts, dpg_bench.csv, scorer),
#           ModelScope damo/mplug_visual-question-answering_coco_large_en (the VQA judge).
# ELLA's requirements-for-dpg_bench.txt pins nothing; installed today it resolves to
# transformers 5 / modelscope 1.40, which cannot load mPLUG. The environment is therefore
# the one verified to run the scorer correctly (scripts/bench/dpg_requirements.lock.txt:
# modelscope 1.13.3, transformers 4.38.2, datasets 2.18.0, diffusers 0.27.2,
# huggingface_hub 0.23.5, fairseq 0.12.2 — the benchmark's release era, resolved in one
# pass), with torch 2.5.1+cu124 for the H200. Deviation D1: mPLUG is downloaded by
# a current ModelScope CLI and loaded from disk (scripts/bench/dpg_compute.py).
# Calibration (dpg_score.sbatch CALIBRATE=1) must reproduce SD v1.5 = 63.18 before use.
# =====================================================================================
#SBATCH --job-name=dpg-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
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
die() { echo "FAILED: $*" >&2; exit 1; }
step() { printf '\n=== %s ===\n' "$1"; }

ELLA_COMMIT="3c228f1dc6c4d3cad0a47493816151a419f14db3"
MPLUG_ID="damo/mplug_visual-question-answering_coco_large_en"
BENCH_ROOT="${BENCH_ROOT:-$HOME/work/bench}"
ENV_NAME="${DPG_ENV:-dpgbench}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=0

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
code=$(curl -sS -o /dev/null -m 20 -w '%{http_code}' https://huggingface.co || true)
[ "$code" = "200" ] || die "no internet from $(hostname)"
mkdir -p "$BENCH_ROOT"

step "ELLA @ $ELLA_COMMIT"
E="$BENCH_ROOT/ella"
[ -d "$E/.git" ] || git clone -q https://github.com/TencentQQGYLab/ELLA "$E"
git -C "$E" fetch -q origin && git -C "$E" checkout -q "$ELLA_COMMIT"
[ "$(git -C "$E" rev-parse HEAD)" = "$ELLA_COMMIT" ] || die "ELLA not at $ELLA_COMMIT"
n=$(find "$E/dpg_bench/prompts" -name '*.txt' | wc -l)
[ "$n" -eq 1065 ] || die "expected 1065 DPG-Bench prompts, found $n"

step "env '$ENV_NAME' (python 3.10, verified lock + torch 2.5.1+cu124)"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u   # conda's activate.d/deactivate.d scripts read unset variables; nounset off for conda only
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" \
  || conda create -y -q -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"
set -u
# fairseq 0.12.2 depends on omegaconf 2.0.x, whose metadata pip >= 24.1 rejects
python -m pip install -q "pip<24.1"
python -m pip install -q --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
# The lock is a complete, consistent environment (resolved in ONE pass and verified:
# mPLUG known-answer test, ELLA scorer, SD v1.5 generator import). Install it exactly as
# frozen — no re-resolution — then let pip check it is self-consistent.
python -m pip install -q --no-deps -r "$REPO/scripts/bench/dpg_requirements.lock.txt"
python -m pip check || die "the DPG-Bench environment is inconsistent (pip check failed)"

step "mPLUG-large (downloaded with a current ModelScope CLI in its own venv — D1)"
MPLUG="$BENCH_ROOT/mplug_large"
if [ ! -f "$MPLUG/pytorch_model.bin" ]; then
  DL="$BENCH_ROOT/_modelscope_dl"
  [ -x "$DL/bin/modelscope" ] || { python -m venv "$DL" && "$DL/bin/pip" install -q "modelscope==1.40.2"; }
  "$DL/bin/modelscope" download --model "$MPLUG_ID" --local_dir "$MPLUG"
fi
[ -f "$MPLUG/pytorch_model.bin" ] || die "mPLUG download incomplete"

step "SD v1.5 weights for calibration"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5", allow_patterns=[
    "*.json", "*.txt", "*/model.safetensors", "*/diffusion_pytorch_model.safetensors"])
print("  cached stable-diffusion-v1-5")
PY

step "verify: mPLUG answers known questions on the H200"
python - "$MPLUG" <<'PY'
import sys
import numpy as np
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
p = pipeline(Tasks.visual_question_answering, model=sys.argv[1], device="cuda")
red = Image.fromarray(np.full((512, 512, 3), (220, 30, 30), dtype=np.uint8))
blue = Image.fromarray(np.full((512, 512, 3), (30, 30, 220), dtype=np.uint8))
checks = [(red, "Is the image red?", "yes"), (red, "Is the image blue?", "no"),
          (blue, "Is the image blue?", "yes"), (blue, "Is the image red?", "no")]
for img, q, want in checks:
    got = p({"image": img, "question": q})["text"]
    print(f"  {q} -> {got}")
    if got != want:
        sys.exit(f"mPLUG answered {got!r}, expected {want!r}: the judge is not working")
print("  mPLUG-large judge OK on the H200")
PY

cat > "$BENCH_ROOT/dpg_env.sh" <<EOF
# Generated by scripts/bench/dpg_setup.sbatch
export BENCH_ROOT="$BENCH_ROOT"
export ELLA_CODE="$E"
export ELLA_COMMIT="$ELLA_COMMIT"
export DPG_MPLUG="$MPLUG"
export DPG_ENV="$ENV_NAME"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
EOF
step "done — next: CALIBRATE=1 sbatch scripts/bench/dpg_score.sbatch"
