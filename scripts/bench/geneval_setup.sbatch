#!/bin/bash
# =====================================================================================
# GenEval scorer: official code + detector, built for the H200.
#
#   sbatch --qos=gpulongjobs scripts/bench/geneval_setup.sbatch
#
# Official: github.com/djghosh13/geneval @ af4902f (evaluator, prompts, thresholds),
#           mmdetection 2.x @ e9cae2d, Mask2Former Swin-S COCO weights, CLIP ViT-L-14.
# ONE forced deviation (E1): GenEval's environment.yml pins PyTorch 1.12.1 / CUDA 11.3,
# which predates Hopper, and its evaluator asserts CUDA. The H200 (sm_90) needs CUDA >= 11.8,
# so torch is 2.0.1+cu118 and mmcv-full 1.7.1 (the pinned version) is compiled from source
# for sm_90. Everything else keeps GenEval's pins. Because the stack changed, the scorer is
# CALIBRATED before use: scripts/bench/geneval_score.sbatch CALIBRATE=1 must reproduce the
# published SD v1.5 score (0.43) or the scorer is not trusted.
#
# Needs internet on the compute node. Output: $BENCH_ROOT/{geneval,mmdetection,models,...}
# =====================================================================================
#SBATCH --job-name=geneval-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
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

GENEVAL_COMMIT="af4902f24d3ca90ebbb446dd9891a59e0f82725f"
MMDET_COMMIT="e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1"      # mmdetection branch 2.x
BENCH_ROOT="${BENCH_ROOT:-$HOME/work/bench}"
ENV_NAME="${GENEVAL_ENV:-geneval}"
SD15="stable-diffusion-v1-5/stable-diffusion-v1-5"            # calibration model only
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=0

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
code=$(curl -sS -o /dev/null -m 20 -w '%{http_code}' https://huggingface.co || true)
[ "$code" = "200" ] || die "no internet from $(hostname)"
mkdir -p "$BENCH_ROOT"

step "official code"
G="$BENCH_ROOT/geneval"; M="$BENCH_ROOT/mmdetection"
[ -d "$G/.git" ] || git clone -q https://github.com/djghosh13/geneval "$G"
git -C "$G" fetch -q origin && git -C "$G" checkout -q "$GENEVAL_COMMIT"
[ -d "$M/.git" ] || git clone -q https://github.com/open-mmlab/mmdetection "$M"
git -C "$M" fetch -q origin 2.x && git -C "$M" checkout -q "$MMDET_COMMIT"
[ "$(git -C "$G" rev-parse HEAD)" = "$GENEVAL_COMMIT" ] || die "geneval not at $GENEVAL_COMMIT"
[ "$(git -C "$M" rev-parse HEAD)" = "$MMDET_COMMIT" ] || die "mmdetection not at $MMDET_COMMIT"

step "env '$ENV_NAME' (python 3.9 as upstream; torch 2.0.1+cu118 for sm_90 — E1)"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" \
  || conda create -y -q -n "$ENV_NAME" python=3.9
conda activate "$ENV_NAME"
# nvcc + headers to compile mmcv's CUDA ops for sm_90
conda install -y -q -c "nvidia/label/cuda-11.8.0" cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
python -m pip install -q --upgrade "pip<24.1"
python -m pip install -q --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 torchvision==0.15.2
TORCH_PIN='torch==2.0.1
torchvision==0.15.2'
# Exactly what the evaluator imports, at GenEval's own pins (tested resolvable with
# torch 2.0.1): scripts/bench/geneval_requirements.txt. clip-benchmark 1.4.0 declares
# torch<2, so it goes in with --no-deps (E1); its zero-shot classifier was verified on
# torch 2.0 (correct colour predictions). pytorch-lightning 1.9.5 replaces GenEval's 1.4.2,
# which cannot import with torch 2.0; the calibration generator only uses seed_everything.
python -m pip install -q -c <(echo "$TORCH_PIN") -r "$REPO/scripts/bench/geneval_requirements.txt"
python -m pip install -q --no-deps clip-benchmark==1.4.0
step "mmcv-full 1.7.1 from source for sm_90 (slow)"
if ! python -c "import mmcv; from mmcv.ops import MultiScaleDeformableAttention" 2>/dev/null; then
  MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}" \
    python -m pip install -q --no-build-isolation -c <(echo "$TORCH_PIN") mmcv-full==1.7.1
fi
python -m pip install -q --no-build-isolation -c <(echo "$TORCH_PIN") -e "$M"

step "weights: Mask2Former (official script), CLIP ViT-L-14, SD v1.5 (calibration)"
MODELS="$BENCH_ROOT/geneval_models"
[ -f "$MODELS/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ] \
  || bash "$G/evaluation/download_models.sh" "$MODELS/"
python - "$SD15" <<'PY'
import sys
import open_clip
from huggingface_hub import snapshot_download
open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")   # caches weights
# GenEval's generator loads the default (non-variant) weights in fp16
snapshot_download(sys.argv[1], allow_patterns=[
    "*.json", "*.txt", "*/model.safetensors", "*/diffusion_pytorch_model.safetensors"])
print("  cached CLIP ViT-L-14 (openai) and", sys.argv[1])
PY

step "verify on the H200"
python - "$MODELS" <<'PY'
import os, sys, torch, mmcv, mmdet
from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401  (compiled CUDA op)
from mmdet.apis import init_detector
cfg = os.path.join(os.path.dirname(mmdet.__file__),
                   "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py")
ckpt = os.path.join(sys.argv[1], "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth")
assert torch.cuda.get_device_capability(0) == (9, 0)
init_detector(cfg, ckpt, device="cuda")
print(f"  torch {torch.__version__}, mmcv {mmcv.__version__}, mmdet {mmdet.__version__}: "
      f"Mask2Former loads on {torch.cuda.get_device_name(0)}")
PY

cat > "$BENCH_ROOT/geneval_env.sh" <<EOF
# Generated by scripts/bench/geneval_setup.sbatch
export BENCH_ROOT="$BENCH_ROOT"
export GENEVAL_CODE="$G"
export GENEVAL_MODELS="$MODELS"
export GENEVAL_ENV="$ENV_NAME"
export GENEVAL_COMMIT="$GENEVAL_COMMIT"
export MMDET_COMMIT="$MMDET_COMMIT"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EOF
step "done — next: CALIBRATE=1 sbatch scripts/bench/geneval_score.sbatch"
