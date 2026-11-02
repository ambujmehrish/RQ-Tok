#!/usr/bin/env bash
# =====================================================================================
# Fetch every remote asset AdaRQ-Flow needs, once, so later jobs run fully offline.
#
#   source ~/miniforge3/etc/profile.d/conda.sh && conda activate adarq
#   ADARQ_CACHE=$HOME/work/adarq-cache bash scripts/prefetch_assets.sh            # open models
#   ADARQ_CACHE=$HOME/work/adarq-cache bash scripts/prefetch_assets.sh --gated    # + FLUX.1-dev
#
# Run it wherever there is internet: a login node, or a compute node if the setup
# survey showed egress there. It installs nothing and runs no training.
#
# Downloads (all into a Hugging Face cache, plus $ADARQ_CACHE for data):
#   * openai/clip-vit-base-patch32         real CLIP vision tower        (~0.6 GB)
#   * Qwen/Qwen2.5-VL-{3B,7B}-Instruct     real MLLM, NOT gated          (~7.5 + ~16.6 GB)
#   * COCO val2017 captions                official annotations          (~4 MB kept)
#   * COCO val2017 images                  DISTINCT captioned photos     (~160 KB each)
#   * --gated: black-forest-labs/FLUX.1-dev  diffusers subfolders only   (~33 GB)
#     FLUX.1-dev IS gated: accept the licence on huggingface.co and `hf auth login`.
#
# Afterwards every model is re-opened with HF_HUB_OFFLINE=1 to prove the cache is
# complete, and the geometry configs must match is printed.
# =====================================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_IMAGES="${NUM_IMAGES:-64}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"
QWEN_MODELS="${QWEN_MODELS:-Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2.5-VL-7B-Instruct}"
FLUX_MODEL="${FLUX_MODEL:-black-forest-labs/FLUX.1-dev}"
WANT_GATED=0
[ "${1:-}" = "--gated" ] && WANT_GATED=1

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }
die()  { printf '\n\033[31mFAILED: %s\033[0m\n' "$1" >&2; exit 1; }

# -- 0. the env must be the project env (transformers/diffusers/adarq_flow) -----------
python3 -c "import transformers, huggingface_hub, adarq_flow" 2>/dev/null || die \
  "run inside the project env (conda activate adarq; see scripts/vhpc01_env.sbatch)"
[ "$WANT_GATED" = "1" ] && { python3 -c "import diffusers" 2>/dev/null || die \
  "diffusers missing: pip install -e '.[models]'"; }

# -- 1. cache location: explicit, and with enough room ---------------------------------
# Checked on FREE SPACE, not on the path: on vhpc01 $HOME is a 26T NFS volume with ~11T
# free while /scratch is 89% full, so a path rule would pick the wrong volume.
MIN_FREE_GB="${MIN_FREE_GB:-80}"
if [ -z "${ADARQ_CACHE:-}" ]; then
  cat >&2 <<'EOF'
ADARQ_CACHE is not set. Point it at a volume with room, e.g.

  export ADARQ_CACHE=$HOME/work/adarq-cache     # vhpc01: /home is 26T, ~11T free

Refusing to guess a location: a wrong default silently fills someone's volume.
Check candidates with:  df -h $HOME /scratch /tmp
EOF
  exit 1
fi
mkdir -p "$ADARQ_CACHE"
check_free() {   # $1 = path
  local free
  free=$(df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -n "$free" ] || die "could not determine free space on $1"
  [ "$free" -ge "$MIN_FREE_GB" ] || die "$1 has only ${free}G free; need >= ${MIN_FREE_GB}G"
  echo "  $1: ${free}G free (need >= ${MIN_FREE_GB}G)"
}

# -- 2. Hugging Face cache: KEEP an existing one ---------------------------------------
# An earlier version always set HF_HOME=$ADARQ_CACHE/hf. That hid the token `hf auth
# login` had stored under the user's own HF_HOME (-> 401 on FLUX) and re-downloaded
# anything already cached there. Now an existing HF_HOME is reused as is.
if [ -n "${HF_HOME:-}" ]; then
  echo "reusing existing HF_HOME=$HF_HOME"
else
  export HF_HOME="$ADARQ_CACHE/hf"
  echo "HF_HOME unset; using $HF_HOME"
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=0
mkdir -p "$HF_HUB_CACHE"
step "free space"
check_free "$ADARQ_CACHE"
check_free "$HF_HUB_CACHE"

IMAGES="$ADARQ_CACHE/coco_val2017"
CAPTIONS="$ADARQ_CACHE/annotations/captions_val2017.json"

# -- 3. captions FIRST: they decide which distinct images to fetch ---------------------
step "real COCO captions -> $CAPTIONS"
if [ ! -f "$CAPTIONS" ]; then
  ZIPDIR="$ADARQ_CACHE/_coco_ann_download"      # own dir: untrusted archive
  mkdir -p "$ZIPDIR"
  curl -sS -L --fail -o "$ZIPDIR/ann.zip" \
      "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
      || die "could not download COCO annotations (egress to images.cocodataset.org?)"
  python3 -I "$REPO/scripts/extract_coco_captions.py" \
      --zip "$ZIPDIR/ann.zip" --out-dir "$ADARQ_CACHE/annotations"
  rm -rf "$ZIPDIR"
fi
python3 -I "$REPO/scripts/extract_coco_captions.py" --verify "$CAPTIONS"

# -- 4. CLIP + DISTINCT real images ----------------------------------------------------
step "real CLIP ($CLIP_MODEL) + $NUM_IMAGES distinct COCO images -> $IMAGES"
if [ -d "$IMAGES" ]; then
  # refuse a cache left by the old repeating downloader (one image, several names)
  n_files=$(find "$IMAGES" -maxdepth 1 -name '*.jpg' | wc -l)
  n_ids=$(find "$IMAGES" -maxdepth 1 -name '*.jpg' -printf '%f\n' | cut -d_ -f1 \
          | sed 's/\.jpg$//' | sort -u | wc -l)
  [ "$n_files" -eq "$n_ids" ] || die "$IMAGES has $n_files files but only $n_ids distinct \
images (written by the old repeating downloader). Delete it and re-run."
fi
python3 "$REPO/scripts/extract_clip_latents.py" \
    --out "$ADARQ_CACHE/prefetch_probe.pt" \
    --num-images "$NUM_IMAGES" \
    --model "$CLIP_MODEL" \
    --annotations "$CAPTIONS" \
    --save-images "$IMAGES"
rm -f "$ADARQ_CACHE"/prefetch_probe.pt*
echo "  $(find "$IMAGES" -maxdepth 1 -name '*.jpg' | wc -l) distinct images cached"

# -- 5. Qwen2.5-VL (open weights, no login needed) -------------------------------------
step "Qwen2.5-VL: $QWEN_MODELS"
python3 - $QWEN_MODELS <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    print(f"  {repo} ...", flush=True)
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    print(f"  cached {repo} -> {path}", flush=True)
PY

# -- 6. FLUX.1-dev (gated) -------------------------------------------------------------
if [ "$WANT_GATED" = "1" ]; then
  step "FLUX.1-dev (gated: needs accepted licence + hf auth login)"
  python3 - "$FLUX_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

repo = sys.argv[1]
# Only the diffusers-format subfolders the loader reads. The repo root also holds a
# standalone flux1-dev.safetensors (~24 GB) + ae.safetensors that diffusers never uses.
patterns = ["model_index.json", "scheduler/*", "transformer/*", "vae/*",
            "text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*"]
try:
    path = snapshot_download(repo, allow_patterns=patterns)
except GatedRepoError as e:
    sys.exit(f"{repo} is gated and this account has no access yet.\n"
             f"  1. accept the licence at https://huggingface.co/{repo}\n"
             f"  2. hf auth login   (token is read from $HF_HOME/token)\n{e}")
print(f"  cached {repo} -> {path}")
PY
else
  step "skipping FLUX.1-dev (re-run with --gated after accepting its licence)"
fi

# -- 7. prove the cache is complete: re-open everything OFFLINE ------------------------
step "offline verification (HF_HUB_OFFLINE=1)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 - "$CLIP_MODEL" "$WANT_GATED" \
    "$FLUX_MODEL" $QWEN_MODELS <<'PY'
import sys
from transformers import AutoConfig
from adarq_flow.mllm.qwen import qwen_dims

clip, gated, flux, *qwens = sys.argv[1:]
AutoConfig.from_pretrained(clip)
print(f"  {clip}: config OK")
for q in qwens:
    d = qwen_dims(q)
    print(f"  {q}: mllm.hidden_dim={d['text_hidden']} tokenizer.clip_dim={d['visual_out']}"
          f" (tower-internal {d['vision_hidden']}, not the bridge latent)")
if gated == "1":
    from diffusers import FluxTransformer2DModel
    from adarq_flow.renderer.flux import flux_vae_dims
    v = flux_vae_dims(flux)
    t = FluxTransformer2DModel.load_config(flux, subfolder="transformer")
    print(f"  {flux}: renderer.latent_dim={int(v['packed_dim'])} "
          f"(transformer in_channels={int(t['in_channels'])}; must match) | "
          f"double/single blocks={t['num_layers']}/{t['num_single_layers']}")
    if int(v["packed_dim"]) != int(t["in_channels"]):
        sys.exit("FLUX VAE packed width != transformer in_channels — do not run")
PY

# -- 8. the env file offline jobs source -----------------------------------------------
cat > "$ADARQ_CACHE/adarq_env.sh" <<EOF
# Generated by scripts/prefetch_assets.sh — source this in OFFLINE jobs.
export ADARQ_CACHE="$ADARQ_CACHE"
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HUB_CACHE"
export ADARQ_IMAGES="$IMAGES"
export ADARQ_CAPTIONS="$CAPTIONS"
# Fail loudly instead of hanging on a network call that cannot succeed:
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EOF

step "done"
cat <<EOF
Cache:      $ADARQ_CACHE      (models in $HF_HUB_CACHE)
In jobs:    source $ADARQ_CACHE/adarq_env.sh
$( [ "$WANT_GATED" = "1" ] || echo "FLUX.1-dev: not fetched yet — re-run with --gated." )
EOF
