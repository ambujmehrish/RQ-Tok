#!/usr/bin/env bash
# =====================================================================================
# Prefetch every remote asset to a shared cache OUTSIDE $HOME.  RUN ON A LOGIN NODE.
#
#   ADARQ_CACHE=$WORK/adarq-cache bash scripts/prefetch_assets.sh
#
# Cineca Leonardo (and most HPC sites): login nodes have internet, COMPUTE NODES DO NOT,
# and $HOME has a small quota that model weights will blow through. So every download
# happens here, once, into $ADARQ_CACHE — then compute nodes run fully offline against
# that directory.
#
# Downloads:
#   * openai/clip-vit-base-patch32   real CLIP vision tower (~600 MB)
#   * COCO val2017 sample            real photographs, saved as .jpg
#   * (optional) --gated             Qwen2.5-VL / FLUX.1-dev, needs `hf auth login`
#
# On the compute node, source the generated env file and everything resolves locally:
#   source $ADARQ_CACHE/adarq_env.sh      # sets HF_HOME + HF_HUB_OFFLINE=1
# =====================================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_IMAGES="${NUM_IMAGES:-64}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"
WANT_GATED=0
[ "${1:-}" = "--gated" ] && WANT_GATED=1

# -- 1. cache location: explicit, and with enough room ---------------------------------
# The check is on FREE SPACE, not on the path. An earlier version blacklisted $HOME,
# which is wrong on clusters (vhpc01 among them) where $HOME is a large shared NFS
# volume and the small/full filesystem is /scratch. What actually breaks a run is
# running out of room mid-download, so that is what we measure.
MIN_FREE_GB="${MIN_FREE_GB:-80}"     # CLIP + Qwen-VL-7B + FLUX.1-dev + images, bf16
if [ -z "${ADARQ_CACHE:-}" ]; then
  cat >&2 <<'EOF'
ADARQ_CACHE is not set. Point it at a volume with room for model weights, e.g.

  export ADARQ_CACHE=$HOME/work/adarq-cache     # vhpc01: /home is 26T, ~11T free
  bash scripts/prefetch_assets.sh

Refusing to guess a location: a wrong default silently fills someone's volume.
Check candidates with:  df -h $HOME /scratch /tmp
EOF
  exit 1
fi
mkdir -p "$ADARQ_CACHE"
FREE_GB=$(df -BG --output=avail "$ADARQ_CACHE" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -z "$FREE_GB" ]; then
  echo "could not determine free space on $ADARQ_CACHE; aborting rather than guessing" >&2
  exit 1
fi
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "ADARQ_CACHE ($ADARQ_CACHE) has only ${FREE_GB}G free; need >= ${MIN_FREE_GB}G." >&2
  echo "Pick a roomier volume (df -h \$HOME /scratch) or raise MIN_FREE_GB if you know" >&2
  echo "the gated weights are already cached elsewhere. Aborting." >&2
  exit 1
fi
echo "cache has ${FREE_GB}G free (need >= ${MIN_FREE_GB}G)"

mkdir -p "$ADARQ_CACHE"
export HF_HOME="$ADARQ_CACHE/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"   # legacy readers
export HF_HUB_OFFLINE=0                              # login node: downloads allowed
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
IMAGES="$ADARQ_CACHE/coco_val2017"

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }
step "cache: $ADARQ_CACHE  (HF_HOME=$HF_HOME)"
df -h "$ADARQ_CACHE" | tail -1

# -- 2. real CLIP vision tower ---------------------------------------------------------
step "downloading real CLIP: $CLIP_MODEL"
python3 - "$CLIP_MODEL" <<'PY'
import sys
from transformers import CLIPImageProcessor, CLIPVisionModel
mid = sys.argv[1]
CLIPImageProcessor.from_pretrained(mid)
m = CLIPVisionModel.from_pretrained(mid)
print(f"  cached {mid}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
PY

# -- 3. real images --------------------------------------------------------------------
step "downloading $NUM_IMAGES real COCO val2017 images -> $IMAGES"
python3 "$REPO/scripts/extract_clip_latents.py" \
    --out "$ADARQ_CACHE/prefetch_probe.pt" \
    --num-images "$NUM_IMAGES" \
    --model "$CLIP_MODEL" \
    --save-images "$IMAGES"
rm -f "$ADARQ_CACHE/prefetch_probe.pt" "$ADARQ_CACHE/prefetch_probe.pt.meta.json"
echo "  $(find "$IMAGES" -name '*.jpg' | wc -l) images cached"

# -- 3b. real COCO captions (the real dataset's text half) -----------------------------
step "downloading real COCO captions -> $ADARQ_CACHE/annotations"
CAPTIONS="$ADARQ_CACHE/annotations/captions_val2017.json"
if [ ! -f "$CAPTIONS" ]; then
  ZIPDIR="$ADARQ_CACHE/_coco_ann_download"      # own dir: untrusted archive
  mkdir -p "$ZIPDIR"
  curl -sS -L -o "$ZIPDIR/ann.zip" \
      "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
  python3 -I "$REPO/scripts/extract_coco_captions.py" \
      --zip "$ZIPDIR/ann.zip" --out-dir "$ADARQ_CACHE/annotations"
  rm -rf "$ZIPDIR"
fi
python3 -I "$REPO/scripts/extract_coco_captions.py" --verify "$CAPTIONS"

# -- 4. gated weights (opt-in) ---------------------------------------------------------
if [ "$WANT_GATED" = "1" ]; then
  step "downloading GATED weights (requires prior 'hf auth login' + accepted licences)"
  python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen2.5-VL-7B-Instruct", "black-forest-labs/FLUX.1-dev"):
    print(f"  {repo} ...", flush=True)
    snapshot_download(repo)          # raises on missing auth / unaccepted licence
    print(f"  cached {repo}")
PY
else
  step "skipping gated weights (pass --gated once Qwen2.5-VL / FLUX.1-dev are wired)"
fi

# -- 5. the offline env file compute nodes source --------------------------------------
cat > "$ADARQ_CACHE/adarq_env.sh" <<EOF
# Generated by scripts/prefetch_assets.sh — source this on OFFLINE compute nodes.
export ADARQ_CACHE="$ADARQ_CACHE"
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export ADARQ_IMAGES="$IMAGES"
export ADARQ_CAPTIONS="$CAPTIONS"
# Fail loudly instead of hanging on a network call that cannot succeed:
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EOF

step "done"
cat <<EOF
Compute-node usage (no internet required):

  source $ADARQ_CACHE/adarq_env.sh
  bash scripts/smoke_test.sh          # picks up \$ADARQ_IMAGES / \$ADARQ_CAPTIONS

Nothing was written under \$HOME.
EOF
