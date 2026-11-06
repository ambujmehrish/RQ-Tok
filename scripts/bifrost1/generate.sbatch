#!/bin/bash
# =====================================================================================
# Bifrost-1 REFERENCE (Level A), step 2: generate benchmark images with the OFFICIAL
# script and the RELEASED checkpoint, unchanged, on one H200.
#
#   sbatch --qos=gpulongjobs --export=ALL,BENCH=geneval  scripts/bifrost1/generate.sbatch
#   sbatch --qos=gpulongjobs --export=ALL,BENCH=dpgbench scripts/bifrost1/generate.sbatch
#
# Runs upstream inference_geneval_dpgbench.py with its own defaults (seed 42, 4 images per
# prompt, 256 MLLM decoding steps, cfg_weight 1.0, FLUX 28 steps / controlnet scale 0.7 /
# guidance 3.5, 512x512 output). Nothing is re-implemented here; only the benchmark is
# selected. Scoring (GenEval detector, DPG-Bench mPLUG VQA) is a separate step.
#
# The upstream script has no resume: a killed job restarts from the first prompt.
# The vhpc01 'gpus' partition caps jobs at 24 h (MaxTime=1-00:00:00), so that is the
# limit here; if a benchmark cannot finish inside it, it must be split, not extended.
# PREREQUISITE: scripts/bifrost1/setup.sbatch has completed.
# =====================================================================================
#SBATCH --job-name=bifrost1-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
# upstream loads FLUX (12B), T5 and the ControlNet in fp32 on the CPU before casting;
# 96G was not enough headroom. vhpc13-15 have 200 GB each, one GPU each.
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }

BENCH="${BENCH:?set BENCH=geneval or BENCH=dpgbench}"
case "$BENCH" in
  geneval)  FLAGS=(--eval_geneval True --eval_dpgbench False); EXPECT=553 ;;
  dpgbench) FLAGS=(--eval_geneval False --eval_dpgbench True); EXPECT=1065 ;;
  *) die "BENCH must be geneval or dpgbench, got '$BENCH'" ;;
esac

BIFROST_ROOT="${BIFROST_ROOT:-$HOME/work/bifrost1}"
[ -f "$BIFROST_ROOT/bifrost1_env.sh" ] || die "run scripts/bifrost1/setup.sbatch first"
# shellcheck disable=SC1091
source "$BIFROST_ROOT/bifrost1_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u   # conda's activate.d/deactivate.d scripts read unset variables; nounset off for conda only
conda activate "$BIFROST_ENV"
set -u
# diffusers 0.33.1 (pinned upstream) calls the Hub API to list the shards of a sharded
# checkpoint even when every file is cached, so FLUX.1-dev cannot load with
# HF_HUB_OFFLINE=1. The compute nodes have internet; the run goes online and the exact
# revisions it resolved are recorded in RUN.json below.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
commit=$(git -C "$BIFROST_CODE" rev-parse HEAD)
grep -q "\"upstream_commit\": \"$commit\"" "$BIFROST_ROOT/PROVENANCE.json" \
  || die "code checkout $commit does not match PROVENANCE.json"

# --- optional sharding (a benchmark that cannot finish inside the 24 h limit) ----------
# Upstream reads its prompts from evaluation/... RELATIVE TO THE WORKING DIRECTORY and
# imports bifrost/ from the script's own directory. A shard therefore runs the unmodified
# script from a work dir holding a contiguous slice of the official prompt files. FLUX is
# re-seeded with the same seed for every prompt, so slicing changes no image.
NSHARDS="${NSHARDS:-1}"; SHARD="${SHARD:-0}"
[ "$SHARD" -ge 0 ] && [ "$SHARD" -lt "$NSHARDS" ] || die "need 0 <= SHARD < NSHARDS"
if [ "$NSHARDS" -gt 1 ]; then
  OUT="${OUT:-$BIFROST_ROOT/outputs/${BENCH}_shard${SHARD}of${NSHARDS}_${SLURM_JOB_ID:-local}}"
else
  OUT="${OUT:-$BIFROST_ROOT/outputs/${BENCH}_${SLURM_JOB_ID:-local}}"
fi
mkdir -p "$OUT"
RUN_DIR="$BIFROST_CODE"
if [ "$NSHARDS" -gt 1 ]; then
  RUN_DIR="$OUT/workdir"
  EXPECT=$(python "$REPO/scripts/bifrost1/shard_prompts.py" "$BENCH" "$BIFROST_CODE" \
             "$RUN_DIR" "$SHARD" "$NSHARDS")
  cp "$RUN_DIR/shard.json" "$OUT/shard.json"
fi
echo "### $BENCH shard $SHARD/$NSHARDS ($EXPECT prompts) on $(hostname) ($gpu), upstream $commit -> $OUT"
date

( cd "$RUN_DIR" && python "$BIFROST_CODE/inference_geneval_dpgbench.py" "${FLAGS[@]}" \
    --output_dir "$OUT" --local_checkpoint_path "$BIFROST_CKPT" )

# --- completeness: the benchmark must be fully covered, or the score is not comparable --
if [ "$BENCH" = "geneval" ]; then
  got=$(find "$OUT/geneval_bifrost" -mindepth 1 -maxdepth 1 -type d | wc -l)
  imgs=$(find "$OUT/geneval_bifrost" -path '*/samples/*.png' | wc -l)
  [ "$got" -eq "$EXPECT" ] || die "GenEval: $got prompt folders, expected $EXPECT"
  [ "$imgs" -eq $((EXPECT * 4)) ] || die "GenEval: $imgs images, expected $((EXPECT * 4))"
else
  got=$(find "$OUT/dpgbench_bifrost" -maxdepth 1 -name '*.png' | wc -l)
  [ "$got" -eq "$EXPECT" ] || die "DPG-Bench: $got grids, expected $EXPECT"
fi
python - "$OUT/RUN.json" "$BENCH" "$commit" "$gpu" <<'PY'
import json, sys, socket, time
from huggingface_hub import HfApi
out, bench, commit, gpu = sys.argv[1:]
api = HfApi()
revisions = {r: api.model_info(r).sha for r in
             ("black-forest-labs/FLUX.1-dev", "Qwen/Qwen2.5-VL-7B-Instruct")}
json.dump({"bench": bench, "upstream_commit": commit, "gpu": gpu,
           "backbone_revisions": revisions,
           "host": socket.gethostname(), "finished": time.strftime("%F %T")},
          open(out, "w"), indent=2)
PY
date
echo "=== done: $BENCH complete in $OUT (score it next) ==="
