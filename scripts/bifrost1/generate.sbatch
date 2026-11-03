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
# PREREQUISITE: scripts/bifrost1/setup.sbatch has completed.
# =====================================================================================
#SBATCH --job-name=bifrost1-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
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
conda activate "$BIFROST_ENV"
[ "${HF_HUB_OFFLINE:-0}" = "1" ] || die "bifrost1_env.sh must set HF_HUB_OFFLINE=1"

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
[[ "$gpu" == *H200* ]] || die "expected an H200, got '${gpu:-no GPU}'"
commit=$(git -C "$BIFROST_CODE" rev-parse HEAD)
grep -q "\"upstream_commit\": \"$commit\"" "$BIFROST_ROOT/PROVENANCE.json" \
  || die "code checkout $commit does not match PROVENANCE.json"

OUT="${OUT:-$BIFROST_ROOT/outputs/${BENCH}_${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT"
echo "### $BENCH on $(hostname) ($gpu), upstream $commit -> $OUT"
date

# upstream reads prompts via paths relative to its repo root
( cd "$BIFROST_CODE" && python inference_geneval_dpgbench.py "${FLAGS[@]}" \
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
out, bench, commit, gpu = sys.argv[1:]
json.dump({"bench": bench, "upstream_commit": commit, "gpu": gpu,
           "host": socket.gethostname(), "finished": time.strftime("%F %T")},
          open(out, "w"), indent=2)
PY
date
echo "=== done: $BENCH complete in $OUT (score it next) ==="
