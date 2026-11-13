#!/bin/bash
# =====================================================================================
# Copy the small result files from the run folders into results/ for git: configs,
# training/held-out metrics, tokenizer summary, sampling configs, scorer metrics and the
# calibration record. No checkpoints, no images, no latents.
#
#   bash scripts/levelb/collect_results.sh
#   git add results && git commit -m "Results: <what changed>" && git push origin claude/great-volta-KaJdU
# =====================================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LEVELB_ROOT="${LEVELB_ROOT:-$HOME/work/levelb}"
BENCH_ROOT="${BENCH_ROOT:-$HOME/work/bench}"
D="$REPO/results"

copy() {   # $1 = source file, $2 = destination dir; skip missing files
  [ -f "$1" ] || return 0
  mkdir -p "$2" && cp "$1" "$2/" && echo "  $1"
}

echo "== Level B runs"
for run in controlnet branch_bifrost branch_adarq branch_adarq_tuned branch_adarq_soft; do
  for f in config.json metrics.jsonl; do copy "$LEVELB_ROOT/$run/$f" "$D/levelb/$run"; done
  copy "$LEVELB_ROOT/$run/diag/overfit_diag.json" "$D/levelb/$run"
done
copy "$LEVELB_ROOT/adarq_tok/tokenizer.json" "$D/levelb/adarq_tok"
copy "$LEVELB_ROOT/imagenet256/meta.json" "$D/levelb/imagenet256"
copy "$LEVELB_ROOT/imagenet256/flux_text_meta.json" "$D/levelb/imagenet256"

echo "== samples (sampling configs and test metrics)"
if [ -d "$LEVELB_ROOT/samples" ]; then
  for s in "$LEVELB_ROOT"/samples/*/; do
    name="$(basename "$s")"
    for f in "$s"shard*.json "$s"*.metrics.json; do copy "$f" "$D/levelb/samples/$name"; done
  done
fi

echo "== ADM evaluator (calibration and diagnostics)"
copy "$BENCH_ROOT/ADM_CALIBRATED" "$D/bench/adm"
for f in "$BENCH_ROOT"/adm/*.metrics.json "$BENCH_ROOT"/adm/SHA256SUMS "$BENCH_ROOT"/adm_diag/adm_diag.json; do
  copy "$f" "$D/bench/adm"
done

echo "== status"
mkdir -p "$D/levelb"
LEVELB_ROOT="$LEVELB_ROOT" python "$REPO/scripts/levelb/status.py" > "$D/levelb/status.txt"
echo "  $D/levelb/status.txt"
du -sh "$D"
