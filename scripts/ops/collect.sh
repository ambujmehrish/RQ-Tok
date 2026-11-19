#!/bin/bash
# Results collector for the job queue (called by rqops/queue.py; also usable by hand on
# the login node):
#
#   bash scripts/ops/collect.sh "<message>"
#
# 1. scripts/levelb/collect_results.sh: configs, training/held-out metrics, sampling
#    configs, scorer metrics, calibration record, status table -> results/ (small text
#    files only; never checkpoints, images or latents).
# 2. $OPS_BASE/status.json -> results/ops/status.json (queue + cluster snapshot).
# 3. Every log in logs/ changed in the last 7 days -> results/logs/<name>.txt (a log over
#    10 MB keeps its first 2,000 + last 20,000 lines, with a marker line).
# 4. git add results/ (files < 20 MB), commit "Results: <message>" (skipped if nothing
#    changed), then fetch + rebase onto origin/$BRANCH and push.
#
#   bash scripts/ops/collect.sh --commit-only "<message>"
#      step 4 without fetch/rebase/push (the runner calls this before each update).
set -uo pipefail
cd "$(dirname "$0")/../.."
COMMIT_ONLY=""
if [ "${1:-}" = --commit-only ]; then COMMIT_ONLY=1; shift; fi
MSG=${1:?usage: collect.sh [--commit-only] "<message>"}
BRANCH=${OPS_BRANCH:-claude/great-volta-KaJdU}
# never wait for a typed password: fail at once and say so (a push that prompts for
# credentials hangs until the queue's 900 s timeout and nothing reaches git)
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes}"
OPS_BASE=${OPS_BASE:-$HOME/work/ops}
mkdir -p results/logs results/ops

if [ -z "$COMMIT_ONLY" ]; then
    bash scripts/levelb/collect_results.sh > /dev/null || { echo "collect_results.sh failed"; exit 1; }
    [ -f "$OPS_BASE/status.json" ] && cp "$OPS_BASE/status.json" results/ops/status.json
    find logs -maxdepth 1 -type f -mtime -7 -print0 2>/dev/null | while IFS= read -r -d '' f; do
        name=$(basename "$f"); dst="results/logs/${name%.*}.txt"
        if [ "$(stat -c %s "$f")" -le 10485760 ]; then
            cp "$f" "$dst"
        else
            { head -n 2000 "$f"; echo "[collect] log over 10 MB: lines between the first 2000 and the last 20000 omitted"; tail -n 20000 "$f"; } > "$dst"
        fi
    done
fi

find results -type f -size -20M -print0 | xargs -0 -r git add -f --
if git diff --cached --quiet; then
    echo "collector: nothing changed"; exit 0
fi
git commit -q -m "Results: $MSG" || { echo "commit failed"; exit 1; }
[ -n "$COMMIT_ONLY" ] && { echo "collector: committed locally ($MSG)"; exit 0; }
if git fetch -q origin "$BRANCH" && git rebase -q FETCH_HEAD && git push -q origin "HEAD:$BRANCH"; then
    echo "collector: pushed ($MSG)"
else
    git rebase --abort 2>/dev/null
    echo "collector: push failed (non-interactive git needs stored credentials: SSH key or token,"
    echo "  see ops/README.md); the commit stays local and goes out with the next push"; exit 1
fi
