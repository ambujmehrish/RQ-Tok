#!/bin/bash
# Git-driven job queue runner — vhpc01 LOGIN node, inside tmux (see ops/README.md):
#
#   tmux new -s rqtok_runner
#   cd ~/projects/RQ-Tok && bash scripts/ops/runner.sh       # Ctrl-b d to detach
#
# Every INTERVAL seconds: commit local results/ changes, fast-forward the checkout to
# origin/$BRANCH by rebase (never merge), then one queue tick (rqops/queue.py): submit new
# requests from ops/queue/, poll jobs, run post steps, push results and
# results/ops/status.json through scripts/ops/collect.sh. The state lives outside the repo
# ($OPS_BASE/state.json), so a restart never re-submits anything.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
BRANCH=${OPS_BRANCH:-claude/great-volta-KaJdU}
# never wait for a typed password: fail at once and say so (a push that prompts for
# credentials hangs until the queue's 900 s timeout and nothing reaches git)
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes}"
INTERVAL=${INTERVAL:-300}
export OPS_PYTHON=${OPS_PYTHON:-python3}
export OPS_BRANCH=$BRANCH
export OPS_BASE=${OPS_BASE:-$HOME/work/ops}
export LEVELB_ROOT=${LEVELB_ROOT:-$HOME/work/levelb}
export BENCH_ROOT=${BENCH_ROOT:-$HOME/work/bench}
export BIFROST_ROOT=${BIFROST_ROOT:-$HOME/work/bifrost1}
STATE=$OPS_BASE/state.json
LOG=$REPO/logs/ops_runner.log
mkdir -p "$OPS_BASE" "$REPO/logs"

cur=$(git rev-parse --abbrev-ref HEAD)
if [ "$cur" != "$BRANCH" ]; then
    echo "checkout is on '$cur', not '$BRANCH' - see ops/README.md (first-time setup)"; exit 1
fi
if ! "$OPS_PYTHON" -c "import sys; assert sys.version_info >= (3, 9)" 2>/dev/null; then
    echo "$OPS_PYTHON is not Python >= 3.9; set OPS_PYTHON"; exit 1
fi
echo "runner: repo=$REPO branch=$BRANCH every ${INTERVAL}s state=$STATE log=$LOG"

while true; do
    # results/ is written by the collector only; commit any leftover so the rebase can run
    if [ -n "$(git status --porcelain --untracked-files=no -- results)" ]; then
        bash scripts/ops/collect.sh --commit-only "local results (runner)" >> "$LOG" 2>&1
    fi
    if git fetch -q origin "$BRANCH" && git rebase -q FETCH_HEAD; then
        "$OPS_PYTHON" -m rqops.queue tick --repo "$REPO" --state "$STATE" >> "$LOG" 2>&1 \
            || echo "$(date -Is) tick exited with an error" >> "$LOG"
    else
        git rebase --abort 2>/dev/null
        echo "$(date -Is) git update failed (local edits outside results/?)" >> "$LOG"
    fi
    tail -n 3 "$LOG"
    sleep "$INTERVAL"
done
