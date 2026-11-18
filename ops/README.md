# Git-driven job queue (vhpc01)

Adapted from the FaCT project's queue. Claude works in a sandbox without cluster access:
it submits Slurm jobs on vhpc01 by committing request files and reads results back
through git. A runner on the vhpc01 login node, in tmux, does the cluster side.

```
Claude: ops/queue/<id>.json --commit/push--> branch --fetch/rebase--> runner (login node)
                                                                       | sbatch / sacct / squeue
Claude <-- results/ (+ results/ops/status.json) <-- scripts/ops/collect.sh <-'
```

- **Branch:** `claude/great-volta-KaJdU` (override with `OPS_BRANCH`).
- **QOS:** `gpulongjobs` unless a request sets `sbatch.qos`. vhpc01 has no accounts.
- **Scripts:** `scripts/levelb/*.sbatch` and `scripts/bench/*.sbatch`, run from the repo.
  They choose their own partition (`gpus` for H200 jobs, `vhpc` for CPU scoring).

## Start / stop the runner

First time only — the checkout must be on the branch with no local edits, and able to push:
```bash
cd ~/projects/RQ-Tok
git status --short | grep -v '^??'      # must print nothing
git checkout claude/great-volta-KaJdU && git pull origin claude/great-volta-KaJdU
git push --dry-run origin HEAD:claude/great-volta-KaJdU   # must not ask for a password
```
Start:
```bash
tmux new -s rqtok_runner
cd ~/projects/RQ-Tok && bash scripts/ops/runner.sh      # Ctrl-b d to detach
```
- **One runner only.** A lock on `$OPS_BASE/state.json.lock` makes a second one idle.
- **Look:** `tmux attach -t rqtok_runner`, or `tail -f logs/ops_runner.log`.
- **Stop:** attach, then Ctrl-c, or `tmux kill-session -t rqtok_runner`.
- **After a login-node restart** the session is gone (jobs keep running); start it again.
- `INTERVAL=300` seconds by default. State: `$OPS_BASE/state.json` (`~/work/ops`), outside
  the repo, so a restart never re-submits anything.

Each loop: commit any local `results/` change, `git fetch` + `git rebase` (never a merge;
on failure "git update failed" in the log), then one tick. Keep the checkout free of other
local edits.

## Where things live

| what | where |
|---|---|
| requests | `ops/queue/<id>.json` (committed by Claude) |
| cancels | `ops/cancel/<id>.json` |
| runner state | `$OPS_BASE/state.json` (outside the repo) |
| queue + cluster status | `$OPS_BASE/status.json` -> pushed as `results/ops/status.json` |
| runner log | `logs/ops_runner.log` -> `results/logs/ops_runner.txt` |
| job logs | `logs/*.out` changed in the last 7 days -> `results/logs/<name>.txt` |
| results | `scripts/levelb/collect_results.sh` -> `results/` (configs, metrics, scores, status; never checkpoints or images) |

A heartbeat push about once an hour sends the status even when no job has finished.

## Request format

One request per file, `ops/queue/<id>.json`, `id` equal to the file stem.

Submit:
```json
{"id": "2026-11-18-flux1024-p0", "what": "plain FLUX at 1024 px, part 0 of 4",
 "script": "scripts/levelb/sample.sbatch",
 "env": {"ARM": "flux", "SEED": "0", "RES": "1024", "BATCH": "4", "TAG": "flux1024",
         "SHARD": "0", "NSHARDS": "4"},
 "sbatch": {"time": "24:00:00"},
 "after": ["<request id>"], "after_mode": "afterok",
 "collect": true}
```
Watch-only (jobs submitted by hand): `{"id", "what", "after_jobs": ["1507923"], "collect": true}`.

Rules (the runner refuses anything else, naming the field):
- **Keys:** `id what script env sbatch after after_mode after_jobs post collect`.
- **`script`:** `^scripts/(levelb|bench)/[a-z0-9_]+\.sbatch$` and must exist.
- **`env`:** names `^[A-Z][A-Z0-9_]{0,40}$` (PATH, HOME, PYTHON*, SLURM_*, LD_* ... denied);
  values `^[A-Za-z0-9_./:+=, %@{}$-]{0,400}$`; `$` only as `${NAME}`/`$NAME` with NAME in
  LEVELB_ROOT, BENCH_ROOT, BIFROST_ROOT, OPS_BASE (expanded by the runner). Passed through
  the process environment, never a shell.
- **`sbatch`:** only `time`, `qos`, `array`.
- **`after`:** request ids -> `--dependency=<after_mode>:<job ids>`.
- **`post.module`:** `^levelb\.[a-z0-9_]+$`, run as `python3 -m` when every job is terminal.
- **`collect`:** push results when the request finishes.

Cancel: `ops/cancel/<id>.json` with `id`, `what`, `requests` (request ids; never raw job
ids, so jobs from elsewhere are out of reach). Each cancel acts once.

## How Claude uses it

1. Write `ops/queue/<YYYY-MM-DD>-<what>.json`.
2. `python3 -m rqops.queue validate --repo .`
3. Commit and push. The runner submits within one interval.
4. Read `results/ops/status.json` and `results/` after `git fetch`; no pasted terminal output.
