"""Git-driven Slurm job queue for vhpc01 (adapted from the FaCT project's fact/ops/queue.py).

Claude (in a sandbox without cluster access) commits one request per file to
ops/queue/<id>.json; a runner on the vhpc01 login node (scripts/ops/runner.sh, in tmux)
pulls the repo every few minutes and calls tick(), which submits each request at most
once, polls its jobs, runs an optional post-processing module when they finish, and
pushes results and a status file (results/ops/status.json) back through git
(scripts/ops/collect.sh).

    python3 -m rqops.queue tick --repo <repo> --state <state.json>
    python3 -m rqops.queue validate --repo <repo>

Request forms (see ops/README.md):
  submit:     {"id", "what", "script": "scripts/(levelb|bench)/<name>.sbatch", "env": {...},
               "sbatch": {"time", "qos", "array"}, "after": [<request ids>],
               "after_mode": "afterok" (default) | "afterany",
               "post": {"module", "args"}, "collect": bool}
  watch-only: {"id", "what", "after_jobs": ["<job id>", ...], "post", "collect"}

The state file lives OUTSIDE the repo ($OPS_BASE/state.json), so a restarted runner
never submits a request twice. Standard library only: runs with the login node's python3.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_QOS = "gpulongjobs"  # vhpc01 has no accounts; QOS is the only knob
QUEUE_DIR = "ops/queue"
CANCEL_DIR = "ops/cancel"  # {"id", "what", "requests": [request ids]} -> scancel
# (no hold/release: vhpc01 refuses `scontrol hold` for users)
COLLECTOR = ["bash", "scripts/ops/collect.sh"]
MAX_NEW = 40  # sbatch calls per tick
POST_TIMEOUT = 30 * 60
HEARTBEAT_EVERY = 3600
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}

# Placeholders a request may use; expanded from the runner's environment.
PLACEHOLDERS = ("LEVELB_ROOT", "BENCH_ROOT", "BIFROST_ROOT", "OPS_BASE")
_PH = "|".join(PLACEHOLDERS)
PLACEHOLDER_RE = re.compile(r"\$\{(" + _PH + r")\}|\$(" + _PH + r")(?![A-Za-z0-9_])")

ALLOWED_KEYS = {
    "id",
    "what",
    "script",
    "env",
    "sbatch",
    "after",
    "after_mode",
    "after_jobs",
    "post",
    "collect",
}
AFTER_MODES = ("afterok", "afterany")  # afterany: training chains that resume after TIMEOUT
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
SCRIPT_RE = re.compile(r"^scripts/(levelb|bench)/[a-z0-9_]+\.sbatch$")
ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,40}$")
ENV_DENIED = {
    "PATH",
    "HOME",
    "SHELL",
    "BASH_ENV",
    "ENV",
    "IFS",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PROMPT_COMMAND",
    "SBATCH_ACCOUNT",
    "SLURM_CONF",
    "USER",
    "LOGNAME",
}
ENV_DENIED_PREFIXES = ("LD_", "DYLD_", "BASH_FUNC", "SLURM_", "PYTHON")
VALUE_RE = re.compile(r"^[A-Za-z0-9_./:+=, %@{}$-]{0,400}$")
SBATCH_RULES = {
    "time": re.compile(r"^\d{1,2}:\d{2}:\d{2}$"),
    "qos": re.compile(r"^[a-z_]{1,40}$"),
    "array": re.compile(r"^[0-9,\-]{1,40}(%[0-9]{1,3})?$"),
}
JOB_ID_RE = re.compile(r"^\d{1,12}$")
POST_MODULE_RE = re.compile(r"^levelb\.[a-z0-9_]+$")


# --------------------------------------------------------------------- running


@dataclass
class Result:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def default_run(
    cmd: list[str], env: dict | None = None, cwd: str | None = None, timeout: float | None = 300
) -> Result:
    """Run cmd without a shell; never raises for a non-zero exit."""
    p = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return Result(p.returncode, p.stdout, p.stderr)


Run = Callable[..., Result]


def _call(run: Run, cmd, env, cwd, timeout=None) -> Result:
    """run() that turns exceptions (missing binary, timeout) into a failed Result."""
    try:
        return run(cmd, env=env, cwd=cwd, timeout=timeout)
    except Exception as e:  # noqa: BLE001 - the tick must never crash on a command
        return Result(-1, "", f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ validation


def check_value(field: str, value) -> str | None:
    """Error message for an env value / post arg, or None if allowed."""
    if not isinstance(value, str):
        return f"{field}: must be a string"
    if not VALUE_RE.match(value):
        return (
            f"{field}: only letters, digits, space and _./:+=,%@-{{}}$ are allowed "
            f"(max 400 chars; no ; | & ` < > quotes backslash newline)"
        )
    rest = PLACEHOLDER_RE.sub("", value)
    if any(c in rest for c in "${}"):
        return (
            f"{field}: '$', '{{' and '}}' may only appear as ${{NAME}} or $NAME with NAME in "
            f"{', '.join(PLACEHOLDERS)}"
        )
    return None


def expand(value: str, environ) -> str:
    """Replace the allowed placeholders from the runner's environment."""

    def sub(m):
        name = m.group(1) or m.group(2)
        v = environ.get(name)
        if not v:
            raise ValueError(f"${name} is not set in the runner environment")
        return v

    return PLACEHOLDER_RE.sub(sub, value)


def validate(req, stem: str, repo: Path, known_ids: set | None = None) -> list[str]:
    """All rule violations of one request (empty list = valid)."""
    if not isinstance(req, dict):
        return ["request: must be a JSON object"]
    errs = []
    unknown = sorted(set(req) - ALLOWED_KEYS)
    if unknown:
        errs.append(f"keys: unknown {unknown}; allowed {sorted(ALLOWED_KEYS)}")
    rid = req.get("id")
    if not isinstance(rid, str) or not ID_RE.match(rid):
        errs.append("id: must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
    elif rid != stem:
        errs.append(f"id: must equal the file stem {stem!r}")
    what = req.get("what")
    if not isinstance(what, str) or len(what) > 200:
        errs.append("what: required string of at most 200 chars")

    if "script" in req:
        s = req["script"]
        if not isinstance(s, str) or not SCRIPT_RE.match(s):
            errs.append(r"script: must match ^scripts/(levelb|bench)/[a-z0-9_]+\.sbatch$")
        elif not (repo / s).is_file():
            errs.append(f"script: {s} does not exist")
        if "after_jobs" in req:
            errs.append("after_jobs: only for watch-only requests (no script)")
    else:
        if not req.get("after_jobs"):
            errs.append("script: required unless the request is watch-only (after_jobs)")
        for k in ("env", "sbatch", "after", "after_mode"):
            if k in req:
                errs.append(f"{k}: not allowed in a watch-only request (no script)")

    env = req.get("env", {})
    if not isinstance(env, dict):
        errs.append("env: must be an object")
    else:
        for name, value in env.items():
            if not isinstance(name, str) or not ENV_NAME_RE.match(name):
                errs.append(f"env: name {name!r} must match ^[A-Z][A-Z0-9_]{{0,40}}$")
            elif name in ENV_DENIED or name.startswith(ENV_DENIED_PREFIXES):
                errs.append(f"env: name {name!r} is denied")
            e = check_value(f"env.{name}", value)
            if e:
                errs.append(e)

    sb = req.get("sbatch", {})
    if not isinstance(sb, dict):
        errs.append("sbatch: must be an object")
    else:
        for k, v in sb.items():
            rule = SBATCH_RULES.get(k)
            if rule is None:
                errs.append(f"sbatch: key {k!r} not allowed; only {sorted(SBATCH_RULES)}")
            elif not isinstance(v, str) or not rule.match(v):
                errs.append(f"sbatch.{k}: must match {rule.pattern}")

    after = req.get("after", [])
    if not isinstance(after, list):
        errs.append("after: must be a list of request ids")
    else:
        for a in after:
            if not isinstance(a, str) or not ID_RE.match(a):
                errs.append(f"after: {a!r} is not a request id")
            elif a == rid:
                errs.append("after: a request cannot depend on itself")
            elif known_ids is not None and a not in known_ids:
                errs.append(f"after: unknown request id {a!r}")

    if "after_mode" in req:
        if req["after_mode"] not in AFTER_MODES:
            errs.append(f"after_mode: must be one of {list(AFTER_MODES)}")
        elif not req.get("after"):
            errs.append("after_mode: only together with a non-empty after")

    if "after_jobs" in req:
        aj = req["after_jobs"]
        if (
            not isinstance(aj, list)
            or not aj
            or not all(isinstance(j, str) and JOB_ID_RE.match(j) for j in aj)
        ):
            errs.append(r"after_jobs: non-empty list of job ids matching ^\d{1,12}$")

    if "post" in req:
        post = req["post"]
        if not isinstance(post, dict) or set(post) - {"module", "args"}:
            errs.append("post: object with keys module and args only")
        else:
            mod = post.get("module")
            if not isinstance(mod, str) or not POST_MODULE_RE.match(mod):
                errs.append(f"post.module: must match {POST_MODULE_RE.pattern}")
            args = post.get("args", [])
            if not isinstance(args, list):
                errs.append("post.args: must be a list of strings")
            else:
                for i, a in enumerate(args):
                    e = check_value(f"post.args[{i}]", a)
                    if e:
                        errs.append(e)

    if "collect" in req and not isinstance(req["collect"], bool):
        errs.append("collect: must be true or false")
    return errs


def load_requests(repo: Path) -> tuple[dict, dict]:
    """(valid requests by id, invalid file stems -> reason)."""
    raw, invalid = {}, {}
    for f in sorted((repo / QUEUE_DIR).glob("*.json")):
        try:
            raw[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            invalid[f.stem] = f"file: not valid JSON ({str(e)[:200]})"
    known = set(raw) | set(invalid)
    valid = {}
    for stem, req in raw.items():
        errs = validate(req, stem, repo, known)
        if errs:
            invalid[stem] = "; ".join(errs)
        else:
            valid[stem] = req
    return valid, invalid


def load_cancels(repo: Path, known_ids: set) -> tuple[dict, dict]:
    """(valid cancel requests by id, invalid stems -> reason). A cancel names
    queue request ids only, never raw job ids, so it can only reach jobs this
    queue submitted (other projects' jobs are out of reach)."""
    return _load_controls(repo / CANCEL_DIR, known_ids, actions=None)


def _load_controls(directory: Path, known_ids: set, actions) -> tuple[dict, dict]:
    keys = {"id", "what", "requests"} | ({"action"} if actions else set())
    valid, invalid = {}, {}
    for f in sorted(directory.glob("*.json")):
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            invalid[f.stem] = f"file: not valid JSON ({str(e)[:200]})"
            continue
        errs = []
        if not isinstance(c, dict):
            invalid[f.stem] = "not a JSON object"
            continue
        if set(c) - keys:
            errs.append(f"unknown keys {sorted(set(c) - keys)}")
        if actions and c.get("action") not in actions:
            errs.append(f"action: one of {list(actions)}")
        if c.get("id") != f.stem or not ID_RE.match(str(c.get("id", ""))):
            errs.append("id must equal the file name")
        if not isinstance(c.get("what"), str) or not c["what"].strip():
            errs.append("what: required")
        rs = c.get("requests")
        if not isinstance(rs, list) or not rs or not all(isinstance(r, str) for r in rs):
            errs.append("requests: non-empty list of request ids")
        else:
            unknown = [r for r in rs if r not in known_ids]
            if unknown:
                errs.append(f"requests not in the queue: {unknown}")
        if errs:
            invalid[f.stem] = "; ".join(errs)
        else:
            valid[f.stem] = c
    return valid, invalid


def _cancel(c: dict, reqs: dict, run: Run, environ, cwd) -> dict:
    """scancel the still-open jobs of the named requests (once per cancel)."""
    ids, skipped = [], {}
    for rid in c["requests"]:
        e = reqs.get(rid, {})
        if not e.get("jobid"):
            skipped[rid] = "never submitted"
        elif e.get("finished"):
            skipped[rid] = "already finished"
        elif e.get("watch_only"):
            skipped[rid] = "watch-only (not submitted by this queue)"
        else:
            ids += job_ids(e)
    out = {"jobs": ids, "skipped": skipped}
    if ids:
        r = _call(run, ["scancel", *ids], environ, cwd, timeout=120)
        if r.returncode != 0:
            out["error"] = (r.stderr or r.stdout or f"scancel exit {r.returncode}")[:300]
    return out


# ----------------------------------------------------------------------- state


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_state(path: Path) -> dict:
    if path.exists():
        s = json.loads(path.read_text(encoding="utf-8"))
    else:
        s = {}
    s.setdefault("requests", {})
    s.setdefault("cancels", {})
    s.setdefault("last_heartbeat", 0.0)
    return s


# ---------------------------------------------------------------------- slurm


def parse_job_id(out: str) -> str | None:
    """--parsable output: "123" or "123;cluster"; None unless numeric."""
    first = (out or "").strip().splitlines()[0] if (out or "").strip() else ""
    jid = first.split(";", 1)[0].strip()
    return jid if JOB_ID_RE.match(jid) else None


def job_ids(entry: dict) -> list[str]:
    return [j for j in (entry.get("jobid") or "").split(",") if j]


def poll_states(run: Run, ids: list[str], env, cwd) -> tuple[dict | None, str | None]:
    """{base job id: [state per allocation]} from sacct, or (None, error)."""
    if not ids:
        return {}, None
    r = _call(
        run,
        ["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "--format=JobID,State"],
        env,
        cwd,
        timeout=120,
    )
    if r.returncode != 0:
        return None, (r.stderr or r.stdout)[:300]
    states: dict[str, list[str]] = {}
    for line in r.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2 or not parts[0]:
            continue
        jid = parts[0]
        if "." in jid:  # job step (123.batch, 123.0)
            continue
        base = jid.split("_", 1)[0]
        st = parts[1].split()[0] if parts[1].split() else ""
        states.setdefault(base, []).append(st)
    return states, None


def cluster_snapshot(run: Run, env=None, cwd=None) -> dict:
    """All of the user's jobs (informational): squeue now + sacct last 24 h."""
    snap: dict = {"squeue": [], "counts": {}, "ended_24h": []}
    r = _call(run, ["squeue", "--me", "-h", "-o", "%i|%j|%T|%M|%l|%D|%R|%S"], env, cwd, timeout=120)
    if r.returncode != 0:
        snap["squeue_error"] = (r.stderr or r.stdout or "squeue failed")[:300]
    else:
        keys = [
            "jobid",
            "name",
            "state",
            "elapsed",
            "time_limit",
            "nodes",
            "reason_or_nodelist",
            "start",
        ]
        for line in r.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 7)
            parts += [""] * (8 - len(parts))
            row = dict(zip(keys, (p.strip() for p in parts), strict=True))
            snap["squeue"].append(row)
            snap["counts"][row["state"]] = snap["counts"].get(row["state"], 0) + 1
    r = _call(
        run,
        [
            "sacct",
            "-n",
            "-P",
            "-X",
            "-S",
            "now-24hours",
            "--format=JobID,JobName,State,Elapsed,ExitCode,End",
        ],
        env,
        cwd,
        timeout=120,
    )
    if r.returncode != 0:
        snap["sacct_error"] = (r.stderr or r.stdout or "sacct failed")[:300]
    else:
        for line in r.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 6 or "." in parts[0]:
                continue
            # A job name containing '|' would shift the fields: rejoin it.
            jid, name, st, el, ex, end = (parts[0], "|".join(parts[1:-4]), *parts[-4:])
            state = st.split()[0] if st.split() else ""
            if state in TERMINAL:
                snap["ended_24h"].append(
                    {
                        "jobid": jid,
                        "name": name,
                        "state": state,
                        "elapsed": el,
                        "exit": ex,
                        "end": end,
                    }
                )
    return snap


# ------------------------------------------------------------------------ tick


def _submit(
    req: dict, dep_ids: list[str], repo: Path, run: Run, environ
) -> tuple[str | None, str | None]:
    """(job id, None) or (None, error)."""
    try:
        env = dict(environ) | {k: expand(v, environ) for k, v in req.get("env", {}).items()}
    except ValueError as e:
        return None, str(e)[:300]
    sb = req.get("sbatch", {})
    cmd = ["sbatch", "--parsable", f"--qos={sb.get('qos', DEFAULT_QOS)}"]
    for k in ("time", "array"):
        if k in sb:
            cmd.append(f"--{k}={sb[k]}")
    if dep_ids:
        cmd.append(f"--dependency={req.get('after_mode', 'afterok')}:" + ":".join(dep_ids))
    cmd.append(req["script"])
    r = _call(run, cmd, env, str(repo), timeout=120)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout or f"sbatch exit {r.returncode}")[:300]
    jid = parse_job_id(r.stdout)
    if jid is None:
        return None, f"unparseable sbatch output: {r.stdout!r}"[:300]
    return jid, None


def _run_post(req: dict, repo: Path, run: Run, environ) -> tuple[str, str | None]:
    """(output tail, error or None)."""
    post = req["post"]
    try:
        args = [expand(a, environ) for a in post.get("args", [])]
    except ValueError as e:
        return "", str(e)[:500]
    r = _call(
        run,
        [sys.executable, "-m", post["module"], *args],
        dict(environ),
        str(repo),
        timeout=POST_TIMEOUT,
    )
    out = ((r.stdout or "") + (r.stderr or ""))[-3000:]
    err = None if r.returncode == 0 else (f"exit {r.returncode}: " + (r.stderr or r.stdout))[:500]
    return out, err


def _collect(message: str, repo: Path, run: Run, environ) -> str | None:
    r = _call(run, [*COLLECTOR, message], dict(environ), str(repo), timeout=900)
    return None if r.returncode == 0 else (r.stderr or r.stdout or "collector failed")[-300:]


def tick(
    repo,
    state_path,
    status_path=None,
    run: Run = default_run,
    environ=None,
    clock: Callable[[], float] = time.time,
) -> dict | None:
    """One pass: submit, poll, post-process, collect, status, heartbeat.
    Returns a summary dict, or None when another runner holds the lock."""
    repo = Path(repo)
    state_path = Path(state_path)
    environ = dict(os.environ if environ is None else environ)
    if status_path is None:
        base = environ.get("OPS_BASE")
        if not base:
            raise SystemExit(
                "OPS_BASE is not set (scripts/ops/runner.sh sets it) and no --status given"
            )
        status_path = Path(base) / "status.json"
    status_path = Path(status_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(f"{state_path}.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        print(f"another runner holds {state_path}.lock; this tick does nothing")
        return None
    try:
        return _tick_locked(repo, state_path, status_path, run, environ, clock)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _tick_locked(repo: Path, state_path: Path, status_path: Path, run: Run, environ, clock) -> dict:
    now = clock()
    state = load_state(state_path)
    reqs = state["requests"]
    valid, invalid = load_requests(repo)
    summary = {"submitted": 0, "finished": 0, "collected": 0, "invalid": len(invalid)}

    for rid, reason in invalid.items():
        e = reqs.setdefault(rid, {})
        e["invalid"] = reason
    for rid in valid:
        reqs.setdefault(rid, {}).pop("invalid", None)

    # Submission: each valid request at most once; repeat passes so a request
    # whose dependency was submitted earlier in this tick goes out now too.
    # At most one pass per request (+1): a dependency chain of length n needs n
    # passes; the bound keeps the tick finite whatever the state holds.
    attempts = 0
    progress = True
    for _ in range(len(valid) + 1):
        if not progress:
            break
        progress = False
        for rid, req in valid.items():
            e = reqs[rid]
            if e.get("jobid"):
                continue  # already submitted / registered: never again
            if "script" not in req:  # watch-only
                e.update(
                    jobid=",".join(req["after_jobs"]),
                    watch_only=True,
                    submitted=_iso(now),
                    what=req["what"],
                )
                e.pop("waiting_for", None)
                atomic_write_json(state_path, state)
                progress = True
                continue
            missing = [a for a in req.get("after", []) if not reqs.get(a, {}).get("jobid")]
            if missing:
                e["waiting_for"] = missing
                continue
            e.pop("waiting_for", None)
            # A dependency that already finished is checked here, not by Slurm:
            # sbatch rejects --dependency on a job slurmctld has purged
            # ("Job dependency problem"), which a long-finished job is.
            mode = req.get("after_mode", "afterok")
            failed = [
                a
                for a in req.get("after", [])
                if reqs[a].get("finished")
                and mode == "afterok"
                and set(reqs[a].get("states", [])) != {"COMPLETED"}
            ]
            if failed:
                e["submit_error"] = (
                    f"dependency did not complete: {failed} (afterok); not submitted"
                )
                continue
            if attempts >= MAX_NEW:
                continue
            attempts += 1
            dep_ids = [
                j
                for a in req.get("after", [])
                if not reqs[a].get("finished")
                for j in job_ids(reqs[a])
            ]
            jid, err = _submit(req, dep_ids, repo, run, environ)
            if jid:
                e.update(jobid=jid, submitted=_iso(now), what=req["what"])
                e.pop("submit_error", None)
                summary["submitted"] += 1
                progress = True
            else:
                e["submit_error"] = err
                e["submit_attempts"] = e.get("submit_attempts", 0) + 1
            atomic_write_json(state_path, state)  # immediately after each submission

    # Cancellations (each cancel file acts once; a failed scancel is retried).
    cancels, bad_cancels = load_cancels(repo, set(valid))
    summary["cancelled"] = 0
    for cid, c in cancels.items():
        done = state["cancels"].get(cid)
        if done and not done.get("error"):
            continue
        res = _cancel(c, reqs, run, environ, str(repo))
        res["at"] = _iso(now)
        state["cancels"][cid] = res
        summary["cancelled"] += len(res["jobs"]) if not res.get("error") else 0
        atomic_write_json(state_path, state)
    for cid, reason in bad_cancels.items():
        state["cancels"][cid] = {"invalid": reason}
    # Poll unfinished jobs.
    open_ids = sorted(
        {
            j
            for rid, e in reqs.items()
            if rid in valid and e.get("jobid") and not e.get("finished")
            for j in job_ids(e)
        }
    )
    states, poll_err = poll_states(run, open_ids, environ, str(repo))
    state["poll_error"] = poll_err
    to_collect = []
    if states is not None:
        for rid, e in reqs.items():
            if rid not in valid or not e.get("jobid") or e.get("finished"):
                continue
            ids = job_ids(e)
            got = [s for j in ids for s in states.get(j, [])]
            if got:
                e["states"] = sorted(set(got))
            if all(states.get(j) for j in ids) and all(s in TERMINAL for s in got):
                e["finished"] = _iso(now)
                summary["finished"] += 1

    # Post-process and collect finished requests (once).
    for rid, e in reqs.items():
        if rid not in valid or not e.get("finished") or e.get("collected"):
            continue
        req = valid[rid]
        if "post" in req:
            out, err = _run_post(req, repo, run, environ)
            e["post_output"] = out
            if err:
                e["post_error"] = err
            else:
                e.pop("post_error", None)
        e["collected"] = _iso(now)
        summary["collected"] += 1
        atomic_write_json(state_path, state)
        if req.get("collect"):
            to_collect.append((rid, f"ops {rid}: {req['what']} ({', '.join(e.get('states', []))})"))

    snap = cluster_snapshot(run, environ, str(repo))
    summary["cluster"] = snap["counts"]
    status = {
        "updated": _iso(now),
        "queue": sorted(set(valid) | set(invalid)),
        "cluster": snap,
        "requests": {},
        "cancels": state["cancels"],
    }
    for rid in status["queue"]:
        e = reqs.get(rid, {})
        status["requests"][rid] = {
            "jobid": e.get("jobid"),
            "states": e.get("states", []),
            "submitted": e.get("submitted"),
            "finished": e.get("finished"),
            "collected": e.get("collected"),
            "submit_error": e.get("submit_error"),
            "post_error": e.get("post_error"),
            "invalid": e.get("invalid"),
            "waiting_for": e.get("waiting_for"),
            "post_output_tail": (e.get("post_output") or "")[-800:],
        }
    atomic_write_json(status_path, status)

    # One collector call (one commit + push) per tick: it pushes all of
    # results/ anyway, and one push per finished request made a tick with many
    # finished requests take tens of minutes.
    if to_collect:
        ids = [rid for rid, _ in to_collect]
        msg = (
            to_collect[0][1]
            if len(ids) == 1
            else (
                f"ops {len(ids)} requests: "
                + ", ".join(ids[:6])
                + (", ..." if len(ids) > 6 else "")
            )
        )
        err = _collect(msg, repo, run, environ)
        for rid in ids:
            reqs[rid]["collect_error"] = err
        if err:
            print(f"collector failed for {', '.join(ids)}: {err}")
    # Heartbeat: status reaches git about hourly even before any job ends.
    if now - float(state.get("last_heartbeat", 0)) >= HEARTBEAT_EVERY:
        err = _collect("ops heartbeat (status only)", repo, run, environ)
        state["heartbeat_error"] = err
        if err:
            print(f"heartbeat push failed (retried next hour): {err}")
        state["last_heartbeat"] = now
    atomic_write_json(state_path, state)
    summary["time"] = _iso(now)
    return summary


# ------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m rqops.queue")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tick")
    t.add_argument("--repo", required=True)
    t.add_argument("--state", required=True)
    t.add_argument("--status", default=None, help="default $OPS_BASE/status.json")
    v = sub.add_parser("validate")
    v.add_argument("--repo", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        valid, invalid = load_requests(Path(a.repo))
        cvalid, cinvalid = load_cancels(Path(a.repo), set(valid))
        for stem, reason in sorted(invalid.items()):
            print(f"INVALID {stem}: {reason}")
        for stem, reason in sorted(cinvalid.items()):
            print(f"INVALID control {stem}: {reason}")
        print(
            f"{len(valid)} valid, {len(invalid)} invalid; {len(cvalid)} cancel files valid, "
            f"{len(cinvalid)} invalid"
        )
        return 1 if invalid or cinvalid else 0
    s = tick(a.repo, a.state, a.status)
    if s is not None:
        counts = " ".join(f"{k}={v}" for k, v in sorted(s["cluster"].items())) or "none"
        print(
            f"{s['time']} submitted={s['submitted']} finished={s['finished']} "
            f"collected={s['collected']} invalid={s['invalid']} | cluster: {counts}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
