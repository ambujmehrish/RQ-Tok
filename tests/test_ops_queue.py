"""Git-driven job queue (rqops/queue.py, adapted from FaCT) against a fake Slurm: no real
sbatch / sacct / squeue is ever called."""

import fcntl
import json
import os
import sys

import pytest

from rqops import queue as q

ENV = {
    "LEVELB_ROOT": "/work/levelb",
    "BENCH_ROOT": "/work/bench",
    "BIFROST_ROOT": "/work/b1",
    "OPS_BASE": "/work/ops",
    "PATH": "/usr/bin",
}


class FakeSlurm:
    """run(cmd, env, cwd, timeout) stand-in. Jobs get ids 1001, 1002, ...;
    sacct reports self.states (default PENDING)."""

    def __init__(self):
        self.calls = []
        self.next_id = 1001
        self.states = {}
        self.sbatch_error = None
        self.squeue_out = ""
        self.squeue_fail = False
        self.sacct24_out = ""
        self.step_lines = True

    def __call__(self, cmd, env=None, cwd=None, timeout=None):
        self.calls.append({"cmd": list(cmd), "env": dict(env or {}), "cwd": cwd})
        if cmd[0] == "sbatch":
            if self.sbatch_error:
                return q.Result(1, "", self.sbatch_error)
            jid = str(self.next_id)
            self.next_id += 1
            self.states.setdefault(jid, "PENDING")
            return q.Result(0, f"{jid};vhpc\n", "")
        if cmd[0] == "sacct" and "-j" in cmd:
            ids = cmd[cmd.index("-j") + 1].split(",")
            lines = []
            for j in ids:
                if j in self.states:
                    lines.append(f"{j}|{self.states[j]}")
                    if self.step_lines:
                        lines.append(f"{j}.batch|FAILED")  # step lines must be ignored
            return q.Result(0, "\n".join(lines) + "\n", "")
        if cmd[0] == "sacct":
            return q.Result(0, self.sacct24_out, "")
        if cmd[0] == "squeue":
            if self.squeue_fail:
                return q.Result(1, "", "slurm_load_jobs error: Socket timed out")
            return q.Result(0, self.squeue_out, "")
        if cmd[0] == "scancel":
            if getattr(self, "scancel_fail", False):
                return q.Result(1, "", "scancel: error: Kill job error")
            for j in cmd[1:]:
                self.states[j] = "CANCELLED"
            return q.Result(0, "", "")
        if cmd[0] == "scontrol":
            self.scontrol = getattr(self, "scontrol", []) + [cmd[1:]]
            return q.Result(0, "", "")
        if cmd[:2] == q.COLLECTOR:
            return q.Result(0, "collector: pushed", "")
        if cmd[0] == sys.executable:
            return q.Result(0, "post done\n", "")
        raise AssertionError(f"unexpected command {cmd}")

    def of(self, name):
        return [c for c in self.calls if c["cmd"][0] == name]

    def sbatch(self):
        return self.of("sbatch")

    def collects(self, heartbeat=False):
        cs = [c["cmd"][2] for c in self.calls if c["cmd"][:2] == q.COLLECTOR]
        return cs if heartbeat else [m for m in cs if "heartbeat" not in m]

    def posts(self):
        # Locate post-step commands, skipping sbatch/sacct/squeue calls.
        return [
            c
            for c in self.calls
            if c["cmd"][0] not in ("sbatch", "sacct", "squeue") and c["cmd"][0] == sys.executable
        ]


@pytest.fixture
def repo(tmp_path):
    # State and status live next to (outside) the repo, as on the cluster.
    r = tmp_path / "repo"
    (r / "scripts" / "levelb").mkdir(parents=True)
    (r / "scripts" / "bench").mkdir(parents=True)
    (r / "scripts/levelb/eval_baselines.sbatch").write_text("#!/bin/bash\n")
    (r / "scripts/bench/tripwires.sbatch").write_text("#!/bin/bash\n")
    (r / "ops" / "queue").mkdir(parents=True)
    return r


def put(repo, rid, **fields):
    req = {"id": rid, "what": f"test {rid}"} | fields
    (repo / "ops" / "queue" / f"{rid}.json").write_text(json.dumps(req))
    return req


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def do_tick(repo, fake, clock=None, environ=ENV):
    return q.tick(
        repo,
        repo.parent / "state" / "state.json",
        repo.parent / "status.json",
        run=fake,
        environ=environ,
        clock=clock or Clock(),
    )


def status(repo):
    return json.loads((repo.parent / "status.json").read_text())


# ------------------------------------------------------------------ validation

GOOD = {"id": "r1", "what": "x", "script": "scripts/levelb/eval_baselines.sbatch"}


@pytest.mark.parametrize(
    "patch, needle",
    [
        ({"script": "../scripts/levelb/eval_baselines.sbatch"}, "script"),
        ({"script": "eval_baselines.sbatch"}, "script"),
        ({"script": "scripts/bifrost1/generate.sbatch"}, "script"),
        ({"script": "scripts/levelb/../bench/x.sbatch"}, "script"),
        ({"script": "scripts/levelb/missing.sbatch"}, "does not exist"),
        ({"env": {"OUT": "a; rm -rf /"}}, "env.OUT"),
        ({"env": {"OUT": "a|b"}}, "env.OUT"),
        ({"env": {"OUT": "`id`"}}, "env.OUT"),
        ({"env": {"OUT": "a\nb"}}, "env.OUT"),
        ({"env": {"OUT": "'q'"}}, "env.OUT"),
        ({"env": {"OUT": "${HOME}/x"}}, "env.OUT"),
        ({"env": {"OUT": "$HOME/x"}}, "env.OUT"),
        ({"env": {"OUT": "${LEVELB_ROOT/x"}}, "env.OUT"),
        ({"env": {"OUT": "$LEVELB_ROOTX/x"}}, "env.OUT"),
        ({"env": {"OUT": "{x}"}}, "env.OUT"),
        ({"env": {"lower": "x"}}, "env"),
        ({"sbatch": {"partition": "gpus"}}, "sbatch"),
        ({"sbatch": {"account": "OTHER"}}, "sbatch"),
        ({"sbatch": {"time": "1 day"}}, "sbatch.time"),
        ({"sbatch": {"array": "0-3;id"}}, "sbatch.array"),
        ({"post": {"module": "os.system", "args": []}}, "post.module"),
        ({"post": {"module": "adarq_flow.training", "args": []}}, "post.module"),
        ({"post": {"module": "levelb.status_x", "args": ["$(id)"]}}, "post.args"),
        ({"after": ["r1"]}, "itself"),
        ({"after": ["nope"]}, "unknown request id"),
        ({"nonsense": 1}, "keys"),
        ({"id": "other"}, "file stem"),
        ({"collect": "yes"}, "collect"),
        ({"what": "x" * 201}, "what"),
        ({"after_mode": "afterany"}, "after_mode"),
        ({"after": [], "after_mode": "afterany"}, "after_mode"),
        ({"after": ["r0"], "after_mode": "singleton"}, "after_mode"),
    ],
)
def test_validation_refuses(repo, patch, needle):
    errs = q.validate(GOOD | patch, "r1", repo, known_ids={"r0", "r1"})
    assert errs and any(needle in e for e in errs), errs


@pytest.mark.parametrize(
    "name",
    sorted(q.ENV_DENIED)
    + ["LD_PRELOAD", "DYLD_X", "BASH_FUNC_x", "SLURM_JOB_ID", "PYTHONWARNINGS"],
)
def test_validation_refuses_denied_env_names(repo, name):
    errs = q.validate(GOOD | {"env": {name: "x"}}, "r1", repo, {"r1"})
    assert any("denied" in e or "must match" in e for e in errs), errs


@pytest.mark.parametrize(
    "patch, needle",
    [
        ({"after_jobs": ["12a"]}, "after_jobs"),
        ({"after_jobs": []}, "after_jobs"),
        ({"after_jobs": ["123"], "env": {"A": "b"}}, "env: not allowed"),
        ({"after_jobs": ["123"], "sbatch": {"time": "01:00:00"}}, "sbatch: not allowed"),
    ],
)
def test_validation_watch_only_rules(repo, patch, needle):
    req = {"id": "w1", "what": "watch"} | patch
    errs = q.validate(req, "w1", repo, {"w1"})
    assert any(needle in e for e in errs), errs


def test_validation_accepts_placeholders_and_typical_values(repo):
    req = GOOD | {
        "env": {
            "BASELINES": "x:ckpt=${BIFROST_ROOT}/r/step_0100000.pt:label=a@1_T0.5",
            "EXTRA": "--wer-model large-v3 --utmos --partner-subset",
            "OUT": "$LEVELB_ROOT/results/x",
        },
        "sbatch": {"time": "12:00:00", "qos": "normal", "array": "0-3%4"},
        "post": {"module": "levelb.status_x", "args": ["--out", "${LEVELB_ROOT}/x"]},
        "collect": True,
    }
    assert q.validate(req, "r1", repo, {"r1"}) == []


def test_validate_cli(repo, capsys):
    put(repo, "ok", script="scripts/levelb/eval_baselines.sbatch")
    put(repo, "bad", script="../x.sbatch")
    assert q.main(["validate", "--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "INVALID bad: script" in out and "1 valid, 1 invalid" in out


def test_parse_job_id():
    assert q.parse_job_id("123\n") == "123" and q.parse_job_id("123;vhpc") == "123"
    assert q.parse_job_id("Submitted batch job 5") is None and q.parse_job_id("") is None


# ------------------------------------------------------------------------ tick


def test_tick_full_lifecycle(repo):
    fake = FakeSlurm()
    put(
        repo,
        "a",
        script="scripts/levelb/eval_baselines.sbatch",
        env={"OUT": "${LEVELB_ROOT}/results/x", "CKPT": "$BIFROST_ROOT/m/step.pt"},
        sbatch={"time": "02:00:00"},
        post={"module": "levelb.status_x", "args": ["--out", "${LEVELB_ROOT}/results/x/s.json"]},
        collect=True,
    )
    put(repo, "b", script="scripts/bench/tripwires.sbatch", after=["a"], collect=True)
    put(repo, "bad", script="../evil.sbatch")
    put(repo, "c", script="scripts/bench/tripwires.sbatch", after=["bad"])

    s = do_tick(repo, fake)
    sb = fake.sbatch()
    assert s["submitted"] == 2 and len(sb) == 2
    a_cmd, b_cmd = sb[0]["cmd"], sb[1]["cmd"]
    assert a_cmd[:3] == ["sbatch", "--parsable", f"--qos={q.DEFAULT_QOS}"]
    assert "--time=02:00:00" in a_cmd and a_cmd[-1] == "scripts/levelb/eval_baselines.sbatch"
    # env through the process environment, placeholders expanded; no shell string.
    assert (
        sb[0]["env"]["OUT"] == "/work/levelb/results/x"
        and sb[0]["env"]["CKPT"] == "/work/b1/m/step.pt"
    )
    assert sb[0]["env"]["PATH"] == "/usr/bin" and sb[0]["cwd"] == str(repo)
    assert "--dependency=afterok:1001" in b_cmd
    st = status(repo)["requests"]
    assert st["a"]["jobid"] == "1001" and st["b"]["jobid"] == "1002"
    assert "script" in st["bad"]["invalid"]
    assert st["c"]["waiting_for"] == ["bad"] and st["c"]["jobid"] is None

    # Running: nothing is finished or collected.
    fake.states.update({"1001": "RUNNING", "1002": "PENDING"})
    s = do_tick(repo, fake)
    assert s["finished"] == 0 and s["collected"] == 0 and not fake.posts()
    assert len(fake.sbatch()) == 2

    # Completed (the ".batch FAILED" step lines are ignored): post once, one collector call.
    fake.states.update({"1001": "COMPLETED", "1002": "CANCELLED by 42"})
    s = do_tick(repo, fake)
    assert s["finished"] == 2 and s["collected"] == 2
    posts = fake.posts()
    assert len(posts) == 1
    assert posts[0]["cmd"][1:] == [
        "-m",
        "levelb.status_x",
        "--out",
        "/work/levelb/results/x/s.json",
    ]
    # one push when a and b were submitted, then one when they finished
    assert fake.collects() == ["ops submitted 2", "ops 2 requests: a, b"]
    st = status(repo)["requests"]
    assert st["a"]["states"] == ["COMPLETED"] and st["a"]["collected"]
    assert "post done" in st["a"]["post_output_tail"]

    # Further ticks (and a restarted runner: same state file) do nothing new.
    for _ in range(2):
        s = do_tick(repo, fake)
        assert s["submitted"] == 0 and s["collected"] == 0
    assert len(fake.sbatch()) == 2 and len(fake.posts()) == 1 and len(fake.collects()) == 2


def test_array_tasks_map_to_base_id(repo):
    fake = FakeSlurm()
    put(repo, "arr", script="scripts/levelb/eval_baselines.sbatch", sbatch={"array": "0-1"})
    do_tick(repo, fake)
    fake.states = {}
    fake.step_lines = False
    real = fake.__call__

    def run(cmd, env=None, cwd=None, timeout=None):
        if cmd[0] == "sacct" and "-j" in cmd:
            fake.calls.append({"cmd": list(cmd), "env": {}, "cwd": cwd})
            return q.Result(0, f"1001_0|{tasks[0]}\n1001_1|{tasks[1]}\n1001_1.batch|FAILED\n", "")
        return real(cmd, env, cwd, timeout)

    tasks = ["COMPLETED", "RUNNING"]
    assert do_tick(repo, run)["finished"] == 0
    tasks = ["COMPLETED", "FAILED"]
    assert do_tick(repo, run)["finished"] == 1
    assert status(repo)["requests"]["arr"]["states"] == ["COMPLETED", "FAILED"]


def test_submit_errors_recorded_and_retried(repo):
    fake = FakeSlurm()
    fake.sbatch_error = "sbatch: error: QOSMaxSubmitJobPerUserLimit " + "x" * 400
    put(repo, "a", script="scripts/levelb/eval_baselines.sbatch")
    do_tick(repo, fake)
    st = status(repo)["requests"]["a"]
    assert st["jobid"] is None and "QOSMaxSubmitJobPerUserLimit" in st["submit_error"]
    assert len(st["submit_error"]) <= 300
    fake.sbatch_error = None
    do_tick(repo, fake)
    st = status(repo)["requests"]["a"]
    assert st["jobid"] == "1001" and st["submit_error"] is None
    assert len(fake.sbatch()) == 2


def test_unparseable_sbatch_output_is_an_error(repo):
    fake = FakeSlurm()
    real = fake.__call__

    def run(cmd, env=None, cwd=None, timeout=None):
        if cmd[0] == "sbatch":
            return q.Result(0, "Submitted batch job 77\n", "")
        return real(cmd, env, cwd, timeout)

    put(repo, "a", script="scripts/levelb/eval_baselines.sbatch")
    do_tick(repo, run)
    assert "unparseable" in status(repo)["requests"]["a"]["submit_error"]


def test_unset_placeholder_is_a_submit_error(repo):
    fake = FakeSlurm()
    put(repo, "a", script="scripts/levelb/eval_baselines.sbatch", env={"OUT": "${BENCH_ROOT}/x"})
    env = {k: v for k, v in ENV.items() if k != "BENCH_ROOT"}
    do_tick(repo, fake, environ=env)
    assert not fake.sbatch() and "BENCH_ROOT" in status(repo)["requests"]["a"]["submit_error"]


def test_max_new_bounds_one_tick(repo, monkeypatch):
    monkeypatch.setattr(q, "MAX_NEW", 3)
    fake = FakeSlurm()
    for i in range(5):
        put(repo, f"r{i}", script="scripts/levelb/eval_baselines.sbatch")
    assert do_tick(repo, fake)["submitted"] == 3
    assert do_tick(repo, fake)["submitted"] == 2
    assert len(fake.sbatch()) == 5


def test_watch_only(repo):
    fake = FakeSlurm()
    put(
        repo,
        "w",
        after_jobs=["555", "556"],
        post={"module": "levelb.status_x", "args": ["${LEVELB_ROOT}/r"]},
        collect=True,
    )
    do_tick(repo, fake)
    assert not fake.sbatch()
    assert status(repo)["requests"]["w"]["jobid"] == "555,556"
    # One job sacct does not know yet: not finished.
    fake.states = {"555": "COMPLETED"}
    assert do_tick(repo, fake)["finished"] == 0 and not fake.posts()
    fake.states = {"555": "COMPLETED", "556": "TIMEOUT"}
    s = do_tick(repo, fake)
    assert s["finished"] == 1 and s["collected"] == 1
    assert fake.posts()[0]["cmd"][-1] == "/work/levelb/r"
    assert fake.collects() == ["ops w: test w (COMPLETED, TIMEOUT)"]
    assert not fake.sbatch()


def test_post_failure_recorded_without_crash(repo):
    fake = FakeSlurm()
    real = fake.__call__

    def run(cmd, env=None, cwd=None, timeout=None):
        if cmd[0] == sys.executable:
            raise TimeoutError("post took too long")
        return real(cmd, env, cwd, timeout)

    put(repo, "w", after_jobs=["9"], post={"module": "levelb.status_x", "args": []})
    fake.states = {"9": "COMPLETED"}
    do_tick(repo, run)
    st = status(repo)["requests"]["w"]
    assert "post took too long" in st["post_error"] and st["collected"]


def test_heartbeat_hourly(repo):
    fake = FakeSlurm()
    clock = Clock()
    do_tick(repo, fake, clock)
    assert fake.collects(heartbeat=True) == ["ops heartbeat (status only)"]
    clock.t += 600
    do_tick(repo, fake, clock)
    assert len(fake.collects(heartbeat=True)) == 1
    clock.t += 3001
    do_tick(repo, fake, clock)
    assert len(fake.collects(heartbeat=True)) == 2


def test_cluster_snapshot_lists_all_jobs(repo):
    fake = FakeSlurm()
    fake.squeue_out = (
        "61816481|fact-nc8rd05|RUNNING|8:47:08|1-00:00:00|2|lrdn[2166,2449]|2026-10-24T01:00:00\n"
        "61967096|paperA_s|PENDING|0:00|4:00:00|1|(Priority)|N/A\n"
        "61816485|fact-nc8rd05|PENDING|0:00|1-00:00:00|2|(Dependency)|N/A\n"
    )
    fake.sacct24_out = (
        "61816479|fact-nc8rd05|FAILED|04:01:00|1:0|2026-10-24T01:00:00\n"
        "61816481|fact-nc8rd05|RUNNING|08:47:08|0:0|Unknown\n"
        "62095097|fact-tripwires-ft|COMPLETED|00:20:00|0:0|2026-10-24T09:00:00\n"
    )
    do_tick(repo, fake)
    snap = status(repo)["cluster"]
    hand = [r for r in snap["squeue"] if r["jobid"] == "61967096"][0]
    assert hand["name"] == "paperA_s" and hand["reason_or_nodelist"] == "(Priority)"
    assert snap["counts"] == {"RUNNING": 1, "PENDING": 2}
    assert [r["jobid"] for r in snap["ended_24h"]] == ["61816479", "62095097"]
    assert snap["ended_24h"][0]["exit"] == "1:0"


def test_failing_squeue_does_not_stop_the_tick(repo):
    fake = FakeSlurm()
    fake.squeue_fail = True
    put(repo, "a", script="scripts/levelb/eval_baselines.sbatch")
    s = do_tick(repo, fake)
    assert s["submitted"] == 1
    snap = status(repo)["cluster"]
    assert "Socket timed out" in snap["squeue_error"] and snap["squeue"] == []


def test_second_runner_holding_the_lock_does_nothing(repo, capsys):
    fake = FakeSlurm()
    put(repo, "a", script="scripts/levelb/eval_baselines.sbatch")
    state = repo.parent / "state" / "state.json"
    state.parent.mkdir(parents=True)
    fd = os.open(f"{state}.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert do_tick(repo, fake) is None
        assert not fake.calls and "another runner" in capsys.readouterr().out
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert do_tick(repo, fake)["submitted"] == 1


def test_after_mode_afterany_for_training_chains(repo):
    fake = FakeSlurm()
    put(repo, "p1", script="scripts/levelb/eval_baselines.sbatch")
    put(
        repo,
        "p2",
        script="scripts/levelb/eval_baselines.sbatch",
        after=["p1"],
        after_mode="afterany",
    )
    put(repo, "w2", after_jobs=["5"], after_mode="afterany")
    do_tick(repo, fake)
    sb = fake.sbatch()
    assert len(sb) == 2 and "--dependency=afterany:1001" in sb[1]["cmd"]
    assert "after_mode" in status(repo)["requests"]["w2"]["invalid"]


# ------------------------------------------------------------------ cancel


def put_cancel(repo, cid, **fields):
    c = {"id": cid, "what": f"cancel {cid}"} | fields
    (repo / "ops" / "cancel").mkdir(parents=True, exist_ok=True)
    (repo / "ops" / "cancel" / f"{cid}.json").write_text(json.dumps(c))


def test_cancel_reaches_only_this_queues_open_jobs_once(repo):
    fake = FakeSlurm()
    put(repo, "a", script="scripts/levelb/eval_baselines.sbatch")
    put(repo, "b", script="scripts/bench/tripwires.sbatch", after=["a"])
    put(repo, "done", script="scripts/bench/tripwires.sbatch")
    put(repo, "w", after_jobs=["777"])
    do_tick(repo, fake)  # a=1001, b=1002, done=1003
    fake.states["1003"] = "COMPLETED"
    do_tick(repo, fake)
    put_cancel(repo, "c1", requests=["a", "b", "done", "w"])
    s = do_tick(repo, fake)
    sc = fake.of("scancel")
    assert [c["cmd"] for c in sc] == [["scancel", "1001", "1002"]] and s["cancelled"] == 2
    res = status(repo)["cancels"]["c1"]
    assert res["skipped"] == {
        "done": "already finished",
        "w": "watch-only (not submitted by this queue)",
    }
    # Acts once; the cancelled requests then finish normally.
    s = do_tick(repo, fake)
    assert len(fake.of("scancel")) == 1 and s["cancelled"] == 0
    assert status(repo)["requests"]["a"]["states"] == ["CANCELLED"]


def test_cancel_failure_is_retried_and_bad_cancels_refused(repo, capsys):
    fake = FakeSlurm()
    put(repo, "a", script="scripts/levelb/eval_baselines.sbatch")
    do_tick(repo, fake)
    fake.scancel_fail = True
    put_cancel(repo, "c1", requests=["a"])
    do_tick(repo, fake)
    assert "Kill job error" in status(repo)["cancels"]["c1"]["error"]
    fake.scancel_fail = False
    do_tick(repo, fake)
    assert len(fake.of("scancel")) == 2 and "error" not in status(repo)["cancels"]["c1"]
    # Raw job ids, unknown request ids and extra keys are refused.
    put_cancel(repo, "c2", requests=["12345"])
    put_cancel(repo, "c3", requests=["a"], jobs=["999"])
    do_tick(repo, fake)
    st = status(repo)["cancels"]
    assert "not in the queue" in st["c2"]["invalid"] and "unknown keys" in st["c3"]["invalid"]
    assert len(fake.of("scancel")) == 2
    assert q.main(["validate", "--repo", str(repo)]) == 1
    assert "INVALID control c2" in capsys.readouterr().out


def test_dependency_on_long_finished_job_is_checked_by_the_queue(repo):
    """sbatch rejects --dependency on a purged job ("Job dependency problem"):
    a dependency already finished in the state is checked here instead."""
    fake = FakeSlurm()
    put(repo, "prep", script="scripts/levelb/eval_baselines.sbatch")
    put(repo, "bad", script="scripts/levelb/eval_baselines.sbatch")
    do_tick(repo, fake)
    ids = {k: v["jobid"] for k, v in status(repo)["requests"].items()}
    fake.states.update({ids["prep"]: "COMPLETED", ids["bad"]: "FAILED"})
    do_tick(repo, fake)
    put(repo, "ok_after", script="scripts/bench/tripwires.sbatch", after=["prep"])
    put(repo, "blocked", script="scripts/bench/tripwires.sbatch", after=["bad"])
    put(
        repo,
        "any_after",
        script="scripts/bench/tripwires.sbatch",
        after=["bad"],
        after_mode="afterany",
    )
    put(repo, "chain", script="scripts/bench/tripwires.sbatch", after=["ok_after"])
    do_tick(repo, fake)
    st = status(repo)["requests"]
    by_jid = {str(1001 + i): c["cmd"] for i, c in enumerate(fake.sbatch())}
    assert st["blocked"]["jobid"] is None and all(
        st[r]["jobid"] for r in ("ok_after", "any_after", "chain")
    )
    sb = {r: by_jid[st[r]["jobid"]] for r in ("ok_after", "any_after", "chain")}
    assert "dependency did not complete" in st["blocked"]["submit_error"]
    assert not any(a.startswith("--dependency") for r in ("ok_after", "any_after") for a in sb[r])
    # A dependency still open is passed to Slurm as before.
    assert f"--dependency=afterok:{st['ok_after']['jobid']}" in sb["chain"]


def test_qos_from_request_or_default(repo):
    put(repo, "a", script="scripts/levelb/eval_baselines.sbatch")
    put(repo, "b", script="scripts/bench/tripwires.sbatch", sbatch={"qos": "normal"})
    fake = FakeSlurm()
    do_tick(repo, fake)
    qos = sorted(c["cmd"][2] for c in fake.sbatch())
    assert qos == ["--qos=gpulongjobs", "--qos=normal"]


def test_holds_are_not_supported(repo):
    put(repo, "a", script="scripts/levelb/eval_baselines.sbatch", hold=True)
    valid, invalid = q.load_requests(repo)
    assert "a" in invalid and "keys" in invalid["a"]


def test_submissions_and_first_failures_are_pushed_once(repo):
    put(repo, "a", script="scripts/levelb/eval_baselines.sbatch")
    fake = FakeSlurm()
    do_tick(repo, fake)
    assert fake.collects() == ["ops submitted 1"]
    do_tick(repo, fake)                       # nothing new: no extra push
    assert fake.collects() == ["ops submitted 1"]
    put(repo, "b", script="scripts/levelb/eval_baselines.sbatch")
    fake.sbatch_error = "sbatch: error: QOSMaxSubmitJobPerUserLimit"
    do_tick(repo, fake)                       # first failure: pushed once
    do_tick(repo, fake)                       # retried, not pushed again
    assert fake.collects() == ["ops submitted 1", "ops submitted 0; submit failed: b"]

