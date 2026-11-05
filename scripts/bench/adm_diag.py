#!/usr/bin/env python3
"""Why does the unmodified ADM evaluator's FID fail on vhpc01?  (diagnostic only)

    python scripts/bench/adm_diag.py $ADM_DIR $OUT_DIR

The calibration job (1506465) reproduced IS 215.837158203125 exactly, then FID raised
"Imaginary component 0.0646". Off-cluster, the same lock, evaluator commit and official
batches give FID 3.94257 / sFID 6.14043 (the published values) with a fully real sqrtm,
for every OpenBLAS kernel type and thread count tried. This script measures, on the
cluster CPU:

1. the machine and the BLAS each library loaded;
2. the sample-batch statistics, computed with the evaluator's own code (cached in OUT),
   with fingerprints to compare against the off-cluster values below;
3. the evaluator's own ``frechet_distance`` and the raw ``sqrtm`` step, in fresh
   processes, under several OpenBLAS thread counts and kernel types.

Nothing here feeds a reported number; it only locates the cause.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import subprocess
import sys

OFF_CLUSTER = {"IS": 215.837158203125, "mu_sum": 484.324591524899, "trace": 159.40084783863261,
               "FID": 3.9425650421262617, "sFID": 6.14042664133342,
               "prod_sum": {"FID": 142402.55221067026, "sFID": 41910.87643270046}}

REPLAY = r'''
import json, sys
import numpy as np
from scipy import linalg
sys.path.insert(0, sys.argv[1])
import evaluator as ev                                   # unmodified, imports TensorFlow
r, s = np.load(sys.argv[2]), np.load(sys.argv[3])
out = {}
for name, mu1, s1, mu2, s2 in (("FID", s["mu"], s["sigma"], r["mu"], r["sigma"]),
                               ("sFID", s["mu_s"], s["sigma_s"], r["mu_s"], r["sigma_s"])):
    prod = s1.dot(s2)
    cm, err = linalg.sqrtm(prod, disp=False)
    rec = {"prod_finite": bool(np.isfinite(prod).all()),
           "sqrtm_finite": bool(np.isfinite(cm).all()), "complex": bool(np.iscomplexobj(cm)),
           "max_imag": float(np.abs(np.imag(cm)).max()), "sqrtm_err": float(err),
           "prod_sum": float(prod.sum())}
    try:
        rec["value"] = float(ev.FIDStatistics(mu1, s1).frechet_distance(ev.FIDStatistics(mu2, s2)))
    except Exception as e:                                # report, do not hide
        rec["error"] = f"{type(e).__name__}: {e}"
    out[name] = rec
print("RESULT " + json.dumps(out))
'''


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    adm, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    evals = adm / "guided-diffusion" / "evaluations"
    sys.path.insert(0, str(evals))

    import numpy as np
    import scipy
    print("=== 1. machine and libraries")
    print("host", platform.node(), "|", sh("lscpu | grep 'Model name'"), "| cpus usable",
          len(os.sched_getaffinity(0)))
    print("numpy", np.__version__, "scipy", scipy.__version__)
    print(sh(f"OPENBLAS_VERBOSE=2 {sys.executable} -c 'import numpy, scipy.linalg' 2>&1 | head -4"))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_CORETYPE",
              "TF_ENABLE_ONEDNN_OPTS", "SLURM_CPUS_PER_TASK"):
        print(f"  {k}={os.environ.get(k)}")

    stats = out / "sample_stats.npz"
    print("=== 2. sample-batch statistics (evaluator code)")
    if not stats.exists():
        import evaluator as ev
        import tensorflow.compat.v1 as tf
        e = ev.Evaluator(tf.Session(config=tf.ConfigProto(allow_soft_placement=True)))
        e.warmup()
        pool, spatial = e.read_activations(str(adm / "admnet_guided_upsampled_imagenet256.npz"))
        fs, fss = e.compute_statistics(pool), e.compute_statistics(spatial)
        np.savez(stats, mu=fs.mu, sigma=fs.sigma, mu_s=fss.mu, sigma_s=fss.sigma,
                 IS=e.compute_inception_score(pool), n=len(pool))
    z = np.load(stats)
    got = {"IS": float(z["IS"]), "mu_sum": float(z["mu"].astype(np.float64).sum()),
           "trace": float(np.trace(z["sigma"]))}
    for k, v in got.items():
        print(f"  {k}: {v!r}  (off-cluster {OFF_CLUSTER[k]!r}, diff {v - OFF_CLUSTER[k]:.3e})")

    print("=== 3. FID replay in fresh processes")
    ref = adm / "VIRTUAL_imagenet256_labeled.npz"
    ncpu = str(len(os.sched_getaffinity(0)))
    runs = [("default threads", {}), ("1 thread", {"OPENBLAS_NUM_THREADS": "1"}),
            (f"{ncpu} threads", {"OPENBLAS_NUM_THREADS": ncpu}),
            ("Haswell kernels", {"OPENBLAS_CORETYPE": "Haswell"}),
            ("Haswell, 1 thread", {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1"})]
    summary = {}
    for label, extra in runs:
        env = {**os.environ, **extra}
        r = subprocess.run([sys.executable, "-c", REPLAY, str(evals), str(ref), str(stats)],
                           env=env, capture_output=True, text=True)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print(f"  [{label}] process failed rc={r.returncode}: {r.stderr.strip()[-300:]}")
            summary[label] = {"failed": r.returncode}
            continue
        res = json.loads(line[7:])
        summary[label] = res
        for name, rec in res.items():
            print(f"  [{label}] {name}: " + ", ".join(f"{k}={v}" for k, v in rec.items()))
    (out / "adm_diag.json").write_text(json.dumps({"stats": got, "replay": summary}, indent=2))
    print(f"=== written {out / 'adm_diag.json'}")


if __name__ == "__main__":
    main()
