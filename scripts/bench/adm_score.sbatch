#!/bin/bash
# =====================================================================================
# Score an ImageNet-256 sample batch with the unmodified ADM evaluator.
#
#   # 1. calibrate once (reproduces the published ADM-G + ADM-U numbers):
#   sbatch --qos=gpulongjobs --export=ALL,CALIBRATE=1 scripts/bench/adm_score.sbatch
#   # 2. score 10,000 samples (Bifrost-1 §5.1 protocol, ADM reference batch):
#   sbatch --qos=gpulongjobs --export=ALL,SAMPLES=<samples_10000.npz> scripts/bench/adm_score.sbatch
#   # secondary (ours): against the 50,000 ImageNet validation images
#   sbatch --qos=gpulongjobs --export=ALL,SAMPLES=<npz>,REF=val scripts/bench/adm_score.sbatch
#
# Calibration requires every published value (guided-diffusion evaluations/README.md:
# IS 215.837, FID 3.9426, sFID 6.1404, Precision 0.8265, Recall 0.5309) within tolerance;
# until it passes no model is scored. Writes <samples>.<ref>.metrics.json.
#
# CPU only: the evaluator runs on tensorflow-cpu. On an H200 node (job 1506465) its FID
# raised "Imaginary component 0.0646"; on a CPU node (adm_diag, job 1506499) and off-cluster
# the same lock, commit and batches reproduce FID 3.9426 / sFID 6.1404 with a real sqrtm.
# BLAS threads are pinned to the allocated CPUs; the node and CPU are recorded with every
# result, and calibration and scoring run on the same kind of node.
# =====================================================================================
#SBATCH --job-name=adm-score
#SBATCH --partition=vhpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs
die() { echo "FAILED: $*" >&2; exit 1; }

BENCH_ROOT="${BENCH_ROOT:-$HOME/work/bench}"
LEVELB_ROOT="${LEVELB_ROOT:-$HOME/work/levelb}"
[ -f "$BENCH_ROOT/adm_env.sh" ] || die "run scripts/bench/adm_setup.sbatch first"
# shellcheck disable=SC1091
source "$BENCH_ROOT/adm_env.sh"
# shellcheck disable=SC1090
source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u   # conda's activate.d/deactivate.d scripts read unset variables; nounset off for conda only
conda activate "$ADM_ENV"
set -u
[ "$(git -C "$ADM_DIR/guided-diffusion" rev-parse HEAD)" = "$ADM_COMMIT" ] || die "evaluator checkout moved"
export TMPDIR="/tmp/$USER-${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
NCPU="${SLURM_CPUS_PER_TASK:?run through sbatch}"
export OMP_NUM_THREADS="$NCPU" OPENBLAS_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
CPU_MODEL=$(lscpu | sed -n 's/^Model name: *//p' | head -1)
echo "  node $(hostname) | $CPU_MODEL | $NCPU threads"

CALIBRATE="${CALIBRATE:-0}"
if [ "$CALIBRATE" = "1" ]; then
  SAMPLES="$ADM_DIR/admnet_guided_upsampled_imagenet256.npz"; REF="adm"
else
  [ -f "$BENCH_ROOT/ADM_CALIBRATED" ] || die "evaluator not calibrated yet; run with CALIBRATE=1 first"
  SAMPLES="${SAMPLES:?set SAMPLES=<npz with arr_0 uint8 [N,256,256,3]> or a levelb sample folder}"
  REF="${REF:-adm}"
  if [ -d "$SAMPLES" ]; then
    # a levelb.sample folder: merge its shards (all of them, in order) into samples_10000.npz
    python - "$SAMPLES" <<'PY'
import pathlib, re, sys
import numpy as np
d = pathlib.Path(sys.argv[1])
out = d / "samples_10000.npz"
if out.exists():
    sys.exit(0)
shards = sorted(d.glob("shard*of*.npz"))
if not shards:
    sys.exit(f"{d}: no shard files; has sampling finished?")
totals = {int(re.match(r"shard(\d+)of(\d+)\.npz", p.name).group(2)) for p in shards}
if len(totals) != 1 or len(shards) != totals.pop():
    sys.exit(f"{d}: incomplete or mixed shard sets: {[p.name for p in shards]}")
parts = [np.load(p) for p in shards]
arr = np.concatenate([p["arr_0"] for p in parts])
labels = np.concatenate([p["labels"] for p in parts])
if arr.shape[0] != 10000 or labels.shape != (10000,):
    sys.exit(f"{d}: merged {arr.shape[0]} images, expected 10000")
np.savez(out, arr_0=arr, labels=labels)
print(f"  merged {len(shards)} shard(s) -> {out}")
PY
    SAMPLES="$SAMPLES/samples_10000.npz"
  fi
fi
case "$REF" in
  adm) REF_NPZ="$ADM_DIR/VIRTUAL_imagenet256_labeled.npz" ;;
  val) REF_NPZ="$LEVELB_ROOT/imagenet256/val_ref_50000.npz" ;;
  *) die "REF must be adm or val" ;;
esac
[ -f "$REF_NPZ" ] || die "$REF_NPZ missing"
[ -f "$SAMPLES" ] || die "$SAMPLES missing"
python - "$SAMPLES" "$CALIBRATE" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], mmap_mode="r") as z:
    a = z["arr_0"]
    if a.dtype != np.uint8 or a.shape[1:] != (256, 256, 3):
        sys.exit(f"samples must be uint8 [N, 256, 256, 3], got {a.dtype}{a.shape}")
    if sys.argv[2] != "1" and a.shape[0] != 10000:
        sys.exit(f"{a.shape[0]} samples; the Bifrost-1 protocol scores exactly 10,000")
    print(f"  {a.shape[0]} samples")
PY

OUT="${SAMPLES%.npz}.${REF}.metrics"
( cd "$ADM_DIR" && python guided-diffusion/evaluations/evaluator.py "$REF_NPZ" "$SAMPLES" ) \
  2>&1 | tee "$OUT.log"
python - "$OUT.log" "$OUT.json" "$CALIBRATE" "$(hostname)" "$CPU_MODEL" "$NCPU" <<'PY'
import json, re, sys
log, out, calibrate = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
node = {"host": sys.argv[4], "cpu": sys.argv[5], "threads": int(sys.argv[6])}
text = open(log).read()
keys = {"Inception Score": "IS", "FID": "FID", "sFID": "sFID",
        "Precision": "Precision", "Recall": "Recall"}
m = {}
for label, k in keys.items():
    hit = re.findall(rf"^{label}: ([0-9.eE+-]+)$", text, re.M)
    if not hit:
        sys.exit(f"evaluator output has no '{label}' line")
    m[k] = float(hit[-1])
json.dump({**m, "node": node}, open(out, "w"), indent=2)
print("  " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))
if calibrate:
    ref = {"IS": (215.837, 1.0), "FID": (3.9426, 0.05), "sFID": (6.1404, 0.10),
           "Precision": (0.8265, 0.005), "Recall": (0.5309, 0.005)}
    bad = {k: (m[k], v, tol) for k, (v, tol) in ref.items() if abs(m[k] - v) > tol}
    for k, (v, tol) in ref.items():
        print(f"  CALIBRATION {k}: {m[k]:.4f} vs published {v} (+-{tol})")
    if bad:
        sys.exit(f"CALIBRATION FAILED on {sorted(bad)} — do not use this evaluator")
    print("CALIBRATION PASSED — the evaluator reproduces ADM's published numbers.")
PY
[ "$CALIBRATE" = "1" ] && echo "$(hostname) | $CPU_MODEL" > "$BENCH_ROOT/ADM_CALIBRATED"
echo "=== metrics: $OUT.json ==="
