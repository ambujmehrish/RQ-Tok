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
# Runs on an H200 node (user: H200 only) but computes on CPU: the evaluator is tensorflow-cpu. On an H200 node (job 1506465) its FID
# raised "Imaginary component 0.0646"; on a CPU node (adm_diag, job 1506499) and off-cluster
# the same lock, commit and batches reproduce FID 3.9426 / sFID 6.1404 with a real sqrtm.
# BLAS threads are pinned to the allocated CPUs; the node and CPU are recorded with every
# result, and calibration and scoring run on the same kind of node.
# =====================================================================================
#SBATCH --job-name=adm-score
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
# On the H200 nodes' Xeon 6760P, the OpenBLAS kernel numpy 1.23.5 selects by itself computes
# the covariance product wrongly (diagnostic job 1508122: sum 137135 vs 142402, then an
# imaginary sqrtm); the SkylakeX kernel is exact there (bit-identical product, FID 3.9426).
# Only for that verified CPU; any other model must calibrate as it is.
case "$CPU_MODEL" in
  *"Xeon(R) 6760P"*) export OPENBLAS_CORETYPE=SkylakeX ;;
esac
echo "  node $(hostname) | $CPU_MODEL | $NCPU threads | OpenBLAS core ${OPENBLAS_CORETYPE:-auto}"

CALIBRATE="${CALIBRATE:-0}"
if [ "$CALIBRATE" = "1" ]; then
  SAMPLES="$ADM_DIR/admnet_guided_upsampled_imagenet256.npz"; REF="adm"
else
  [ -f "$BENCH_ROOT/ADM_CALIBRATED" ] || die "evaluator not calibrated yet; run with CALIBRATE=1 first"
  # calibration holds only for the CPU type it ran on (the evaluator's linear algebra failed on
  # one node type before threads were pinned): refuse to score anywhere else
  CAL_CPU=$(sed 's/^[^|]*| //' "$BENCH_ROOT/ADM_CALIBRATED")
  [ "$CAL_CPU" = "$CPU_MODEL" ] || die "calibrated on '$CAL_CPU', this node is '$CPU_MODEL': run CALIBRATE=1 here first"
  SAMPLES="${SAMPLES:?set SAMPLES=<npz with arr_0 uint8 [N,256,256,3]> or a levelb sample folder}"
  REF="${REF:-adm}"
  # NSAMPLES: the protocol scores exactly 10,000; a smaller validation pilot (e.g. 2,000) may
  # only be scored against the held-out validation reference, never a test reference
  NSAMPLES="${NSAMPLES:-10000}"
  if [ "$NSAMPLES" != 10000 ] && [ "$REF" != heldout ]; then
    die "NSAMPLES=$NSAMPLES is a validation pilot: only REF=heldout is allowed"
  fi
  if [ -d "$SAMPLES" ]; then
    # a levelb.sample folder: merge its shards (all of them, in order) into samples_<N>.npz
    python - "$SAMPLES" "$NSAMPLES" <<'PY'
import pathlib, re, sys
import numpy as np
d, N = pathlib.Path(sys.argv[1]), int(sys.argv[2])
out = d / f"samples_{N}.npz"
if out.exists():
    sys.exit(0)
shards = sorted(d.glob("shard*of*.npz"))
if not shards:
    sys.exit(f"{d}: no shard files; has sampling finished?")
totals = {int(re.match(r"shard(\d+)of(\d+)\.npz", p.name).group(2)) for p in shards}
if len(totals) != 1 or len(shards) != totals.pop():
    sys.exit(f"{d}: incomplete or mixed shard sets: {[p.name for p in shards]}")
import zipfile
def header(path):
    """(shape, dtype) of arr_0 without loading it."""
    with zipfile.ZipFile(path) as z, z.open("arr_0.npy") as f:
        ver = np.lib.format.read_magic(f)
        rd = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
              else np.lib.format.read_array_header_2_0)
        shape, _, dtype = rd(f)
    return shape, dtype
heads = [header(p) for p in shards]
n = sum(h[0][0] for h in heads)
if n != N or len({(h[0][1:], h[1]) for h in heads}) != 1:
    sys.exit(f"{d}: shards hold {n} images (expected {N}) or differ in shape/dtype")
# stream shard by shard through a memmap: peak memory is one shard, not the whole set
tmp = d / "merge.tmp.npy"
mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=heads[0][1], shape=(n,) + heads[0][0][1:])
labels, pos = [], 0
for p in shards:
    with np.load(p) as z:
        a = z["arr_0"]
        mm[pos:pos + len(a)] = a
        pos += len(a)
        labels.append(z["labels"])
mm.flush()
labels = np.concatenate(labels)
part = d / f"samples_{N}.partial.npz"
np.savez(part, arr_0=mm, labels=labels)
part.replace(out)
del mm
tmp.unlink()
print(f"  merged {len(shards)} shard(s) -> {out}")
PY
    SAMPLES="$SAMPLES/samples_${NSAMPLES}.npz"
  fi
fi
# adm (primary) and val (secondary) are the test references; the others exist only to
# locate the Bifrost-1 paper's FID pipeline (scripts/bench/eval_search_report.py)
REFS="$LEVELB_ROOT/imagenet256/refs"
case "$REF" in
  adm) REF_NPZ="$ADM_DIR/VIRTUAL_imagenet256_labeled.npz" ;;
  val) REF_NPZ="$LEVELB_ROOT/imagenet256/val_ref_50000.npz" ;;
  adm64) REF_NPZ="$ADM_DIR/VIRTUAL_imagenet64_labeled.npz" ;;
  adm128) REF_NPZ="$ADM_DIR/VIRTUAL_imagenet128_labeled.npz" ;;
  adm512) REF_NPZ="$ADM_DIR/VIRTUAL_imagenet512.npz" ;;
  valsquash) REF_NPZ="$REFS/val_squash_50000.npz" ;;
  heldout) REF_NPZ="$REFS/heldout_10000.npz" ;;
  *) die "REF must be adm, val, adm64, adm128, adm512, valsquash or heldout" ;;
esac
[ -f "$REF_NPZ" ] || die "$REF_NPZ missing"
[ -f "$SAMPLES" ] || die "$SAMPLES missing"
# RESIZE=<px>: score a copy of the samples resized to <px> (antialiased bicubic, TF's
# resize, already in the pinned environment) — for the evaluation search on samples
# generated at another size (e.g. FLUX at 1024)
if [ -n "${RESIZE:-}" ] && [ "$CALIBRATE" != "1" ]; then
  python - "$SAMPLES" "$RESIZE" <<'PY'
import pathlib, sys
import numpy as np
import tensorflow as tf
src, r = pathlib.Path(sys.argv[1]), int(sys.argv[2])
out = src.with_name(f"{src.stem}.r{r}.npz")
if not out.exists():
    with np.load(src, mmap_mode="r") as z:
        a, labels = z["arr_0"], np.asarray(z["labels"])
        b = np.empty((len(a), r, r, 3), dtype=np.uint8)
        for s in range(0, len(a), 256):
            x = tf.image.resize(np.asarray(a[s:s + 256], dtype=np.float32), (r, r),
                                method="bicubic", antialias=True)
            b[s:s + 256] = np.clip(np.round(x.numpy()), 0, 255).astype(np.uint8)
    part = out.with_name(out.stem + ".partial.npz")
    np.savez(part, arr_0=b, labels=labels)
    part.replace(out)
print(f"  resized {src.name} -> {out.name}")
PY
  SAMPLES="${SAMPLES%.npz}.r${RESIZE}.npz"
fi
# SAMPLE_RES: the expected sample size (the protocol's 256 unless searching)
python - "$SAMPLES" "$CALIBRATE" "${SAMPLE_RES:-256}" "${NSAMPLES:-10000}" <<'PY'
import sys
import numpy as np
r, N = int(sys.argv[3]), int(sys.argv[4])
with np.load(sys.argv[1], mmap_mode="r") as z:
    a = z["arr_0"]
    if a.dtype != np.uint8 or a.shape[1:] != (r, r, 3):
        sys.exit(f"samples must be uint8 [N, {r}, {r}, 3], got {a.dtype}{a.shape}")
    if sys.argv[2] != "1" and a.shape[0] != N:
        sys.exit(f"{a.shape[0]} samples; expected {N} (the protocol scores exactly 10,000)")
    print(f"  {a.shape[0]} samples at {r} px")
PY

OUT="${SAMPLES%.npz}.${REF}.metrics"
( cd "$ADM_DIR" && python guided-diffusion/evaluations/evaluator.py "$REF_NPZ" "$SAMPLES" ) \
  2>&1 | tee "$OUT.log"
python - "$OUT.log" "$OUT.json" "$CALIBRATE" "$(hostname)" "$CPU_MODEL" "$NCPU" \
    "${OPENBLAS_CORETYPE:-auto}" <<'PY'
import json, re, sys
log, out, calibrate = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
node = {"host": sys.argv[4], "cpu": sys.argv[5], "threads": int(sys.argv[6]),
        "openblas_core": sys.argv[7]}
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
