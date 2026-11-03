# AdaRQ-Flow — Falsification Plan

Every claim in `NOVELTY.md` is paired here with an experiment that can **kill** it. The
rules below are pre-registered: they are decided now, before any number exists, so that
a disappointing result is a finding rather than a tuning target.

---

## 0. Ground rules

**R1 — Effect-size floor (no marginal gains).** A claim counts as supported only if the
improvement is **≥ 10% relative** on the primary metric **and** the gap exceeds
**2× the pooled seed standard deviation**. Anything smaller is recorded as *no effect*,
regardless of p-value. A 2% FID delta is not a result.

**R2 — Seeds.** n ≥ 3 seeds per arm; report mean ± std. Single-seed numbers are never
reported as outcomes.

**R3 — Matched controls.** Every comparison holds constant: trainable parameter count,
optimizer/schedule, training steps, data, and — critically for C3 — the **mean token
budget**. A win bought with more tokens or more parameters is not a win.

**R4 — Pre-declared direction.** Each claim states what result refutes it. Post-hoc
metric substitution is prohibited: the primary metric is fixed before the run.

**R5 — Ablations must bite.** Any ablation used as evidence must be covered by
`tests/test_ablations_effective.py`, which asserts the mechanism is actually removed.
(This rule exists because three ablations in this repo were previously silent no-ops
that would have manufactured null results.)

**R6 — Capability claims outrank metric claims.** C5/C6 assert things baselines cannot
do at *any* hyperparameter. These are pass/fail, not effect sizes.

---

## Protocol — follow Bifrost-1 exactly (decided 2026-11)

The baseline is **Bifrost-1** (Lin et al., arXiv 2508.05954, NeurIPS 2025), run under its
own experimental protocol. Conclusions are drawn **only** from held-out test-set results;
pilots (E0) decide nothing.

**What upstream released.** Code: `github.com/HL-hanlin/Bifrost-1` @ `3047de2` — inference
only (GenEval / DPG-Bench generation); no training code, data pipeline or ImageNet setup.
Weights: `hanlincs/Bifrost-1` — the Table 3 model (Qwen2.5-VL-7B, 256 visual tokens at
448 px, linear head, MSE, latent ControlNet with 4 double + 1 single FLUX.1-dev blocks).

### Level A — reference reproduction (released checkpoint, official script)

`scripts/bifrost1/setup.sbatch` then `scripts/bifrost1/generate.sbatch` (one H200 each).
Generation uses upstream defaults unchanged: seed 42, 4 images per prompt, MAR decoding
with one step per token (256), cosine mask schedule, `cfg_weight` 1.0, FLUX 28 steps,
ControlNet scale 0.7, guidance 3.5, 512 x 512 output. Targets (paper Table 3, 7B):
**GenEval 0.81, DPG-Bench 77.67, COCO-30k FID 34.35, MJHQ-30k FID 16.21.**
This anchors our evaluation stack; it is *not* a head-to-head (62M training images,
16 GB200 — not reproducible on 3 H200).

Deviations from upstream, each forced by a missing file and kept minimal:
- **A1** `bifrost/models/mar/utils.py` is imported but not shipped. `sample_orders` is
  reproduced from MAR (one random permutation per sample, NumPy global RNG, unseeded as
  upstream); `random_masking` is unused at inference and raises if called.
- **A2** DPG-Bench prompts are not shipped; taken from the benchmark's own repo
  (`TencentQQGYLab/ELLA` @ `3c228f1`, 1065 prompts).
- **A3** `av==14.3.0` in upstream `requirements.txt` no longer has files on PyPI, and
  14.4.0 has no Python 3.11 wheel; `av==14.2.0`, the nearest release with one, is
  installed instead. No upstream code imports `av`.
Upstream NumPy RNG is unseeded, so decoding orders vary run to run; reported as-is.

**Scoring stacks are calibrated before use.** GenEval (`djghosh13/geneval` @ `af4902f`,
mmdetection 2.x @ `e9cae2d`, Mask2Former Swin-S + CLIP ViT-L-14) is run with its own
evaluator and thresholds. Deviation **E1**: its environment pins PyTorch 1.12 / CUDA 11.3,
which cannot target the H200 (sm_90), and the evaluator asserts CUDA; the scorer is
rebuilt with torch 2.0.1+cu118 and mmcv-full 1.7.1 compiled for sm_90, all other pins kept.
Because the stack changed, it must first reproduce GenEval's published Stable Diffusion
v1.5 score (0.43, regenerated with GenEval's own generator and defaults) within ±0.02;
until that passes, no GenEval number is reported (`scripts/bench/geneval_score.sbatch
CALIBRATE=1`). DPG-Bench (ELLA's `compute_dpg_bench.py`, mPLUG-large VQA) follows the
same rule against its published SD v1.5 score.

### Level B — controlled head-to-head (paper §5.1, ImageNet 256 x 256)

The protocol the paper ran on a single GH200, which a single H200 matches. Both arms use
identical backbones, data, epochs and evaluation; only the bridge differs.

| item | Bifrost-1 (retrained by us) | AdaRQ-Flow |
|---|---|---|
| MLLM | Qwen2.5-VL-3B, frozen, + trainable vision branch (LMFusion copy of QKV/MLP/norms) | same |
| bridge | 64 continuous CLIP latents (224 px -> 8x8 post-merger), linear head, **MSE** | adaptive RVQ codes + flow-matching residual |
| masking | MAR, ratio ~ truncated N(1.0, 0.25) on [0.7, 1.0] | same |
| renderer | latent ControlNet (4 double + 1 single), FLUX.1-dev frozen, flow-matching loss | same |
| epochs | vision branch 16, ControlNet 2 | same |
| decoding | 64 steps (one per token), random order | same step budget |
| FLUX inference | 28 steps, scale 0.7, guidance 3.5 | same |
| test | 10K samples, classes uniform over 1k; FID / sFID / IS | same |

Paper reference for the baseline: **FID 25.77, sFID 53.67, IS 98.57** (Table 1).

Values the paper does not state, fixed **before** training and identical for both arms:
Qwen size (3B: a trainable 7B branch with Adam does not fit 141 GB; upstream's own
checkpoint paths name a `QwenVL3B ... 224res_64tokens` run); vision-branch optimizer
(upstream paths name `lr5e4_warm5k_step50k`); batch size; ImageNet class prompt; FID
reference statistics (ADM ImageNet-256 reference batch, as in MAR/xAR which the paper
follows). Each is recorded here, with its source, before the first training run.

**Faithfulness gate.** Our retrained Bifrost-1 must reach FID <= 30.9 (within 20% of
25.77). If it does not, the baseline is not faithful and no comparison is reported until
it is fixed.

**Decision rule.** Validation: a fixed 10K-image slice held out from ImageNet train, used
only for checkpoint selection. Test: the 10K-sample FID/sFID/IS evaluation, run once per
final checkpoint, 3 sampling seeds. AdaRQ-Flow is reported better only if FID improves by
>= 10% relative **and** by more than 2 sigma over the seeds (R1); anything else is
reported as no improvement. Marginal gains are not claimed.

---

## E0 — Headroom pilot. **Pre-flight check, NOT a decision gate.**

*Scope decision (2026-11):* E0 runs on CLIP ViT-B/32 at 224 px (49 patches) — a small
proxy, not the Qwen2.5-VL + FLUX bridge. It checks that the tokenizer, allocators and
measurement work end to end on real images. It does **not** decide the thesis; that is
decided only by the trained system on a held-out test set against the baselines (C1–C6).

*Run this before building the learned allocator, and before any A100 hours are spent on
C3.* The central thesis is that non-uniform bit allocation matters. If it doesn't, no
amount of allocator engineering will help.

**Setup.** Fit the tokenizer. Fix a mean depth `d̄`. Compare three allocation policies at
**identical mean rate**:

| Policy | Allocation |
|---|---|
| `uniform` | every patch gets depth `d̄` |
| `random` | per-patch depth drawn i.i.d., mean `d̄` |
| `oracle` | greedy: iteratively give the next code to the patch whose extra code most reduces **downstream** error (renderer output, not latent L2) |

**Measure.** Renderer-output quality (rFID / LPIPS) vs. mean rate.

**Kill criterion.** If `oracle` improves on `uniform` by **< 10% relative**, the
premise — that allocation matters — is false. **Stop. Do not proceed to C3.** Report the
negative result; it is a genuine finding about generative bridges.

**Configuration prerequisites (learned from the first run of E0).** Both are required or
the measurement is confounded:
- `tokenizer.include_zero_code = True` — otherwise extra codes can *increase* error
  (27.5% of patches) and adaptive depth is rewarded for avoiding that, not for allocating.
- `tokenizer.shared_codebook = False` — with a shared codebook, depth 1→4 buys only 1.7%
  (vs 26.1% per-depth), so no allocator can show a gain regardless of the thesis.

**Headroom is measured against the i.i.d. null, not against zero.** Even latents with no
semantic structure show non-zero oracle gain (measured +6.8%) from per-sample
quantization luck. Only gain *in excess of* the i.i.d. baseline is evidence for
content-aware allocation.

**Result on real data (2026-08, `scripts/smoke_test.sh`).** *Correction: these "64" images were 20 distinct photographs repeated (ISSUES.md F15); the effective n is 20 and the numbers below must be re-measured on distinct images before being cited.* 64 real COCO val2017 images
→ real CLIP-B/32 → 3136 patch latents, matched mean depth 2.5:
`oracle vs uniform +34.2%`, `oracle vs random +34.1%` → **PROCEED** (well clear of the
10% floor; the random control confirms the gain is content-aware). Note the shipped
threshold rule already captures ~100% of that headroom on latent-L2, so C3's learned
allocator must be judged on **downstream** distortion, not latent L2.

**Downstream distortion is now wired** (`--downstream`). The decoder is the *real
frozen CLIP tower suffix*: quantize at layer `L`, replay layers `L..end`, compare the
final CLIP embedding. Verified bit-exact against the true embedding (max|diff| = 0.0).
Because self-attention mixes patches, distortion is **non-separable** and the Lagrangian
oracle does not apply — `allocate_greedy_downstream` performs the honest greedy search
over actual decoder output (`O(budget x N)` decoder passes, so it is the expensive path).

**Status: UNDERPOWERED, not an outcome.** Two small runs disagreed sharply — headroom
+15.7% with latent-L2 capturing 66% (8 images), versus +45.4% with latent-L2 capturing
98% (6 images); the `random` control itself moved from -0.3% to +12.9%. At this sample
size the estimate is dominated by which images are drawn. The report now prints the
per-image spread and refuses to issue a verdict below 32 images. Per R2 this needs
>= 32 images and >= 3 seeds before any conclusion — including any claim that latent-L2
is or is not the wrong criterion.

**The powered run is `scripts/e0_downstream_64.sbatch`** — 64 distinct real images x
3 seeds, run sequentially on ONE vhpc01 H200, fully offline from the prefetched cache.
The same job also re-measures the latent-L2 E0 on those 64 distinct images (F15). It
aggregates across seeds and applies R1/R2 automatically: it refuses below 3 seeds,
reports UNDERPOWERED below 32 images, and returns INCONCLUSIVE when the 2-sigma lower
bound on headroom falls under the 10% floor. All six verdict branches are exercised.

**Result — downstream E0 pre-flight (2026-11, vhpc01 H200, job 1506210).** 64 distinct
COCO val2017 images x 3 seeds (seeds vary the tokenizer fit; the images are fixed),
CLIP ViT-B/32 quantized at layer 8, mean depth 2.49 / D_max 4, cosine distortion of the
final CLIP embedding:

| metric | mean | std |
|---|---|---|
| headroom (downstream oracle vs uniform) | **+31.1%** | 3.5% |
| latent-L2 oracle vs uniform | +22.4% | 2.8% |
| random vs uniform (control) | −0.1% | 1.4% |
| criterion gap (share of headroom latent-L2 misses) | 27.9% | 1.2% |

**Pre-flight passes** (the script prints PROCEED; read it as "the instrument works and shows headroom", not as a decision). The 2-sigma lower bound on headroom (24.1%) clears the 10% floor
(R1); n = 64 >= 32 images and 3 seeds (R2); random is indistinguishable from uniform, so
the gain is content-aware, not rate variance. Per-image gain spread is large (±18–25%),
so the image-level variance is tested separately by the 512-image rate sweep
(mean depth 1.5 / 2.5 / 3.0).
What this does **not** show: the oracle sees the true distortion (a learned policy will
recover only part of it); the decoder is the CLIP tower suffix, a semantic proxy, not
FLUX; and nothing here measures image quality. C3 still needs the real renderer.

**Latent-L2 re-measurement on 64 distinct images (F15).** Oracle vs uniform: +9.6%,
+10.1%, +10.5% over 3 seeds (mean 10.1% ± 0.5%; 2-sigma lower bound 9.2%), i.e. at the
10% floor. The earlier "+34%" came from 20 images repeated and is withdrawn. On latent
L2 alone the pre-flight is INCONCLUSIVE; the downstream measure shows the larger effect.
The per-seed lines also printed a threshold-rule gain (+11–12%, "117% of headroom")
that was **not rate-matched** — the threshold ran at its own rate, not the budget. That
report bug is fixed: the number is now shown only when rate-matched.

**512-image rate sweep (jobs 1506211–13, 3 seeds each, same-data fit — see F17).**

| mean depth | downstream headroom | latent-L2 oracle | random | criterion gap |
|---|---|---|---|---|
| 2.49 | +11.3% ± 6.5% | +3.3% ± 3.0% | +0.2% ± 0.6% | 73% ± 11% |
| 3.00 | +10.1% ± 4.0% | +2.4% ± 2.0% | 0.0% ± 0.0% | 78% ± 19% |

With 8x more images the downstream headroom fell from +31% to about +10–11%. It stays
positive at both rates and random stays at zero, so the signal is content-aware, but it is
noisy across tokenizer fits. **Known flaw in all of the above (F17):** the tokenizer was
fitted and scored on the same images, which favours small sets. E0 now holds out images.

**How these numbers are used.** E0 is a pilot on a small proxy and decides nothing — small
sets can swing either way (+31% at 64 images, ~+10% at 512). Conclusions about the thesis
come only from the trained system on the held-out test set, against the Bifrost-1 baseline
under Bifrost-1's own protocol.

**Interpretation guard.** `random` is the load-bearing control. If `random ≈ oracle`,
the gain comes from *rate variance*, not from *content-aware* allocation, and the
"adaptive" claim collapses even if the oracle beats uniform.

---

## Claims

### C1 — A hybrid discrete+continuous interface beats a purely continuous one

- **Arms:** `baseline` (codes+flow) vs `continuous_flow` (no codes, flow).
- **Control:** identical head module and parameter count; identical decode step count.
- **Primary metric:** FID. **Secondary:** GenEval.
- **Refuted if:** `continuous_flow` is within 10% of `baseline` on FID.
- **Confound to rule out:** the discrete arm also gets confidence-ordered decoding.
  Re-run `baseline` with random reveal order to separate "codes" from "code-derived
  decode ordering".

### C2 — The distributional objective, not head capacity, is what matters

- **Arms:** 2×2 factorial — {codes, no codes} × {flow, mse}:
  `baseline`, `continuous_flow`, `hybrid_mse`, `continuous_mse`.
- **Control:** all four use the **same head module** (`FlowResidualHead`), so parameter
  count is identical by construction; only the objective and sampler differ.
- **Primary metric:** sample **diversity** (mean pairwise LPIPS across samples per fixed
  prompt) — a sharper probe of mode-averaging than FID. **Secondary:** FID.
- **Refuted if:** the mse arms match the flow arms on diversity. That would mean the
  mode-averaging argument is wrong for this interface.

### C3 — Content-adaptive rate dominates fixed rate **at equal mean budget**

This is the central claim. It is also the easiest to accidentally fake.

- **Arms:** `fixed_depth` at `D ∈ {2,4,8}` vs adaptive, **tuned so the adaptive arm's
  measured mean depth equals the fixed arm's `D`** (verify the realized mean, do not
  assume it).
- **Negative control:** `random` allocation at the same mean rate (from E0).
- **Primary metric:** the **rate–quality frontier** (FID vs mean tokens/image), not a
  point estimate.
- **Supported only if:** adaptive dominates fixed across the frontier — at least 3
  operating points, each clearing R1 — **and** beats `random` at equal rate.
- **Refuted if:** adaptive matches fixed at equal mean rate, **or** matches `random`.
  Either outcome means the halting criterion carries no content information.
- **Known risk:** the current halting rule thresholds *reconstruction* residual norm,
  which `NOVELTY.md` §6 argues is the wrong signal. A negative result here is
  informative about **the criterion**, not necessarily about the thesis — but it must be
  reported as a refutation of the shipped mechanism.

### C4 — Training the renderer on the code space removes a real train/inference gap

- **Arms:** `baseline` vs `no_exposure_fix` (`train_on_dequantized=False`).
- **Mechanism check (already automated):**
  `tests/test_ablations_effective.py::test_exposure_flag_changes_renderer_conditioning`.
- **Primary metric:** FID gap between renderer-fed-ground-truth-latents and
  renderer-fed-model-latents. The claim is specifically that this *gap* shrinks.
- **Refuted if:** the gap is unchanged, i.e. the mismatch was never the bottleneck.

### C5 — Inference-time rate control *(capability, pass/fail)*

- **Test:** one trained checkpoint, decoded at ≥ 4 distinct token budgets, produces a
  monotone rate–quality curve without retraining.
- **Fails if:** quality is non-monotone in rate, or budgets cannot be varied at decode
  time.
- **Why it matters:** no fixed-rate interface can do this at any setting. Not an effect
  size — a capability that exists or doesn't.

### C6 — Understanding is preserved *exactly* *(capability, pass/fail)*

- **Test:** the frozen backbone's outputs must be **bit-identical** with the generation
  branch attached vs. detached. Assert tensor equality, do not compare benchmark scores.
- **Fails if:** any deviation. This is a structural property of the architecture; an
  approximate result means something is leaking.

---

## Confound register

Checked explicitly, because each could manufacture a false positive:

1. **Token budget** — adaptive arms must not spend more tokens on average (C3).
2. **Parameter count** — flow vs mse share one module (C2); code-head size varies with
   `D_max`, so `depth_*` arms must be parameter-compensated elsewhere or reported with
   the discrepancy stated.
3. **Decode compute** — `steps` controls unmask rounds; ODE integration steps are a
   *separate* axis. Report both; do not let one arm silently get more NFEs.
4. **Loss balance** — `flow_loss_weight` must be tuned per scale. At `base_gpu` the raw
   CE and flow magnitudes differ ~100×; an untuned run trains one head only, and any
   ablation on the starved head is uninterpretable.
5. **Codebook usage** — report a **usage histogram over eval data**, not the EMA
   `cluster_size` statistic, which is inflated toward uniform by construction
   (see `ISSUES.md`). A collapsed codebook can mimic "adaptive depth".
6. **Renderer target realism** — the CPU stand-in regresses toward a *randomly
   initialized* projection. Loss curves from that configuration measure optimizer
   plumbing, not generative quality, and must never be cited as evidence.

---

## What would make us abandon the direction

Stated in advance, so it is a decision rather than a negotiation:

- **E0 headroom < 10%** → allocation doesn't matter for this bridge. Stop.
- **E0 `random` ≈ `oracle`** → content-awareness doesn't matter; at most a rate-variance
  story remains, which is not worth a paper.
- **C3 refuted with a learned allocator** (not merely the shipped threshold) → the
  central claim is false. Publish the negative result: "adaptive rate allocation does not
  help generative bridges" is a useful finding.
- **C1 and C2 both refuted** → the hybrid interface has no advantage; the remaining
  contribution is the capability claims (C5/C6) alone, which is a workshop note, not a
  paper. Say so rather than inflating it.
