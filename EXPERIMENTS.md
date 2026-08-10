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

## E0 — Headroom pilot. **Go / no-go for the entire thesis.**

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

**Result on real data (2026-08, `scripts/smoke_test.sh`).** 64 real COCO val2017 images
→ real CLIP-B/32 → 3136 patch latents, matched mean depth 2.5:
`oracle vs uniform +34.2%`, `oracle vs random +34.1%` → **PROCEED** (well clear of the
10% floor; the random control confirms the gain is content-aware). Note the shipped
threshold rule already captures ~100% of that headroom on latent-L2, so C3's learned
allocator must be judged on **downstream** distortion, not latent L2.

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
