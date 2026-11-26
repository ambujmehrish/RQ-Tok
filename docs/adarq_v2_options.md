# AdaRQ-Flow after Step 0: what is broken, what the literature says, what to test next

Status 2026-11-26. Nothing in this document is queued. Validation results choose what to
build; only test-set results (Bifrost-1 protocol, 10,000 images × 3 seeds) decide who wins.

## 1. What we know (all on record in `EXPERIMENTS.md`)

Test set, ADM reference, 3 seeds:

| arm | FID | sFID | IS | P | R | top-1 |
|---|---|---|---|---|---|---|
| Bifrost-1 | 16.98 | 28.30 | 266.5 | 0.938 | 0.246 | 0.895 |
| AdaRQ-Flow, sampler F | 15.50 | 24.35 | 90.7 | 0.719 | 0.604 | 0.579 |

Validation (2,000 images), same renderer for every row:

| bridge input | top-1 | IS |
|---|---|---|
| real image, true latent | 0.762 | 179 |
| real image, its codes only, depth 4 / adaptive (2.0) / depth 1 | 0.658 / 0.589 / 0.556 | 132 / 98 / 93 |
| generated, AdaRQ-Flow F, residual noise 1 → 0 | 0.572 → 0.649 | 89 → 117 |
| generated, Bifrost-1 | 0.888 | 281 |

Latent error per element (held-out): Bifrost-1's prediction 1.72; the image's own codes 1.96
(depth 2) and 1.56 (depth 4). The two numbers come from different runs (Bifrost-1's is on
masked patches with up to 30% of the image visible), so treat the comparison as indicative.

Four findings follow.

1. **The gap is class fidelity only.** The IS gap is entirely per-image classifier
   uncertainty, spread over ~all classes.
2. **The tokenizer loses class information** (0.762 → 0.589 on the same images).
3. **Bifrost-1 is more class-typical than real images** (0.888 vs 0.762). Its regression
   mean produces prototypes. A perfect sampler of real latents would stop near 0.76.
4. **Our codes buy almost nothing over a class-level mean.** One code leaves 50% of the
   energy; three more (30 bits) remove only a quarter of what is left. A rough
   rate-distortion reading gives ~130 effective dimensions for the instance-specific part:
   on the order of 15 codes per patch to capture it, not 2–4.

So two things are missing, of similar size: codes (or anything) that carry the class, and a
mechanism that pushes samples toward typical examples.

## 2. What the literature says

Read from the papers unless marked; ids resolve at alphaxiv.org/abs/<id>.

**Guidance and typicality.**
- Unguided class-conditional token models score where we score; guided ones score where
  Bifrost-1 scores. MAR (2406.11838): cross-entropy variant IS 146 → 278, MAR-L 221 → 296.
  LlamaGen (2406.06525): L model IS 64 → 256, precision 0.61 → 0.83, recall 0.67 → 0.52.
  Our unguided IS 91 is inside that unguided range.
- Guidance without retraining exists. Independent condition guidance (2407.02687): replace
  the unconditional branch with the same network fed a random other condition, resampled
  each step. On DiT-XL/2 it matches true guidance (FID 5.50 vs 5.56, precision 0.83 vs 0.81).
  Shown for diffusion only; not validated on discrete or autoregressive models.
- Autoguidance (2406.02507) improves quality without sharpening the class; it would not test
  our class-fidelity gap.
- Confidence-ordered unmasking: no gain over random order in MAR (FID 8.72 vs 8.79).

**Tokenizers on semantic features.**
- Feature error is a poor proxy for semantic retention. FVQ (2610.37662): a quantizer with
  9× higher error scores higher on understanding benchmarks (60.7 vs 51.2). BEiT-v2
  (2208.06366): lower reconstruction loss with a deeper decoder, yet linear probe 78.5 → 63.0.
- Every semantic tokenizer decodes its codes through a learned network trained with a
  cosine or task loss; none feeds a sum of centroids downstream, as we do.
- Vocabulary: nobody uses fewer than ~13 bits per code (K = 8k–65k); we use 10. FVQ on
  frozen features: residual VQ at 28 / 56 bits reaches 59.8 / 63.0 against 64.6 continuous.
- Cosine lookup, removing the global mean, and projection residuals help post-hoc residual
  k-means like ours (2608.24207). Product quantization is no better than residual VQ on
  frozen features. Scalar/binary quantizers have no support for frozen semantic features.

**Mean plus generative residual.**
- Established elsewhere: CorrDiff (2309.15214; regression then diffusion on the residual),
  Whang et al. (2112.02475), PMRF (2410.00418; flow from the regressed mean), HART
  (2410.10812; codes plus residual diffusion). None found for MLLM-to-diffusion bridging in a
  CLIP-like space; BLIP3-o (2505.09568) compares MSE and flow as alternatives, never combined.
- Nobody uses "scale 0 recovers the regressor exactly" as a guarantee, or sweeps the scale.
- Known pitfalls: residual spread too small (CorrDiff); renderer not trained on the mixed
  inputs (HART needed alternating training: FID 21.98 vs 5.30 without); a scale on the
  residual is shrinkage, not guidance.

## 3. Options

### A. Guidance for the sampled bridge
Train with 10% prompt dropout; guide code logits and flow velocity; tune weight and
temperature on validation.
- For: the literature's standard fix, large documented effect on exactly our symptom.
- Against: not novel by itself; a 16-epoch retrain; does not repair the tokenizer.
- Cheap test first: independent condition guidance on the existing checkpoint (no training).
  A positive result is informative. A negative one is not conclusive: untested on token
  models, and our visible context already leaks the class.

### B. A tokenizer that carries meaning
Keep adaptive-rate codes, change how they are built and read:
1. a small learned decoder from the 64 patches' codes to latents (cosine + MSE), instead of
   summing centroids;
2. vocabulary 16k–65k per level; cosine lookup; global mean removed;
3. choose depth by semantic loss, not residual norm.
- For: keeps the paper's original idea (adaptive discrete interface); testable offline with
  no MLLM training (fit, decode, render real images, measure class accuracy).
- Against: best case is the true-latent level (0.76), still below Bifrost-1 (0.89) without
  option A; the learned decoder outputs an average given the codes, which partly is option C.

### C. Mean-anchored bridge
`z = μ(h) + α·δ`, with `μ` the regression head (Bifrost-1's, frozen) and `δ` a generative
head for the deviation, trained on `z − μ(h)`.
- For: α = 0 is Bifrost-1 exactly, so class fidelity cannot be lost by construction; α gives
  the diversity Bifrost-1 lacks (recall 0.25); only a head trains, so it is cheap.
- Against: conceptually close to CorrDiff/PMRF/HART, so novelty risk is moderate to high;
  the deviation is high-dimensional, so discrete codes are unlikely to help there and the
  "adaptive RVQ" part of the method would mostly disappear; needs the baselines
  "Bifrost-1 + noise" and "sampler + guidance" to mean anything.

### D. Stop
Write up the negative result plus the typicality finding (regression bridges score above
real images on class fidelity and collapse recall). Honest, small, workshop-level.

## 4. What a defensible ICML claim could look like

Not "beats Bifrost-1 on every metric": that means being more typical than reality. A claim
the data could support is a **fidelity–diversity frontier**: a bridge with one dial that
matches Bifrost-1 at one end and dominates simple baselines (Bifrost-1 + noise, temperature,
plain guidance) along the curve, reported on test with precision, recall, IS, FID and class
accuracy. The measurement that Bifrost-1 is beyond-real typical is ours and, as far as the
search went, new. Whether that clears the bar is a judgement call; A alone would not.

## 5. Proposed order (cheapest decisive test first)

Phase 1 — no training, validation only, H200 only, about 8 GPU-hours:
1. **Guidance on the existing checkpoint** (independent condition guidance), weights 1.5, 2, 3
   on sampler F. Decides whether option A is worth a retrain.
2. **Bifrost-1 + noise**, three noise levels. The baseline every later result is compared to;
   shows how fast naive noise destroys class fidelity.
3. **Latent statistics** on existing data: spectrum and effective dimension, energy explained
   by class means, class accuracy of a linear probe on true latents vs code-only latents.
   Replaces the back-of-envelope numbers in section 1.

Phase 2 — small offline training, no MLLM, about a day:
4. **Learned decoder over the existing codes**, then over a 16k-vocabulary refit; render real
   images; target: code-only class accuracy ≥ 0.72 (true latent: 0.76).

Gates:
- Guidance lifts top-1 to ≥ 0.75 with recall ≥ 0.5 → retrain with prompt dropout (A), on the
  better tokenizer if step 4 delivers one.
- Guidance does little and noise on Bifrost-1 costs class fidelity quickly → build C and
  require it to beat the noise baseline.
- Neither → D.

Phase 3 — one full training run of the chosen design, then test.

Constraint: H200 waits have reached 11 hours and a 16-epoch run takes several days, so there
is room for perhaps two or three full training runs before a late-January deadline. That is
why Phase 1 and 2 come first.
