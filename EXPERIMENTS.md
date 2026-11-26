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

### Test vs validation — binding rules (decided 2026-11)

Every reported number follows Bifrost-1's protocol and comes from a **test** set; nothing
is reported from validation data.

| | role | Level A | Level B |
|---|---|---|---|
| **test** | the only source of reported results; each final model scored once per seed | GenEval (553 prompts), DPG-Bench (1065 prompts) — the benchmarks' official evaluation sets | Bifrost-1 §5.1: 10K samples, classes uniform over 1000, FID / sFID / IS against the ADM ImageNet-256 reference batch. Secondary (ours, reported separately): FID against the 50,000 ImageNet **validation** images, which no stage ever trains on |
| **validation** | training dynamics and hyper-parameter tuning only; never reported | — (no training) | the fixed 10,000-image slice held out of ImageNet train: held-out losses during training, and any setting the paper and upstream leave open (e.g. the ControlNet micro-batch if 48 does not fit) |
| **train** | fitting only | — | ImageNet train minus the validation slice |

Settings the paper or upstream fixes are used as given; validation may only choose what
they leave open, and any such choice is recorded here before test scoring. The test runs
on the final checkpoint at the fixed training budget, identically for both arms.

**Winning and losing are decided on the test set only.** A validation pilot may choose a
setting to carry forward; it never shows that one method beats another. In particular, the
2,000-image pilot FIDs (held-out reference) are small-sample numbers: AdaRQ-Flow F's 32.04 vs
Bifrost-1's 29.83 there does not mean the two are close. Their IS (89 vs 281), precision
(0.71 vs 0.94) and class top-1 (0.57 vs 0.89) show a large gap, and on the test set the
default AdaRQ-Flow sampler trails by 11 FID (28.38 vs 16.98).

**Level A result — GenEval (job 1506427): Bifrost-1 reproduced, 0.804 vs paper 0.81.**
Released checkpoint, official script, calibrated scorer. By task: single object 0.997,
two object 0.922, counting 0.713, colors 0.838, position 0.718, color attribution 0.638
(the paper reports only the overall score).

**Level A result — DPG-Bench (job 1506473): Bifrost-1 reproduced, 77.88 vs paper 77.67.**
Scorer calibration first (job 1506472): SD v1.5 61.42 vs ELLA's published 63.18 (band
±2.0) — PASSED. Both over ELLA's 1064 scorable prompts (D2). **Level A is complete:** the
released Bifrost-1 checkpoint with its official script reproduces both published
benchmark numbers (GenEval 0.804 / 0.81, DPG-Bench 77.88 / 77.67) on our scorers.

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
- **A4** `protobuf` is unpinned upstream and today resolves to a release that rejects the
  generated code in the pinned `sentencepiece==0.1.99`, breaking FLUX's T5 tokenizer;
  `protobuf==3.20.3` is pinned, with the also-unpinned `google-api-core==2.24.2` and
  `googleapis-common-protos==1.69.2` (March 2025, the upstream era) that accept it.
Upstream NumPy RNG is unseeded, so decoding orders vary run to run; reported as-is.
A benchmark that cannot finish inside the 24 h job limit is generated in shards: the
unmodified script runs from a work dir holding a contiguous slice of the official
prompt files (`scripts/bifrost1/shard_prompts.py`), and `merge_shards.py` rebuilds
the full layout, checking every GenEval folder against the official metadata line.
FLUX is re-seeded with the same seed for every prompt, so sharding changes no image.

**Scoring stacks are calibrated before use.** GenEval (`djghosh13/geneval` @ `af4902f`,
mmdetection 2.x @ `e9cae2d`, Mask2Former Swin-S + CLIP ViT-L-14) is run with its own
evaluator and thresholds. Deviation **E1**: its environment pins PyTorch 1.12 / CUDA 11.3,
which cannot target the H200 (sm_90), and the evaluator asserts CUDA; the scorer is
rebuilt with torch 2.0.1+cu118 and mmcv-full 1.7.1 compiled for sm_90. Only the
packages the evaluator imports are installed, at GenEval's own pins
(`scripts/bench/geneval_requirements.txt`); `clip-benchmark` 1.4.0 declares torch<2 and
is installed without its dependency check (its zero-shot colour classifier was verified
on torch 2.0); `pytorch-lightning` 1.9.5 replaces 1.4.2 for the calibration generator.
Because the stack changed, it must first reproduce GenEval's published Stable Diffusion
v1.5 score (0.43, regenerated with GenEval's own generator and defaults) within ±0.02;
until that passes, no GenEval number is reported (`scripts/bench/geneval_score.sbatch
CALIBRATE=1`).

**GenEval calibration result (vhpc01 H200, job 1506374): PASSED.** SD v1.5 regenerated with
GenEval's generator and defaults, scored by the rebuilt evaluator:

| task | ours | GenEval paper |
|---|---|---|
| single object | 0.953 | 0.97 |
| two object | 0.384 | 0.38 |
| counting | 0.366 | 0.35 |
| colors | 0.758 | 0.76 |
| position | 0.035 | 0.04 |
| color attribution | 0.045 | 0.06 |
| **overall** | **0.423** | **0.43** |

Every task is within 0.02 of the published value, so deviation E1 does not change the scorer.

**Bifrost-1 reference generation (Level A).** GenEval: 553 prompts x 4 images, job
1506357, 17.6 h on one H200 (~115 s per prompt). DPG-Bench: 1065 prompts in 4 shards,
jobs 1506375-78, 8.6-8.9 h each.

DPG-Bench uses ELLA's own scorer (`TencentQQGYLab/ELLA` @ `3c228f1`, mPLUG-large judge),
imported unmodified. Its requirements are unpinned and today resolve to versions that
cannot load mPLUG, so the environment is a lock file verified to run the judge correctly
(`scripts/bench/dpg_requirements.lock.txt`: modelscope 1.13.3, transformers 4.38.2,
datasets 2.18.0, fairseq 0.12.2 — the benchmark's release era; torch 2.5.1+cu124).
Deviation **D1**: ModelScope 1.13 can no longer download from today's hub, so mPLUG is
downloaded by a current ModelScope CLI and loaded from disk; nothing else changes. The
scorer silently skips images that raise, so every run must score all grids or fail —
with one exception that is ELLA's own: its `prepare_dpg_data` skips CSV row 0
(`if i == 0: continue`), proposition 1 of `partiprompts97`, so that prompt raises
`KeyError(1)` for every model and the official script averages **1064** prompts. That
exclusion is reproduced, not patched (D2); any other failure stops the job.
Calibration: SD v1.5 must reproduce ELLA's published 63.18 (Table 4) within ±2.0 —
wider than GenEval's band because ELLA does not publish its SD v1.5 sampling settings
(diffusers defaults: 512 px, 50 steps, guidance 7.5). Bifrost-1's own DPG grids are scored
exactly as its script saves them (2x2 grid with 2-px padding, scored at 512-px tiles).

**ImageNet-256 scorer (Level B test).** OpenAI's ADM evaluation suite
(`openai/guided-diffusion` @ `22e0df8`, `evaluations/evaluator.py`, unmodified, with its
Inception-v3 graph and the ImageNet-256 reference batch) — the suite that computes sFID,
as reported by Bifrost-1 §5.1. Its requirements say `tensorflow-gpu>=2.0`; the environment
is a lock verified here: tensorflow-cpu 2.15.1, numpy 1.23.5 (numpy >= 1.24 removed `np.bool`,
which the evaluator's precision/recall uses), oneDNN off; two runs on the same inputs are
bit-identical. Calibration (`scripts/bench/adm_score.sbatch CALIBRATE=1`) must reproduce the
README's published ADM-G + ADM-U numbers — IS 215.837 (±1.0), FID 3.9426 (±0.05), sFID
6.1404 (±0.10), Precision 0.8265 (±0.005), Recall 0.5309 (±0.005) — before any model is
scored. Secondary test reference: the 50,000 ImageNet validation images, same 256 crop as
training (`levelb/val_ref.py`).
First calibration attempt (job 1506465, on an H200 node): IS 215.837158 matched, then FID
raised "Imaginary component 0.0646" inside the evaluator. Diagnosis (`scripts/bench/adm_diag.*`):
the same lock, commit and official batches reproduce the published FID/sFID with a real
sqrtm off-cluster (3.94257 / 6.14043; scipy 1.15.3, 1.11.4, 1.10.1 identical; every OpenBLAS
kernel type and thread count tried) and on a vhpc CPU node (job 1506499: IS 215.83711, FID
3.94257, sFID 6.14043). The H200-node failure was not reproduced elsewhere; the most likely
cause is BLAS threading on that node, not proven. Scoring therefore runs CPU-only (the
evaluator is tensorflow-cpu) with BLAS threads pinned to the allocated CPUs, the node and
CPU are recorded with every result, and the full five-number calibration is rerun on that
setup before any model is scored.

**ADM calibration result (vhpc01 CPU partition): PASSED.**

| metric | published (ADM-G + ADM-U) | ours | tolerance |
|---|---|---|---|
| IS | 215.837 | 215.8371 | ±1.0 |
| FID | 3.9426 | 3.9426 | ±0.05 |
| sFID | 6.1404 | 6.1404 | ±0.10 |
| Precision | 0.8265 | 0.8256 | ±0.005 |
| Recall | 0.5309 | 0.5288 | ±0.005 |

FID, sFID and IS — the Bifrost-1 §5.1 metrics — match to the reported precision.
Precision/recall (not part of the protocol) are within tolerance.

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

**Level B implementation spec (fixed before any training; source in brackets).** The
baseline trains upstream's own released modules — `MultiModalityCausalLM` (its training
`forward` and MSE loss), `Conversation.t2i_prompt`, `FluxControlNetModel` and
`FluxControlNetPipeline` from `HL-hanlin/Bifrost-1@3047de2` — inside a loop we write,
because upstream released no trainer. AdaRQ-Flow uses the same modules and changes only
the bridge (head + loss + sampler), so the comparison isolates the bridge.

*Data.* ImageNet-1k train `[paper §4]`; a fixed, seeded 10,000-image slice of train is
held out for monitoring only `[ours]`. Prompt = the first English name of the
class from the dataset's `classes.py` `[ours: unstated]`. Image: resize short side to
256 (bilinear) + centre crop 256 → FLUX VAE input `[diffusers train_controlnet_flux.py]`;
the same crop resized to 224 → Qwen2.5-VL visual tower → 8x8 = 64 post-merger latents of
width 2048 `[upstream path name 224res_64tokens; paper §5.1 uses 64 tokens]`; the
224 view is the 256 crop resized bicubically, so the latent grid is aligned with the FLUX
token grid. No flip. The frozen tower's latents are computed once
(`levelb/precompute.py`, fixed batch 32 in dataset order — upstream's visual tower
uses eager attention over one block-diagonal matrix per batch, memory ~ batch^2, and
upstream ships window tables for exactly bsz 32 at 224 px) and fed to upstream's forward
as `image_clip_embs`, which it accepts; both arms and the ControlNet read the same file.

*Stage 1 — vision branch.* Qwen2.5-VL-3B frozen; branch = per-layer copies of
q/k/v/o, MLP, both RMSNorms and the final norm, **initialised from the MLLM's weights**
`[paper §3.2; upstream add_vision_* creates them randomly — the copy is ours, required by
the paper]`; plus mask token, learnable position embedding and linear head. Mask ratio ~
truncated N(1.0, 0.25) on [0.7, 1.0], random order `[paper §3.2]`; MSE over all 64
positions `[upstream forward]`. lr 5e-4, 5k warmup `[upstream path name]`, batch
128 `[upstream path name t2i128]` — on one H200 as 4 accumulated micro-batches of 32
(exact: one sort and one mask draw per batch, each part weighted by its share; for
AdaRQ-Flow the sums are divided by the whole batch's target counts), since 128 in one pass
does not fit 141 GB — 16 epochs `[paper §5.1]`; optimiser and schedule are
the HF `Trainer` defaults implied by upstream's `TrainingArguments`: AdamW (0.9, 0.999),
eps 1e-8, no weight decay, linear decay after warmup, grad-norm clip 1.0, bf16 autocast
`[upstream uses transformers.TrainingArguments]`. The held-out slice is evaluated (fixed
subset, fixed masks) for monitoring only; the test uses the FINAL checkpoint at the fixed
budget, the same rule for both arms, so held-out data cannot favour either.

*Stage 2 — latent ControlNet.* `FluxControlNetModel.from_transformer` with 4 double +
1 single blocks; Qwen latents enter through upstream's `NHWCUpsampleBlock` (2x
transposed conv: the 8x8 bridge grid -> FLUX's 16x16 token grid) and a zero-initialised
linear `[upstream controlnet_flux.py]`, sized for 2048-d instead of upstream's hardcoded
3584 (7B) `[forced by 3B]`. Trained on ground-truth Qwen latents (decoupled training
`[paper §3.3]`), loss = MSE(v, noise − x0), logit-normal timesteps (0, 1), guidance
input 3.5, AdamW lr 5e-6, betas (0.9, 0.999), wd 1e-2, eps 1e-8, grad clip 1.0, constant
lr with 500 warmup steps, T5 at 512 tokens `[diffusers defaults, paper §4]`, batch 48
`[paper §4]`, 2 epochs `[paper §5.1]`. Notes on the diffusers defaults: its
`get_scheduler("constant")` ignores the 500 warmup steps, so the lr is constant from step 0;
its loss is unweighted (logit-normal only samples timesteps); `mixed_precision` is bf16 with
an fp32 ControlNet (accelerate's behaviour); gradient checkpointing is on (memory only).
The 1000 class prompts are encoded once with FLUX's own `encode_prompt` (frozen encoders,
so exact). The builder refuses if FLUX has fewer blocks than requested, since upstream's
`strict=False` copy would otherwise leave them random. **One ControlNet serves both arms**: both bridges
emit latents in the same Qwen space, so the renderer is held fixed and only the bridge
differs.

*Sampling.* MAR decoding with one step per token (64), cosine schedule, random order,
guidance `uncond + 1.0 * (cond - uncond)` `[upstream inference, ported line for line in
levelb/sample.py]`. Upstream's guidance slices the flattened predictions per step, which
is valid only if exactly one token is revealed per step; simulated for 64 and 256 tokens
and asserted on every batch. Upstream leaves NumPy unseeded; each batch is seeded from
(seed, index) so sample sets are reproducible; FLUX 28 steps, ControlNet scale 0.7, guidance 3.5 `[paper §4]`;
10,000 samples with classes uniform over 1,000 `[paper §5.1]`; scored against the ADM
ImageNet-256 reference batch (FID, sFID, IS).

*AdaRQ-Flow arm (same backbone, branch, loop; only the bridge differs).*
`levelb/adarq_tokenizer.py`, `levelb/adarq_model.py`, `levelb/train_adarq.py`.
- **Stage 0:** adaptive RVQ on the precomputed Qwen latents. Per-depth codebooks of K = 1024
  with a reserved zero codeword, D_max = 4, EMA 0.99, dead-code reinit below 0.5, fitted at
  full depth on the train split (20k steps x 64 images). The halting threshold is chosen on
  the validation slice for a **mean depth of 2.0** — a rate choice fixed here, not tuned for
  quality. Codes are prefix-nested, so the threshold only truncates them.
  **Result (vhpc01):** tau 0.76055 gives validation mean depth 2.000 (patches at depth
  1/2/3/4: 132355 / 46359 / 22217 / 55069), mean relative residual 0.699, residual RMS
  1.4013. Files: `codes_train.npy`, `tokenizer.pt`, `tokenizer.json` in
  `$LEVELB_ROOT/adarq_tok`; `train_adarq.sbatch` reuses them.
- **Heads on upstream's hidden states** (`branch_hidden` reproduces upstream's training forward
  exactly; upstream's MSE loss is recovered bit for bit): a depth-autoregressive code head,
  `W_k LN(h + P(zhat^(k)) + e_k)` over K + 1 symbols, no halt at level 1; a flow-matching residual
  head (AdaLN MLP, width 1024, 3 blocks) on the residual normalised by its RMS, conditioned on
  `h + Q(zhat)`, 4 timesteps per token. Loss = CE (masked patches, levels up to the first halt)
  + flow loss. Upstream's linear head is unused and frozen; head sizes are logged against it.
- **Training:** identical to the Bifrost-1 arm (seed, data order, masks, optimiser, schedule,
  budget, held-out monitoring, final checkpoint).
- **Sampling:** the same MAR loop. The revealed patch's codes are sampled depth by depth from
  guided logits (guidance on raw logits, then halt forbidden at level 1), then its residual by
  32 Euler steps of the guided velocity; guidance weight 1.0 as in the Bifrost-1 arm. The same
  ControlNet renders both arms.

**Training observation (held-out slice; monitoring only) and the second AdaRQ-Flow run —
declared before any test-set number exists.** The Bifrost-1 branch finished its 158,880
steps with held-out MSE falling throughout (2.047 at 5k → 1.717 at the end). The
AdaRQ-Flow code head overfits: held-out code cross-entropy reaches its minimum 2.365 at
step 75,000 (~7.5 epochs) and rises to 2.501 by 135,000, while training cross-entropy keeps
falling (1.944 at 137k); from ~70k the held-out value worsens right after each epoch
boundary (steps ≈ 9,930·n) and recovers mid-epoch, the signature of memorising training
images. The flow (continuous residual) loss keeps improving (1.211 → 1.129). Only the
latest checkpoint is kept, so no step-75k weights exist.

Decision (user, 2026-11-10), fixed now:
1. **Primary, as pre-registered:** both arms tested at the final checkpoint of the 16-epoch
   budget. Unchanged.
2. **Second, validation-tuned AdaRQ-Flow run:** identical in everything (seed, data order,
   masks, optimiser, 5k warmup, linear decay, micro-batching, held-out monitoring) except
   the budget, `--total-steps 75000` — the held-out code-CE minimum of run 1 — so the
   schedule decays to zero there and the step-75,000 final checkpoint is tested.
   Selected from the held-out slice only (the user's rule: validation for
   hyperparameter tuning). Output `branch_adarq_tuned`, samples `adarq_tuned_seed{0,1,2}`.
   The Bifrost-1 arm needs no such run: its held-out loss improved to the end of its
   budget. Both AdaRQ-Flow results are reported, each against the same Bifrost-1 result.

**Overfitting diagnostic (job 1507688, `levelb/adarq_diag.py`; final 16-epoch weights;
2,048 held-out + 2,048 train images; reproduces the stored codes 1.0000).**

H1 — memorisation through the visible tokens. Code CE (all targets), train vs held-out:

| masked | train | held-out | gap |
|---|---|---|---|
| 0.70 | 1.4582 | 2.4280 | +0.970 |
| 0.80 | 1.5429 | 2.4808 | +0.938 |
| 0.90 | 1.7193 | 2.5434 | +0.824 |
| 1.00 | 3.1491 | 3.1963 | +0.047 |

The gap vanishes when nothing is visible: the head recognises training images from the
visible latents and recalls their codes. Confirmed.

H2 — the hard code labels are arbitrary near ties (held-out, mask 0.90): the tokenizer's
own choice is a near-tie at 74.1% of positions (2nd-nearest within 5% of the nearest
distance; 87.9% within 10%); 50.5% of the head's wrong top-1 predictions are within 10% of
the true code's distance (median ratio 1.098); the head already puts 51.7% of its
probability on codewords within 10%; top-1 accuracy 0.236. In 2048-d, distances to the 1,024
codewords crowd together, so cross-entropy on the index rewards recalling an arbitrary
per-image tie-break. Confirmed. Median d2²−d1² per level on train: 214.5, 61.4, 27.5, 18.2.

3. **Third AdaRQ-Flow run, soft code targets (declared now, before any test number):**
   identical to run 1 (16 epochs, final checkpoint) except the code loss: each code target
   becomes q(j) = softmax_j(−‖r − c_j‖² / T_k) over the K codewords, r the residual before
   level k; `<halt>` targets stay hard; the flow loss, sampling and everything else are
   unchanged. T_k = median(d2² − d1²) at level k over 1,024 train images (fixed seed),
   measured by the trainer and recorded in its config — a rule fixed here from the train
   split, not tuned on held-out or test. Output `branch_adarq_soft`, samples
   `adarq_soft_seed{0,1,2}`. Both soft and hard held-out code CE are logged; the diagnostic
   is rerun on its final weights. All three AdaRQ-Flow results are reported.

**Test results — seed 0 of 3 (10,000 samples, classes uniform; calibrated ADM evaluator).**

| arm | reference | FID | sFID | IS | Precision | Recall |
|---|---|---|---|---|---|---|
| Bifrost-1 (retrained) | ADM (primary) | 16.9908 | 28.2704 | 270.2169 | 0.9388 | 0.2424 |
| AdaRQ-Flow, 16 epochs | ADM (primary) | 28.0659 | 28.5093 | 56.4303 | 0.6163 | 0.6422 |
| Bifrost-1 (retrained) | ImageNet val (secondary) | 19.2532 | 31.5077 | 270.2169 | 0.9026 | 0.2395 |
| AdaRQ-Flow, 16 epochs | ImageNet val (secondary) | 24.6369 | 31.7164 | 56.4303 | 0.5618 | 0.6469 |

- Faithfulness gate: passed (16.99 ≤ 30.9). Our baseline is better than the paper on every
  metric (paper: FID 25.77, sFID 53.67, IS 98.57), most of all IS; some unstated protocol
  detail differs (prompt template, evaluation details). The head-to-head is unaffected: both
  arms share everything but the bridge.
- AdaRQ-Flow (16 epochs, as pre-registered) is worse on the primary FID by 65% (+28% on
  the secondary reference); sFID is equal within 1%. It has far higher recall (0.64 vs
  0.24) and far lower precision and IS: more diverse, less class-faithful samples. The
  regression bridge outputs the conditional mean latent (prototypical, low-diversity
  images); the sampled bridge draws codes and a flow residual. Consistent with the code
  head's overfitting (diagnostic above).
- Not yet decisive: seeds 1–2 and the two declared variants (75k steps; soft code
  targets) are pending. Both arms sample with guidance 1.0 and temperature 1.0 (upstream's
  inference); for a sampled bridge these set the quality/diversity trade-off. Any tuning
  of them would use a validation FID on the held-out slice, for both arms, never a test
  reference — not done.

**Reproduction concern (open).** The baseline is far *better* than the paper (FID 16.99 vs
25.77, sFID 28.27 vs 53.67, IS 270 vs 98.6). The one-sided gate passes, but a baseline that
beats the published number by this much means our setup is not the paper's, so the
comparison would not be against Bifrost-1 as published. Checked: no real data reaches the
sampler (both arms start from mask tokens; dataset latents are never read); upstream's own
code gives FLUX the text prompt in ControlNet training and in inference, as we do.
Calibration point that needs no training: the paper's Table 1 row "FLUX.1-dev" (the
backbone alone) is FID 51.20, sFID 224.82, IS 157.46, and the paper states that Bifrost-1
lowers IS relative to it. `levelb.sample --arm flux` generates that row under our protocol
(same class draw, class prompt, 256 px, 28 steps, guidance 3.5, same scorer). If it
reproduces 51/157, our protocol matches and the gap is in training; if ours is far better
too, the difference is in the protocol (prompt to FLUX, resolution, evaluation), to be
found by reproducing that row before the baseline is rerun under it.

**Result (FLUX-only, seed 0, ADM reference): FID 24.8120, sFID 52.1073, IS 156.5616,
Precision 0.8710, Recall 0.1838** (paper: FID 51.20, sFID 224.82, IS 157.46).
- Generation matches: IS needs no reference and depends only on the samples; ours is
  within 0.6% of the paper's. Prompt, resolution and FLUX settings reproduce theirs.
- Scoring does not: for the same sample distribution the paper's FID is ~2.1× ours and
  its sFID ~4.3×. The paper's FID/sFID pipeline (reference set or implementation; not
  stated) is on a different scale from the calibrated ADM evaluator, so absolute FID/sFID
  cannot be compared with the paper.
- Normalised by the FLUX anchor, our Bifrost-1 is not "too good": FID ratio
  Bifrost/FLUX is 0.685 ours vs 0.503 paper (sFID 0.543 vs 0.239).
- IS is a real difference, from training: both FLUX anchors ≈ 157; the paper's Bifrost-1
  lowers IS (98.6), ours raises it (270.2, precision 0.94, recall 0.24): our trained
  bridge drives FLUX toward prototypical class images.
- The pre-registered gate (FID ≤ 30.9) assumed the paper's scale and cannot be applied as
  written. An anchor-normalised version (FID ratio within 20% of 0.503, i.e. ≤ 0.604) is
  not met (0.685). Decision on the gate: pending (user). The head-to-head is unaffected
  (same generation, same scorer); AdaRQ-Flow seed 0 (FID 28.07, IS 56.4) is below the
  FLUX-only anchor on both.

**Hypothesis for the paper's FLUX row (stated before testing).** Upstream generates at
`height = t2i_resolution × 16/14` (the bridge's token grid: 224 → 256 px, as ours), but the
backbone-only row has no token grid, and FLUX called without a size generates at its
default 1024×1024. Fed to the evaluator (bilinear to 299 without antialiasing) against a
256-px reference, 1024-px images would keep IS (≈157, as observed) and inflate FID and
especially sFID — the paper's pattern (2.1× FID, 4.3× sFID relative to ours). Test:
`ARM=flux RES=1024` (TAG `flux1024`), scored with the same evaluator (a) directly and (b)
after antialiased downscaling to 256 (`RESIZE=256`). If (a) reproduces 51.20 / 224.82 /
157.46, the evaluator matches the paper, the FLUX row was generated at 1024, and the
Bifrost-1 gap (ours vs Table 1) is a model difference to be found in training/sampling.

**Result: hypothesis rejected.** FLUX at 1024 px, seed 0: scored directly FID 23.170 /
sFID 49.483 / IS 167.45; downscaled to 256 first FID 24.747 / sFID 49.291 / IS 153.32
(256-px FLUX: 24.812 / 52.107 / 156.56). Generation size does not produce the paper's
51.20 / 224.82. Next: the reference-set search (ADM 64/128/512, ImageNet-val squashed,
held-out 10k) and the question to the authors.

**Reference-set search (plain FLUX 256 px, seed 0; IS 156.56 under every reference):**

| reference | FID | sFID | Precision | Recall |
|---|---|---|---|---|
| ADM 256 (ours) | 24.812 | 52.107 | 0.871 | 0.184 |
| ADM 512 | 25.098 | 52.548 | 0.857 | 0.189 |
| ADM 128 | 29.057 | 64.723 | 0.873 | 0.143 |
| ADM 64 | 76.348 | 110.571 | 0.665 | 0.030 |
| ImageNet val, centre crop | 25.406 | 54.498 | 0.815 | 0.190 |
| ImageNet val, squashed | 26.003 | 49.606 | 0.821 | 0.187 |
| held-out train 10k | 27.803 | 66.862 | 0.872 | 0.191 |
| paper (Table 1) | 51.20 | 224.82 | — | — |

No reference set reproduces the paper's FLUX row; with the 1024-px test, generation size and
reference choice are both ruled out. The paper's FID/sFID pipeline remains unknown (question
to the authors pending).

**Tuned AdaRQ-Flow (75k steps), 3 seeds (ADM):** FID 29.829 / 30.174 / 30.317 = 30.11 ± 0.25,
sFID 28.05–28.63, IS 51.3–51.8, precision 0.61–0.62, recall 0.64–0.65 (val reference FID
26.22 / 26.53 / 26.62). Worse than the 16-epoch run (28.38 ± 0.33): stopping at the held-out
code-CE minimum does not help. (Seed 2 may have been sampled after the separate-stream change;
same distribution at default settings.)

**IS of real images (anchor).** The ADM reference batch itself (10,000 real ImageNet images,
activations with the evaluator's own code, off-cluster): **IS 337.85**; its FID against the
batch's stored statistics is 3.90 — the stored statistics are presumably from a larger image
set, so ~3.9 is the finite-sample floor at 10k. Against it: Bifrost-1 (ours) 266.5 = 79% of
real; plain FLUX 157 = 46%; paper Bifrost-1 98.6 (Table 1) / 156.0 (Table 4); AdaRQ-Flow 56.1
= 17%. Our Bifrost-1's IS is below real images, not above: it is high because the bridge
reliably conveys the class, which plain FLUX often misses. Its limitation is diversity
(recall 0.25), not an IS anomaly. The paper's Table 1 value is below plain FLUX, unlike ours.

**ADM recalibration on an H200 node (job 1508105, vhpc13, Intel Xeon 6760P, 8 pinned BLAS
threads): FAILED** — IS 215.837158203125, then FID raised `Imaginary component
0.06457371957532043`, identical to every digit to the first H200-node attempt (job 1506465,
unpinned threads). So it is deterministic and tied to that CPU, not to thread count. Its IS
equals our off-cluster value to every digit, so the activations are identical; replaying the
FID step off-cluster on those activations under OpenBLAS kernels Prescott, Core2, Penryn,
Nehalem, Atom, Barcelona, Haswell (plus SkylakeX, Cooperlake, SapphireRapids, Zen,
Sandybridge earlier) always gives a real sqrtm and FID 3.9426. The calibrated setup remains
the CPU partition (vhpc05, AMD EPYC 7413); the scoring job refuses any other CPU type, so
the soft-target scorings queued behind this calibration cannot run.

**Cause found (H200-node diagnostic, job 1508122, vhpc13, Xeon 6760P).** The node's statistics
equal ours to every digit (IS 215.837158203125, mean sum 484.324591524899, trace
159.40084783863261). The fault is numpy's own matrix product: with the OpenBLAS kernel it
selects by itself (numpy 1.23.5's bundled OpenBLAS reports "Cooperlake"; scipy's reports
"SkylakeX"), the covariance product sums to 137135.17 instead of 142402.55, and the square
root turns complex (imaginary 0.0646 — the observed failure; 1 thread: product 136794.55,
imaginary 0.164). Same without TensorFlow in the process, and with OPENBLAS_CORETYPE=
SapphireRapids. With SkylakeX, Haswell or Prescott kernels the product is bit-identical to
the correct one (142402.55221067026) and FID = 3.9425650421 (published 3.9426). A library
fault on this CPU, not the evaluator: `adm_score.sbatch` now sets OPENBLAS_CORETYPE=SkylakeX
on that CPU model only and records the core in every result; calibration on an H200 node
must pass with it before any scoring there (pending the user's go-ahead). Training and
sampling are unaffected: their heavy linear algebra runs on the GPU through PyTorch.

**ADM recalibration on an H200 node with the fix (job 1508131, vhpc13, Xeon 6760P,
OPENBLAS_CORETYPE=SkylakeX): PASSED** — IS 215.8372 (published 215.837), FID 3.9426
(3.942565042), sFID 6.1404, Precision 0.8256 (0.8265 ± 0.005), Recall 0.5289 (0.5309 ±
0.005). Same five numbers as the CPU-node calibration (vhpc05: FID 3.942567012, Precision
0.8256, Recall 0.5288). `ADM_CALIBRATED` now names the Xeon 6760P, so scoring from here on
runs on H200 nodes; all earlier test scores were computed on the AMD CPU node.

**Soft-target AdaRQ-Flow training finished** (158,880 steps): held-out hard code CE ended at
2.603 (minimum 2.545 at 85k; the hard-target run's minimum was 2.365), flow loss 1.169. Its
overfitting is milder but its code prediction is worse. Sampled (3 × 10,000, default
sampler, jobs 1508106–08), not yet scored (awaiting a valid calibration on the scoring node).
Generated depth histogram (1/2/3/4), identical across seeds: 42.1 / 15.3 / 8.4 / 34.2 %, mean
2.35 — deeper than the tokenizer's 51.7 / 18.1 / 8.7 / 21.5 % (mean 2.00): the soft-target
heads halt later than the data does.

**Soft-target AdaRQ-Flow test results (3 seeds × 10,000; scored on H200 nodes, SkylakeX):**

| reference | FID | sFID | IS | Precision | Recall |
|---|---|---|---|---|---|
| ADM (primary) | 41.537 ± 0.659 | 31.163 ± 0.202 | 39.13 ± 0.54 | 0.542 | 0.623 |
| ImageNet val | 37.265 ± 0.593 | 34.340 ± 0.223 | 39.13 ± 0.54 | 0.485 | 0.633 |

Per-seed FID (ADM): 40.793 / 42.050 / 41.767. Worse than the hard-target run on every
metric (FID 28.38 → 41.54, +46%; sFID 28.61 → 31.16; IS 56.1 → 39.1; precision 0.616 →
0.542; recall 0.649 → 0.623) — the first variant that also loses sFID. Soft targets flatten
the code distribution, and sampling it at temperature 1 is even more random, consistent with
its worse held-out hard code CE (2.60 vs 2.37) and deeper codes (34% at depth 4). This
supports the code-sampling hypothesis behind the planned sweep: randomness in code
selection, not overfitting, is what costs FID.

**H200 scoring cross-check:** Bifrost-1 seed 0 rescored on an H200 node (vhpc14, SkylakeX):
FID 16.991, sFID 28.270, IS 270.217, P 0.939, R 0.242 — identical to its CPU-node score, so
scores from the two node types are interchangeable.

**Test results — 3 seeds (10,000 samples each; calibrated ADM evaluator).** Mean ± std:

| arm | reference | FID | sFID | IS | Precision | Recall |
|---|---|---|---|---|---|---|
| Bifrost-1 (retrained) | ADM (primary) | 16.977 ± 0.093 | 28.301 ± 0.030 | 266.5 ± 3.2 | 0.938 | 0.246 |
| AdaRQ-Flow, 16 epochs | ADM (primary) | 28.380 ± 0.325 | 28.611 ± 0.110 | 56.1 ± 1.0 | 0.616 | 0.649 |
| Bifrost-1 (retrained) | ImageNet val | 19.198 ± 0.142 | 31.507 ± 0.018 | 266.5 ± 3.2 | 0.902 | 0.237 |
| AdaRQ-Flow, 16 epochs | ImageNet val | 24.917 ± 0.299 | 31.839 ± 0.127 | 56.1 ± 1.0 | 0.560 | 0.651 |

Per-seed FID (ADM): Bifrost-1 16.991 / 16.878 / 17.062; AdaRQ-Flow 28.066 / 28.716 / 28.358.
**Decision rule (≥10% FID improvement, >2σ): not met — AdaRQ-Flow (pre-registered,
16 epochs) is 67.2% worse on the primary FID (58σ), 29.8% worse on the secondary.** sFID is
within 1.1%. Caveat: the baseline is not yet confirmed faithful to the paper (see above),
so this is a result of our setup; the direction and size of the gap are not in doubt.

AdaRQ-Flow, validation-tuned budget (75k steps), seed 0: FID 29.829 / sFID 28.052 /
IS 51.79 (ADM); FID 26.223 (val) — slightly worse than the 16-epoch run's seed 0 (28.066),
so stopping at the held-out minimum of the code CE does not close the gap. Seeds 1–2 and
the soft-target run pending.

**Next: code-sampling sweep on the existing 16-epoch AdaRQ-Flow checkpoint (validation only;
declared before any pilot runs; not queued until the running jobs finish).**
- Sampler (`AdaRQHeads.sample_level`): halting is gated — halt vs continue is drawn from the
  ORIGINAL halt probability softmax(ℓ)[K]; only when continuing are temperature / top-p /
  greedy applied, to the K ordinary codes alone. At T = 1 without filtering this equals the
  previous categorical sampler (tested). Level 1 never halts. Separate random streams for
  codes, halting, residuals (torch) and token order (NumPy) and the renderer, so a setting
  changes only what it targets. The generated depth histogram is logged against the
  tokenizer's (1/2/3/4: 51.7 / 18.1 / 8.7 / 21.5 %) as a diagnostic, not a target.
- Fixed: 16-epoch checkpoint, w = 1, 32-step residual solver with unit noise, FLUX settings.
- Configurations (code temperatures T1..T4 / filter on ordinary codes): A 1,1,1,1 / none
  (baseline); B 0.8×4 / none; C 1×4 / top-p 0.95; D 0.8×4 / top-p 0.95; E 0.7,0.85,1,1 /
  top-p 0.95; F greedy ordinary codes (halting and residual still sampled). Starting values,
  not established optima.
- Pilot: 2,000 class-balanced samples (2 per class), validation seed 100, identical prompts,
  token orders and renderer seeds across configurations; Bifrost-1 at the same size and seed
  as reference. Scored ONLY against the held-out 10k (the scoring job refuses any other
  reference for N ≠ 10,000): FID / sFID / precision / recall, plus requested-class top-1/top-5
  with a fixed classifier (torchvision ResNet-50 IMAGENET1K_V2; ceiling = the same classifier
  on 2,000 real held-out crops). Pilot FIDs are compared only with each other.
- Looked for: a large precision and class-accuracy gain that keeps part of the recall
  advantage. If greedy codes (F) barely help, stop on code temperature and examine the
  residual and the rollout. Survivors are confirmed at 10,000 × 3 seeds on the test set as a
  declared variant; the pre-registered 16-epoch result stays as reported.
- Then, in order: residual initial-noise scale α ∈ {1, 0.8, 0.6} on the winning code sampler
  (empirical; the flow was trained with unit noise, so α is not a principled temperature),
  and the rollout-mismatch diagnostic below.

**Sweep result (validation only, 2026-11-23; 2,000 class-balanced samples, seed 100,
held-out 10k reference; not a test result, not comparable with the 10k ADM-reference
numbers).**

| pilot | FID | sFID | IS | P | R | top-1 | top-5 | depth 1/2/3/4 % (mean) |
|---|---|---|---|---|---|---|---|---|
| A (T 1, no filter) | 46.71 | 121.87 | 53.2 | 0.602 | 0.625 | 0.425 | 0.683 | 44.8/15.2/8.4/31.7 (2.27) |
| B (T 0.8) | 40.59 | 119.25 | 62.4 | 0.649 | 0.658 | 0.484 | 0.736 | 44.9/15.6/8.4/31.2 (2.26) |
| C (top-p 0.95) | 43.77 | 120.63 | 55.7 | 0.630 | 0.652 | 0.443 | 0.689 | 44.2/15.4/8.4/32.0 (2.28) |
| D (T 0.8, top-p 0.95) | 40.62 | 118.56 | 61.6 | 0.629 | 0.660 | 0.483 | 0.730 | 44.2/15.7/8.4/31.7 (2.28) |
| E (T 0.7,0.85,1,1, top-p 0.95) | 37.92 | 118.29 | 68.3 | 0.653 | 0.666 | 0.511 | 0.753 | 44.4/15.3/8.4/31.8 (2.28) |
| F (greedy codes) | **32.04** | **115.76** | **89.3** | **0.713** | **0.678** | **0.572** | **0.822** | 45.1/16.6/8.6/29.7 (2.23) |
| Bifrost-1 (reference) | 29.83 | 110.71 | 280.7 | 0.942 | 0.676 | 0.888 | 0.990 | — |

Class-accuracy ceiling (same classifier, 2,000 real held-out crops): top-1 0.9145, top-5
0.9905; inflated because the classifier was trained on ImageNet train (published val
top-1 80.9 %). Reading:
- Monotone in code sharpness; greedy ordinary codes (F) are best on every metric: FID −31 %
  vs A, within 2.2 of Bifrost-1's pilot FID, and recall is not lost (0.678 vs 0.676).
- Gated halting kept depth stable across all settings (mean 2.23–2.28); the generated
  depth over-uses level 4 vs the tokenizer (≈30 % vs 21.5 %) and under-uses level 1.
- Precision and class identity remain far below Bifrost-1 (P 0.71 vs 0.94; top-1 0.57 vs
  0.89; IS 89 vs 281). Code-selection randomness explains much of the FID gap but not the
  class-identity gap, so the next steps target the residual and the rollout.
- Per the declared plan, F (and E as runner-up) are candidates for confirmation at
  10,000 random-class samples × 3 seeds with the ADM reference; not yet run.

**Reveal schedule.** With 64 steps for 64 tokens, upstream's clamps (at least one token per
step, at most masked − 1) make every step reveal exactly one token, so the cosine schedule
has no effect: the rollout is linear. Asserted on every sampled batch.

**Training–sampling mismatch (both arms; diagnostic planned).** Training masks at least 45
of 64 tokens, so the branch sees at most 19 visible tokens, all real; a rollout reaches 63
visible tokens, all generated. To measure on held-out data: prediction quality with real vs
generated visible latents at matched visible counts, and with ground-truth vs sampled code
prefixes, by rollout stage. A sharp drop with generated context points to a training change
(wider mask ratios, exposure to imperfect context) rather than more aggressive filtering.

**Evaluation-protocol conformance (test results).** Paper §5.1/§4 → ours: 10K samples →
10,000 per seed; classes "randomly sampled from 1k categories" → uniform random, seeded;
256×256 → 256; 64 MAR decoding steps, random order → same (one token per step); bridge
guidance not stated → upstream default 1.0; FLUX 28 steps, ControlNet scale 0.7, guidance
3.5 → same; FID / sFID / IS → same (+ precision/recall as extras). Not stated in the paper:
the FID implementation and reference statistics ("following [MAR, xAR]") → OpenAI's ADM
evaluator with its 256-px reference batch, calibrated; the prompt text → the class name
(reproduces the paper's plain-FLUX IS). The code-sampling sweep is a validation procedure,
not this protocol (2,000 class-balanced samples, held-out reference, class accuracy); any
setting it selects is re-run under this exact protocol (10,000 random-class samples × 3
seeds, ADM reference) before a result is reported. The Bifrost-1 arm is never changed.

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

## Queued 2026-11-23: sampler confirmation on test, residual-noise pilot on validation

- **Test (decides).** Sampler F (greedy ordinary codes), chosen on validation **before** any
  test scoring, is the primary sampler; E (T = 0.7, 0.85, 1, 1, top-p 0.95) is reported
  alongside it, not picked between on test. Exact Bifrost-1 protocol: 10,000 random-class
  samples × seeds 0, 1, 2, FID / sFID / IS (+P/R) vs the ADM reference; ImageNet-val FID
  secondary. Class top-1/top-5 on seed 0 for F, E and Bifrost-1. Requests `2026-11-23-test-*`.
- **Validation (tunes only).** Residual-noise scale α ∈ {0.8, 0.6} on sampler F; α = 1 is
  pilot F. Same pilot protocol (seed 100, 2,000 class-balanced, held-out reference). If an α < 1
  wins on validation, that setting gets its own full test run before any claim. Requests
  `2026-11-23-alpha-*`.


### Test results 2026-11-24: sampler F (primary) and E (ADM reference, 3 × 10,000, seeds 0–2)

| arm | FID | sFID | IS | P | R | top-1 / top-5 (seed 0) |
|---|---|---|---|---|---|---|
| Bifrost-1 | 16.98 ± 0.09 | 28.30 | 266.5 | 0.938 | 0.246 | 0.895 / 0.989 |
| AdaRQ-Flow, default sampler | 28.38 ± 0.33 | 28.61 | 56.1 | 0.616 | 0.649 | — |
| AdaRQ-Flow, F (greedy codes) | **15.50 ± 0.13** | **24.35** | 90.7 | 0.719 | 0.604 | 0.579 / 0.831 |
| AdaRQ-Flow, E | 21.36 ± 0.15 | 26.08 | 70.5 | 0.671 | 0.629 | 0.502 / 0.756 |

Mixed result on test: F beats Bifrost-1 on FID (−1.5, ~10 seed s.d.), sFID and recall, and
loses on IS (91 vs 267), precision and class accuracy (0.58 vs 0.89). Validation pilots ranked
F behind Bifrost-1 on FID (32.04 vs 29.83); the test reversed that, as the rule above warns.
ImageNet-val FID (secondary): Bifrost-1 19.20 ± 0.14, default 24.92 ± 0.30, F 13.65 ± 0.08, E 18.63 ± 0.16. F/E were scored on H200 nodes (OpenBLAS SkylakeX), Bifrost-1 on the CPU partition; the H200 cross-check of Bifrost-1 seed 0 matched (16.991 vs 16.99). Validation α pilots, class top-1: α 1 0.572,
0.8 0.615, 0.6 0.641 (α = 0.6 not yet run on test).

### IS decomposition on the test samples (2026-11-24; `levelb/is_decomp.py`, seeds 0–2)

ln IS = H(p(y)) − E[H(p(y|x))], with the evaluator's own Inception network; every job
reproduced the IS on record. ResNet-50 columns from `class_accuracy --per-class`.

| arm | IS | H(p(y)) | eff. classes | E[H(p(y\|x))] | top prob > 0.5 | same-class argmax agreement | ResNet top-1 | p(requested) |
|---|---|---|---|---|---|---|---|---|
| real (ADM reference batch) | 337.85 | 6.826 | 922 | 1.004 | 0.910 | — | — | — |
| Bifrost-1 | 266.55 | 6.822 | 918 | 1.237 | 0.879 | 0.895 | 0.892 | 0.386 |
| AdaRQ-Flow F | 90.75 | 6.814 | 910 | 2.306 | 0.595 | 0.587 | 0.580 | 0.222 |
| AdaRQ-Flow E | 70.50 | 6.798 | 896 | 2.543 | 0.528 | 0.514 | 0.497 | 0.188 |
| AdaRQ-Flow default | 56.09 | 6.775 | 876 | 2.748 | 0.476 | 0.447 | 0.432 | 0.158 |

- Class spread is equal across arms (and equal to real images); the whole IS gap is
  per-image uncertainty (F vs Bifrost-1: 1.069 of the 1.077 nats).
- The deficit is spread over classes: F is below Bifrost-1 in 95% (ResNet top-1) / 98.6%
  (Inception KL) of the 1,000 classes; the worst 10% of classes carry ~21% of the gap; per-class
  correlation between the arms is 0.55–0.57.

### Diagnosis against the closest literature (2026-11-24)

Closest work: HART (2410.10812; discrete tokens + residual diffusion MLP), MAR (2406.11838),
TokenBridge (2503.16430), BLIP3-o (2505.09568; flow matching vs MSE on CLIP features),
Bifrost-1 (2508.05954). Weaknesses of AdaRQ-Flow as built, each tied to a fact on record:

1. **Unguided sampling against a mean regressor.** Both arms run `cfg_weight = 1.0` and train
   with prompt dropout 0 (upstream's settings). Bifrost-1's MSE head outputs the conditional
   mean, which needs no guidance; AdaRQ-Flow samples. Sampled class-conditional token models
   without guidance score IS 60–200 and 220–290 with it (MAR Table 1: AR+CE 60.8 → 227.3,
   MAR+CE 146 → 278); AdaRQ-Flow's 56–91 is in the unguided range. Guidance cannot be
   switched on for the existing checkpoint: it never saw an empty prompt.
2. **The codes do not carry "the big picture".** Stage 0 leaves a mean relative residual of
   0.699 at mean depth 2: about half the latent's energy is in the residual. HART's premise is
   the opposite (discrete tokens alone decode well; the residual is detail, so a 37M MLP with 8
   steps is enough).
3. **Residual head too small for its job.** A per-token MLP (width 1024, 3 blocks; MAR's head
   for 16-dim VAE tokens) models a 2048-dim residual holding half the signal. BLIP3-o models
   64 CLIP tokens jointly with a 1.4B diffusion transformer and reports capacity matters.
   Consistent with the α pilots: less residual noise is monotonically better (α 1 → 0.6).
4. **Shallow depth-autoregressive code head** (one linear readout per level). TokenBridge: head
   capacity 3M → 94M moves FID 2.88 → 1.94; independent prediction is far worse.
5. **Mask-ratio mismatch (≤ 19 visible in training, up to 63 at sampling) is standard MAR
   practice** and works there; downgraded from prime suspect to a secondary one.

### Step 0 (queued 2026-11-24): where is class information lost? Validation only, H200 only

No training. All on the validation setup of the sweep (2,000 class-balanced images, seed 100,
held-out reference, ResNet-50 class accuracy). Requests `2026-11-24-step0-*`.

- **Oracle renders** (`levelb/oracle_render.py`): the latents of 2,000 real held-out images
  (2 per class; the rows of `class_accuracy --heldout`, real-pixel top-1 0.9145) through the
  same ControlNet + FLUX, with no MLLM and no trained head: `latent` (true z; renderer
  ceiling), `adaptive` (Stage-0 codes at mean depth 2, residual dropped), `depth4`, `depth1`.
  Their FID is against a reference that contains the same images, so it is comparable between
  oracle modes only; class accuracy and IS are the read-outs.
- **Residual-noise sweep, completed**: sampler F with α = 0.4, 0.2, 0 (1, 0.8, 0.6 done).

Reading: if code-only renders lose the class that true-latent renders keep, the tokenizer
is the bottleneck and no sampler or guidance can recover it; if they keep it, the loss is in
generation (weakness 1: no guidance). If α = 0 is best, the residual head adds only noise.

### Step 0 results (2026-11-26; validation, 2,000 class-balanced images, seed 100; all 21 jobs completed)

Oracle renders of real held-out images (no MLLM, no trained head) and generated pilots.
Real pixels of the same 2,000 images: top-1 0.9145 (inflated: the classifier trained on them).
FID of oracle rows is against a reference containing the same images: comparable between
oracle rows only. Class-accuracy standard error ≈ 0.011.

| bridge input | energy missing | top-1 | top-5 | IS | P | R | FID |
|---|---|---|---|---|---|---|---|
| real image, true latent | 0% | 0.762 | 0.950 | 178.9 | 0.915 | 0.816 | (21.73) |
| real image, codes only, depth 4 | 36.6% | 0.658 | 0.891 | 131.7 | 0.834 | 0.772 | (25.26) |
| real image, codes only, adaptive (mean 2.0) | 46.0% | 0.589 | 0.842 | 97.6 | 0.778 | 0.717 | (28.96) |
| real image, codes only, depth 1 | 50.0% | 0.556 | 0.804 | 93.4 | 0.739 | 0.704 | (31.00) |
| generated, AdaRQ-Flow F, α = 1 | — | 0.572 | 0.822 | 89.3 | 0.713 | 0.678 | 32.04 |
| generated, F, α = 0.8 | — | 0.615 | 0.853 | 99.8 | 0.765 | 0.682 | 29.08 |
| generated, F, α = 0.6 | — | 0.641 | 0.883 | 108.9 | 0.787 | 0.685 | 27.73 |
| generated, F, α = 0.4 | — | 0.644 | 0.890 | 111.7 | 0.808 | 0.659 | 28.17 |
| generated, F, α = 0.2 | — | 0.649 | 0.891 | 113.4 | 0.819 | 0.640 | 28.90 |
| generated, F, α = 0 | — | 0.649 | 0.886 | 116.7 | 0.788 | 0.642 | 29.25 |
| generated, Bifrost-1 | — | 0.888 | 0.990 | 280.7 | 0.942 | 0.676 | 29.83 |

Stored-code agreement of the adaptive render: 0.99999. Mean depth of the generated pilots falls
with α (2.23, 2.06, 1.89, 1.76, 1.67, 1.70).

Conclusions (validation; they choose what to build, they do not rank methods):
1. **The tokenizer loses class information, with its control in place.** Same images, same
   renderer: true latent 0.762 → codes only at the operating depth 0.589 (−0.17; IS 179 → 98).
   Four codes still lose 0.10. Weakness 2 is confirmed.
2. **Bifrost-1's generated images are more class-typical than real images' own latents**
   through the same renderer (0.888 vs 0.762; IS 281 vs 179). Its class fidelity is above the
   real-latent level: the mean-regression effect. A faithful sampler of real latents would
   top out near the true-latent row, not at Bifrost-1.
3. **The gap to Bifrost-1 therefore has two parts of similar size:** tokenizer loss
   (0.762 → 0.589) and typicality (0.762 → 0.888). Fixing the tokenizer alone cannot reach
   Bifrost-1's IS; it also needs a typicality mechanism (guidance, or anchoring on the mean).
4. **Residual noise:** class accuracy rises from α = 1 to 0.6 (0.572 → 0.641) and is flat
   below (0.649 at 0); FID is best at 0.6 and recall falls below it. The residual head adds
   diversity and little class information; it is not "only noise".
5. The generator at α ≤ 0.6 (0.64–0.65) is above the code-only renders of real images at the
   same depth (0.589): generated codes are somewhat more typical than real ones, so the
   code-only row is not a hard cap on generation.

### Phase 1 (queued 2026-11-26): no-training tests toward a guided, information-aware bridge

Goal set by the user: beat Bifrost-1 on FID, sFID and IS together on test, with an efficiency
advantage. Plan and literature: `docs/adarq_v2_options.md`. Validation only (2,000
class-balanced images, seed 100, held-out reference, ResNet-50 class accuracy); H200 only.
Requests `2026-11-26-p1-*`.

- **Guidance on the existing AdaRQ-Flow checkpoint** (trained without prompt dropout):
  independent condition guidance (arXiv 2407.02687), negative branch = a random other class
  redrawn every decoding step, applied to code logits and flow velocity, weights 1.5 / 2 / 3.
  Two samplers: `gF` (greedy codes, residual noise 0.6; control `pilot_adarq_F_a06`) and `gS`
  (sampled codes, default sampler; control `pilot_adarq_A`). At weight 1 the negative branch
  cancels, so the controls are exact. Shown for diffusion models only: a positive result is
  informative, a negative one is not conclusive.
- **Information logging in those runs:** per decoding step and code level, KL(conditional ||
  negative branch) and the entropies of the conditional and guided code distributions (nats).
  Guided logits are log p(x|c) + (w − 1)·PMI(x; c); these logs show where class information
  sits, for information-gated guidance and rate.
- **Bifrost-1 + noise baseline:** Gaussian noise of std 0.33 / 0.66 / 1.31 on each revealed
  token (held-out RMSE of the regressor: 1.31). Control `pilot_bifrost`.
- **Latent statistics** (`levelb/latent_stats.py`): energy shares (global mean, between-class),
  spectrum and effective dimension, and one linear probe of class information in true
  latents, code-only reconstructions and the residual.
