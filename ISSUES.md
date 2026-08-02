# Known Issues

Findings from a three-part audit of the codebase (tokenizer / branch+decoding /
renderer+training+eval). Kept in the repo because a defect that silently weakens an
experiment is more dangerous than one that crashes.

**Rule:** no number produced by this code is citable until the issues marked
**BLOCKING** are closed.

---

## Fixed

| # | Severity | Issue |
|---|---|---|
| F1 | critical | **Multi-GPU replicas never synchronized.** Model construction was seeded `base+rank` and `wrap_model` (DDP) was never called by the trainer — each rank held different weights, and the "frozen tokenizer" produced *different targets* per rank. Now: identical construction seed + explicit `broadcast_module()`; per-rank seeding applies to the data stream only. |
| F2 | critical | **Renderer conditioning was vacuous.** `res := z − ẑ`, so the conditioning `ẑ + res` was *exactly* the raw continuous latent `z`. The renderer never saw the code space. `train_on_dequantized` is now honored and provably changes the conditioning tensor. |
| F3 | critical | **Three ablations were silent no-ops** (`code_head`, `flow_residual_head`, `train_on_dequantized` were never read). They trained identically to baseline — i.e. they would have fabricated null results. Now honored + guarded by `tests/test_ablations_effective.py`. |
| F4 | critical | **Generated codes violated RVQ semantics.** Depth levels were sampled independently, producing `code, halt, code` (measured 6/48 patches), which dequantizes level *k* against a residual that never passed level *k−1*, and corrupts the depth metric. Now projected onto the valid set; `<halt>` forbidden at level 0; `dequantize()` rejects violations. |
| F5 | major | **MaskGIT reveal order tracked predicted depth, not confidence** (log-probs averaged over near-deterministic post-halt levels), so shallow patches were revealed first regardless of certainty. Now scored over the predicted prefix only. |
| F6 | major | **Ablation YAMLs were unrunnable** — the trainer had no `--config` flag. |
| F7 | minor | `scheduled_sampling_prob` silently ignored → now fails loud. `torch.load(weights_only=False)` → `True`. |

---

## Open — BLOCKING for any quantitative claim

| # | Severity | Issue | Consequence |
|---|---|---|---|
| B1 | critical | **Stage 0 optimizes nothing.** The tokenizer has 0 trainable parameters and never calls `backward()`. `commitment`/`entropy` are computed on **detached** tensors and `rate` is non-differentiable, so `commitment_weight`, `entropy_weight`, `rate_penalty` are inert. | Only EMA learns. Any claim about the rate penalty shaping depth is unsupported. Blocks the learned allocator in `NOVELTY.md` §6. |
| B2 | critical | **Codebook usage metrics are saturated.** `cluster_size` starts at 1.0 for every code and decays at 0.99, so reaching the `1e-2` dead threshold needs ~458 consecutive unused steps. Measured: `usage() = 1.0` while only **55/64** codes were live. Dead-code reinit therefore also ~never fires. | The metric that exists to detect codebook collapse cannot detect it. Any previously reported "usage 1.00" is meaningless. Fix: histogram `out.codes` over eval data. |
| B3 | critical | **Renderer training targets are a randomly-initialized projection.** Stage B regresses toward `DummyCLIPEncoder` output (a single untrained conv). | The loss descends, but measures optimizer plumbing, not generative quality. No renderer claim can come from this configuration. |
| B4 | critical | **Dummy data has zero image↔text mutual information** (images and token ids drawn independently). | The branch's cross-entropy and all CFG machinery cannot express anything. Any CFG ablation on dummy data is guaranteed-null by construction. |
| B5 | major | **Flow head is trained only on masked positions but sampled at 0% masked.** Training masks ≥ 70% and takes the loss only at masked positions; at inference every patch is revealed. | The velocity field is evaluated on a conditioning distribution it never saw. Distinct from ordinary exposure bias, and it feeds the renderer. |
| B6 | major | **CFG guides only the code logits.** The continuous residual — which carries all sub-codebook detail — is generated unconditionally. | `cfg_scale` cannot affect fine detail; C1/C4 interpretation is limited. |

## Open — correctness, non-blocking

| # | Severity | Issue |
|---|---|---|
| N1 | major | **Straight-through estimator has the wrong sign.** `q_st = r + (q−r).detach()` accumulated across depths gives `∂ẑ/∂z = depth·I`; measured `cos(STE grad, true grad) = −0.99999994` at depth 4 — exactly anti-parallel. Latent only because the encoder is frozen; becomes gradient *ascent* the moment anything upstream is trainable. Fix: detach the accumulator once, not per level. |
| N2 | major | **`per_depth_codebook` can deadlock under DDP.** `ema_update` runs collectives inside, but books with no samples are skipped, and which books are populated is data- (hence rank-) dependent under adaptive halting → mismatched collective counts. Safe only because `shared_codebook=True` by default. |
| N3 | major | **Seeded GPU runs will crash.** CPU `torch.Generator` objects are passed to CUDA tensor ops (`multinomial`, `normal_`, `rand`). Only `fit.py` and the DDP script build device-aware generators. |
| N4 | major | **SSIM uses zero padding** instead of a valid convolution, contaminating a `pad`-wide border (6 of 8 rows at the CPU image size). Variances are not clamped ≥ 0. |
| N5 | minor | Codebook init / dead-code reinit use the **global RNG**, so `fit_tokenizer(seed=…)` is not reproducible. First-batch init is applied *after* the assignments it then consumes, so the first EMA step blends codes against a discarded random codebook. |
| N6 | minor | `psnr` derives `data_range` per-batch from the target, so values aren't comparable across batches/arms; applied to CLIP latents it has no peak-signal meaning. `metrics.mse` (elementwise mean) and the training `recon` (sum-over-dim) differ by a factor of `clip_dim`. |
| N7 | minor | `reconstruction_vs_depth` is a **truncation** curve for one fixed tokenizer, confounded with halting (patches that halted below `d` contribute unchanged at every larger `d`). It is *not* comparable to the `depth_2/8` arms, which retrain a differently-shaped codebook. |
| N8 | minor | `null_context.expand_as(context)` makes the unconditional embedding **length-dependent** (measured `max|Δ| = 0.40` between T=8 and T=32), biasing CFG extrapolation once prompts vary in length. |
| N9 | minor | No attention/padding mask is plumbed into the branch — fine for fixed-length dummy text, breaks with real variable-length prompts. |
| N10 | minor | `train.precision` (bf16) is never applied; `guidance_scale`, `flow_sigma_min` are dead. `AdaptiveResidualQuantizer.HALT = -1` is dead and dangerous (`-1` passes the `!= K` check and silently indexes the last code). |
| N11 | minor | `entropy_temp` is fixed at 1.0 with no config field; at CLIP scale `softmax(−dist)` is effectively one-hot, so the sample-entropy term degenerates. |
| N12 | minor | `CodeClassifierHead` is `Linear(3584, D·8193)` = **235M params** at `base_gpu` (~2.8 GB with AdamW states) because every depth level is modelled independently. |
| N13 | minor | No working resume: `load_checkpoint` is never called by the entrypoint and the restored `step` is not fed back into `train()`. Checkpoint writes are non-atomic. |
| N14 | minor | `drop_last=True` with `dataset_length / world_size < batch_size` yields an empty loader and `train()` spins forever. Reachable at `base_gpu` defaults on ≥ 8 ranks. |

---

## Corrections to earlier status reports

Stated plainly, because these were reported as working:

- **"Multi-GPU wired and verified"** — overstated. The 2-process gloo test verified only
  the *EMA codebook* all-reduce. It did not test model-weight synchronization, which was
  in fact broken (F1). The test passed while training was silently corrupt.
- **"codebook usage = 1.00"** cited from smoke runs — meaningless (B2).
- **"exposure-bias-aware renderer"** — was not implemented as described (F2).
- **"12 ablation configs"** — three of them were no-ops (F3), and none could be loaded by
  the trainer (F6).
