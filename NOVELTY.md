# AdaRQ-Flow vs. Bifrost-1 — What's New and Why

This document states precisely how **AdaRQ-Flow** differs from its baseline
**Bifrost-1** ([arXiv:2508.05954](https://arxiv.org/abs/2508.05954)), maps each change
to the Bifrost-1 weakness it fixes, and points at the exact code that implements it.

AdaRQ-Flow **keeps Bifrost-1's central, validated insight** — bridge a *frozen MLLM*
and a *pretrained generative renderer* through **MLLM-native CLIP patch latents** (the
alignment that makes the bridge cheap and effective). Everything below changes *how the
bridge is represented and learned*, not that thesis.

---

## 1. The one-line difference

> **Bifrost-1:** the MLLM predicts a *single continuous CLIP vector per patch* with an
> **MSE** loss; a ControlNet feeds those (blurred) vectors to FLUX.
>
> **AdaRQ-Flow:** the MLLM predicts an **adaptive-depth residual-quantized** code
> sequence with **cross-entropy** *plus* a **flow-matching** head for the continuous
> residual; the renderer is trained on the **dequantized code space** it actually sees
> at inference.

The novel combination — **(MLLM-native CLIP bridge) × (adaptive-depth RVQ) ×
(flow-matching for both the latent residual head and the renderer)** — is, to our
knowledge, not united in any prior work (see §4).

---

## 2. Weakness → fix → code

| # | Bifrost-1 weakness | AdaRQ-Flow mechanism | Where it lives |
|---|---|---|---|
| 1 | **MSE on continuous latents → mode-averaging / blur** (paper §3.3 uses MSE for patch-embedding prediction; likely hurts FID) | **Flow-matching residual head** (rectified-flow / conditional-OT velocity objective) replaces MSE → a proper *distributional* target. Discrete codes carry the coarse signal so the head only models residual detail. | `mllm/flow.py` (`flow_matching_loss`, `rectified_flow_target`), `mllm/heads.py` (`FlowResidualHead`), used in `mllm/model.py:compute_loss` |
| 2 | **Continuous bridge abandons the LLM-native interface** — no cross-entropy, no temperature/top-p, no token-level **CFG**, no likelihood | **Discrete adaptive-RVQ codes** predicted with **cross-entropy**; a learned **null context** enables **classifier-free guidance** on the code logits; sampling supports temperature. | `tokenizer/quantizer.py` (codes), `mllm/heads.py` (`CodeClassifierHead`), `mllm/model.py` (`null_context`, `_apply_cfg_dropout`, `_cfg_logits`, `_sample_codes`) |
| 3 | **Semantic bottleneck of one CLIP vector per patch caps fidelity** — only improvable by scaling token count | **Residual quantization** stacks `D_max` codes per patch (coarse→fine); the leftover **continuous residual** is recovered by the flow head. Fidelity scales with *depth*, not just patch count. | `tokenizer/quantizer.py` (`AdaptiveResidualQuantizer`, `dequantize`), residual `res = z − ẑ` handed to the flow head |
| 4 | **Exposure bias** — ControlNet trained on *ground-truth* CLIP latents but fed *MSE-blurred predicted* latents at inference | Renderer (Stage B) is trained on the **dequantized code space** `ẑ + res` — the *same finite-vocabulary distribution* it receives at inference. The discrete vocabulary makes the train/inference gap small by construction. | `training/trainer.py:_step_renderer` (`control = ẑ + res`), `config.RendererConfig.train_on_dequantized`, `scheduled_sampling_prob` hook |
| 5 | **Fixed, uniform token budget** — flat and detailed patches get identical cost | **Content-adaptive depth**: a residual-norm halting rule stops early on flat patches and spends more codes on detailed ones; a `<halt>` symbol lets the MLLM predict variable depth; a rate penalty keeps budgets honest. | `tokenizer/quantizer.py` (halting loop, `<halt>` sentinel = class `K`), `config.TokenizerConfig.adaptive_depth / halt_residual_threshold / rate_penalty` |

---

## 3. Mechanism details (and how they differ from Bifrost-1)

### 3.1 Adaptive-depth RVQ-CLIP tokenizer — *new component*
Bifrost-1 has **no tokenizer**: a patch *is* its raw CLIP vector. AdaRQ-Flow inserts a
residual quantizer over the (frozen) CLIP latents:

- a **shared (or per-depth) EMA codebook** with commitment + usage-entropy losses and two
  anti-collapse mechanisms (first-batch data init, dead-code reinit) —
  `tokenizer/codebook.py`;
- **per-patch adaptive depth** via residual-norm halting (`adaptive_depth`) —
  `tokenizer/quantizer.py`;
- output = discrete `codes`, per-patch `depths`, dequantized prefix `ẑ`, and continuous
  `res = z − ẑ`. The codes feed the cross-entropy head; `res` feeds the flow head.

This makes the bridge a **finite-vocabulary, variable-rate** interface — the property
that unlocks fixes #2, #4, #5 simultaneously.

### 3.2 Hybrid head — *changed objective + interface*
Bifrost-1's branch has a single linear *vision head* trained with **MSE**. AdaRQ-Flow's
branch (same idea: a trainable copy of the MLLM layers, backbone frozen) carries **two**
heads (`mllm/model.py`):

1. **`CodeClassifierHead`** — per-residual-level softmax over `K+1` classes (the `+1` is
   `<halt>`), trained with **cross-entropy**; CFG-capable.
2. **`FlowResidualHead`** — a flow-matching velocity MLP for the continuous residual,
   conditioned on the branch hidden state *and the predicted dequantized prefix*.

Training is **MAR-style masked** (random masking, CE + flow on masked patches, CFG
text-dropout); decoding is **MaskGIT-style** iterative unmasking with CFG, then the flow
head fills the residual by ODE sampling.

### 3.3 Flow-matching latent ControlNet — *same recipe, exposure-bias-aware, no diffusion*
Like Bifrost-1 we adapt the FLUX ControlNet (input projection + 2D downsample + a few
trainable DiT blocks, zero-init residual injection — `renderer/controlnet.py`). The
differences: it conditions on **`ẑ + res`** (the dequantized code space, fix #4) and the
whole stack is **rectified-flow / velocity** end to end — **no DDPM-style diffusion
anywhere** (`renderer/renderer.py`, `mllm/flow.py`).

---

## 4. Relationship to prior art (why the *combination* is novel)

| Prior work | What it has | What AdaRQ-Flow adds over it |
|---|---|---|
| **Bifrost-1** | CLIP-native bridge + ControlNet + FLUX, **MSE** branch | flow-matching residual + adaptive RVQ + discrete CFG-able interface; exposure-bias fix |
| **NextStep-1** (2508.10711) | AR + continuous tokens + flow-matching head | we borrow the flow head but on a **CLIP-native, MLLM-bridged, residual-quantized** interface (NextStep uses VAE-style tokens, single-arch, no quantization) |
| **ResGen** (2412.10208) | RVQ tokens, **fixed depth**, discrete-diffusion | we make depth **adaptive**, use a **CLIP-native** bridge, and **flow matching** |
| **VRVQ / RAQ** (2405.14222) | variable-rate RVQ for audio/compression | we adapt **per-patch depth** for an **MLLM image-generation bridge** |
| **FlowAR** (2412.15205) | scale-wise AR + flow matching, from scratch | we **bridge pretrained** MLLM + FLUX (efficiency thesis), CLIP-native |

**Novel union:** MLLM-native CLIP bridge **×** adaptive-depth RVQ **×** flow matching for
*both* the residual head and the renderer. No prior work combines all three.

---

## 5. How each claim is checked in this repo

Every mechanism above is exercised by CPU tests on dummy stand-ins (the real
Qwen2.5-VL / FLUX.1-dev swap is isolated to two `build_*` factories — see `DESIGN.md`
§9). Direct evidence:

- **Distributional residual (fix #1):** `tests/test_mllm.py::test_flow_head_learns_target`,
  `tests/test_renderer.py::test_renderer_learns_control_to_latent`.
- **Discrete CFG-able interface (fix #2):** `tests/test_mllm.py::test_generate_*`
  (CFG scale, greedy determinism), `test_compute_loss_*`.
- **Residual depth recovers detail (fix #3):**
  `tests/test_eval.py::test_reconstruction_vs_depth_monotone` — the recon-vs-depth curve
  (our analog of Bifrost-1's token-count scaling, Fig. 4).
- **Exposure-bias-aware renderer (fix #4):** `training/trainer.py:_step_renderer` trains on
  `ẑ + res`; `tests/test_training.py::test_stage_renderer_decreases`.
- **Adaptive depth (fix #5):** `tests/test_tokenizer.py::test_halting_depth_varies_by_detail`,
  `test_non_adaptive_uses_full_depth`.

The planned head-to-head ablations (incl. a **continuous + no-code "Bifrost-style"
baseline** under matched compute) are generated by `adarq_flow/eval/ablations.py`
(`configs/ablations/`), per `DESIGN.md` §7.
