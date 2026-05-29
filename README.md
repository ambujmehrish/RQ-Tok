# Bifrost-Flow

**Bifrost-Flow** is the next-generation successor to [Bifrost-1](https://arxiv.org/abs/2508.05954)
for unified multimodal understanding and generation. The name reflects the core change: the
bridge (*Bifröst*) between a frozen multimodal LLM (MLLM) and a pretrained image renderer is
now driven end-to-end by **flow matching** instead of diffusion + MSE.

> Status: **early development.** See [`DESIGN.md`](DESIGN.md) for the full research design.
> The codebase is being built phase-by-phase; tiny configs run on CPU, real backbones
> (Qwen2.5-VL + FLUX.1-dev) plug in via config on GPU nodes.

## Core idea

Bifrost-Flow keeps Bifrost-1's winning insight — bridging an MLLM and a renderer through
**MLLM-native CLIP latents** — but replaces its three biggest weaknesses:

| Bifrost-1 weakness | Bifrost-Flow fix |
|---|---|
| MSE regression on continuous latents → mode-averaging / blur | **Flow-matching** residual head (proper distributional objective) |
| Single CLIP vector per patch → fidelity ceiling | **Adaptive-depth residual quantization** of CLIP latents (coarse→fine) |
| No discrete LLM-native interface (no CFG, no sampling) | **Discrete adaptive RVQ codes** predicted with cross-entropy → enables classifier-free guidance |
| Decoupled-training exposure bias | Renderer trained on the **dequantized code space** it sees at inference |
| Fixed, uniform token budget | **Content-adaptive depth** (more residual codes for detailed patches) |

Everything generative is **flow matching** — no DDPM-style diffusion anywhere. The only
discrete/cross-entropy part is the adaptive code prediction, which is exactly what buys
the LLM-native interface, classifier-free guidance, and adaptive token budget.

## Architecture (4 components)

1. **Frozen MLLM** (Qwen2.5-VL) — unchanged, so understanding benchmarks are preserved.
2. **Adaptive RVQ-CLIP tokenizer** — residual-quantizes MLLM-native CLIP patch latents to
   an adaptive per-patch depth.
3. **Vision generation branch** — trainable copy of the MLLM (QKV/MLP/norm) with a
   *code-classifier head* (cross-entropy, adaptive depth + CFG) and a *flow-matching
   residual head* (continuous detail).
4. **Flow-matching renderer** — FLUX.1-dev (rectified flow) conditioned via a flow-matching
   **latent ControlNet** on dequantized + flow-refined latents.

## Layout

```
bifrost_flow/
  config.py        # dataclass configs + tiny/base presets
  tokenizer/       # adaptive RVQ-CLIP tokenizer
  mllm/            # frozen MLLM + vision generation branch (hybrid head)
  renderer/        # flow-matching latent ControlNet + FLUX renderer
  training/        # decoupled training (tokenizer -> branch -> renderer)
  inference/       # text -> codes -> dequant+flow refine -> render
  eval/            # FID/sFID/IS, rFID/SSIM/PSNR/LPIPS, GenEval, DPG-Bench
  data/            # ImageNet / BLIP3-o loaders
  utils/
configs/           # tiny_cpu.yaml, base_gpu.yaml
tests/
```

## Quickstart (dev)

```bash
pip install -e ".[dev]"                          # config/scaffold tests, no torch
pip install -e ".[dev,torch]"                    # also run the tokenizer (Phase 1) tests
pytest -q                                         # tokenizer tests auto-skip without torch
python -m bifrost_flow.config --print tiny_cpu   # inspect a config preset
```

The adaptive RVQ-CLIP tokenizer (Phase 1) runs on CPU:

```python
import torch
from bifrost_flow.config import get_preset
from bifrost_flow.tokenizer import build_tokenizer, fit_tokenizer

tok = build_tokenizer(get_preset("tiny_cpu"))
z   = tok.encode(torch.randn(8, 3, 8, 8))   # frozen dummy CLIP -> patch latents
fit_tokenizer(tok, z, steps=200)            # Stage-0 EMA codebook fit
out = tok.tokenize(torch.randn(8, 3, 8, 8)) # adaptive RVQ codes + halt depths + residual
```

The MLLM vision-generation branch + hybrid head (Phase 2) also runs on CPU:

```python
from bifrost_flow.config import get_preset
from bifrost_flow.tokenizer import build_tokenizer
from bifrost_flow.mllm import build_vision_gen_model

cfg = get_preset("tiny_cpu")
tok = build_tokenizer(cfg)
model = build_vision_gen_model(cfg)            # frozen (dummy) MLLM + trainable branch

z  = tok.encode(torch.randn(2, 3, 8, 8))       # CLIP latents
qo = tok(z, update_codebook=False)             # RVQ codes / depths / residual / prefix
B, N, D, d = 2, cfg.tokenizer.num_patches, cfg.tokenizer.max_depth, cfg.tokenizer.clip_dim
text = torch.randint(0, 256, (B, 5))

# MAR masked training: cross-entropy (codes + <halt>) + flow-matching (residual)
loss = model.compute_loss(text, qo.codes.reshape(B, N, D),
                          qo.res.reshape(B, N, d), qo.zhat.reshape(B, N, d))

# MaskGIT decode (CFG) + flow residual sampling -> latents for the renderer (Phase 3)
out = model.generate(text, tok.dequantize, cfg_scale=3.0)
```

> The dummy MLLM backbone and dummy CLIP encoder are CPU development stand-ins; they
> are replaced by the real frozen **Qwen2.5-VL** and its CLIP visual tower after the
> final development phase (see `DESIGN.md` §9). All other code is unchanged by that swap.

## Training, inference, eval

Decoupled training (Stage 0 tokenizer → A branch → B renderer), distributed-aware:

```bash
python -m bifrost_flow.training.train --preset tiny_cpu --stage tokenizer   # Stage 0
python -m bifrost_flow.training.train --preset tiny_cpu --stage branch      # Stage A
python -m bifrost_flow.training.train --preset tiny_cpu --stage renderer    # Stage B
# multi-GPU (4x A100, Cineca Leonardo): sbatch scripts/cineca_leonardo_4xA100.sbatch
```

End-to-end inference and the tokenizer reconstruction-vs-depth ablation:

```python
import torch
from bifrost_flow.config import get_preset
from bifrost_flow.inference import build_pipeline
from bifrost_flow.tokenizer import build_tokenizer, fit_tokenizer
from bifrost_flow.eval import evaluate_tokenizer

cfg = get_preset("tiny_cpu")
pipe = build_pipeline(cfg)
out = pipe.generate(torch.randint(0, 256, (2, 5)), cfg_scale=3.0)   # codes -> image latents

tok = build_tokenizer(cfg); z = tok.encode(torch.randn(64, 3, 8, 8))
fit_tokenizer(tok, z, steps=200)
report = evaluate_tokenizer(tok, z)        # recon MSE/PSNR, mean depth, depth curve, perplexity
```

Ablation configs (depth, adaptive on/off, codebook size, CFG, exposure-bias, flow head,
continuous-Bifrost baseline) are in `configs/ablations/`.

## Build status

Phases 0–6 complete and CPU-tested (`pytest -q`, tokenizer/MLLM/renderer/training/inference/eval).
Multi-GPU (DDP/FSDP) wired. Remaining for GPU production: swap the dummy MLLM/CLIP/FLUX
stand-ins for the real frozen models and wire the real datasets + external eval metrics.
