# Heimdall

**Heimdall** is the next-generation successor to [Bifrost-1](https://arxiv.org/abs/2508.05954)
for unified multimodal understanding and generation. In Norse myth, *Heimdall* is the
guardian of the *Bifröst* bridge — a fitting name for a model that improves the bridge
between a frozen multimodal LLM (MLLM) and a pretrained **flow-matching** image renderer.

> Status: **early development.** See [`DESIGN.md`](DESIGN.md) for the full research design.
> The codebase is being built phase-by-phase; tiny configs run on CPU, real backbones
> (Qwen2.5-VL + FLUX.1-dev) plug in via config on GPU nodes.

## Core idea

Heimdall keeps Bifrost-1's winning insight — bridging an MLLM and a renderer through
**MLLM-native CLIP latents** — but replaces its three biggest weaknesses:

| Bifrost-1 weakness | Heimdall fix |
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
heimdall/
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
pip install -e ".[dev]"
pytest -q                       # pure-Python scaffold tests (no torch needed yet)
python -m heimdall.config --print tiny_cpu   # inspect a config preset
```
