"""Evaluation harness.

Self-contained metrics (MSE/PSNR/SSIM, codebook perplexity) + a tokenizer evaluator
(reconstruction vs. adaptive depth) run on CPU/dummy. The external metrics
(FID/sFID/IS, rFID/LPIPS, GenEval, DPG-Bench) raise until wired with real models/data.
Requires torch.
"""

from .ablations import ablation_configs, dump_ablation_configs
from .evaluator import TokenizerReport, evaluate_tokenizer, reconstruction_vs_depth
from .metrics import (
    codebook_perplexity,
    dpg_bench_score,
    fid,
    geneval_score,
    inception_score,
    lpips,
    mse,
    psnr,
    rfid,
    sfid,
    ssim,
)

__all__ = [
    "mse", "psnr", "ssim", "codebook_perplexity",
    "fid", "sfid", "inception_score", "rfid", "lpips", "geneval_score", "dpg_bench_score",
    "TokenizerReport", "evaluate_tokenizer", "reconstruction_vs_depth",
    "ablation_configs", "dump_ablation_configs",
]
