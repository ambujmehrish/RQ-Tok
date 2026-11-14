# Question to the Bifrost-1 authors — ImageNet 256×256 evaluation (Sec. 5.1, Tables 1 and 4)

(Draft for an issue on github.com/HL-hanlin/Bifrost-1 or an e-mail to the authors.)

Hi, thank you for releasing Bifrost-1. We are reproducing the ImageNet 256×256 experiment
(Sec. 5.1) and could not find the evaluation code for it in the repository (only GenEval and
DPG-Bench). Could you share the following?

1. **FID / sFID / IS pipeline.** Which implementation (OpenAI's guided-diffusion
   `evaluator.py`, torch-fidelity, clean-fid, other) and which reference statistics (e.g.
   `VIRTUAL_imagenet256_labeled.npz`, ImageNet train or val, how many images, what
   resize/crop)? With the ADM evaluator and the ADM 256 reference, plain FLUX.1-dev
   (class name as prompt, 256×256, 28 steps, guidance 3.5, 10k samples) gives us
   IS 156.6 — matching your 157.46 — but FID 24.8 and sFID 52.1 versus your 51.20 and
   224.82, so we believe the reference or implementation differs.
2. **Prompt to FLUX.** What text is given to FLUX (and to the MLLM) for a class: the class
   name, a template such as "a photo of a {class}", or none?
3. **Table 1 vs Table 4.** Table 1 reports Bifrost-1 at FID 25.77 / sFID 53.67 / IS 98.57
   (16 + 2 epochs), and Table 4 reports FID 18.64 / sFID 43.64 / IS 156.01 for
   "64 (default)" decoding steps. Are these the same setting? If not, what does Table 4 use
   (checkpoint / training budget, number of samples, evaluation)? And how many decoding
   steps does the Table 1 row use?
4. **Model and sampling details for Sec. 5.1.** Qwen2.5-VL 3B or 7B; number of tokens and
   input resolution of the CLIP latents; batch size and learning-rate schedule of the
   vision branch; MAR guidance scale and temperature at sampling.

Thank you!
