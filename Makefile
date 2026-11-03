.PHONY: install lint type test check fmt clean ablations smoke prefetch e0 vhpc-setup vhpc-env vhpc-prefetch bifrost1-setup bifrost1-geneval bifrost1-dpg geneval-setup geneval-calibrate dpg-setup dpg-calibrate imagenet-prefetch

install:
	pip install -e ".[dev,torch]"

lint:
	ruff check .

fmt:
	ruff check . --fix

type:
	mypy adarq_flow/

test:
	pytest -q

# Full local gate, mirrors CI.
check: lint type test

ablations:
	python -c "from adarq_flow.eval import dump_ablation_configs; \
		print(dump_ablation_configs('configs/ablations', base='configs/experiment.yaml'))"

clean:
	rm -rf checkpoints/ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# End-to-end smoke test on REAL data (real COCO images -> real CLIP -> tokenizer -> E0).
smoke:
	bash scripts/smoke_test.sh

# Prefetch models/images to $ADARQ_CACHE (outside $HOME). RUN ON A LOGIN NODE.
prefetch:
	bash scripts/prefetch_assets.sh

# Powered E0 (64 distinct images x 3 seeds) on ONE vhpc01 H200, seeds sequential.
e0:
	sbatch scripts/e0_downstream_64.sbatch

# --- vhpc01 (3 H200 nodes x 1 GPU each, submit-only SLURM) ---------------------------------------------
vhpc-setup:      # step 1: survey partitions / GPUs / egress from a compute node
	sbatch scripts/vhpc01_setup.sbatch
vhpc-env:        # step 2: build + verify the conda env on an H200
	sbatch scripts/vhpc01_env.sbatch
vhpc-prefetch:   # step 3: fetch models + data (needs ADARQ_CACHE and compute egress)
	sbatch scripts/vhpc01_prefetch.sbatch

# --- Bifrost-1 reference (Level A): official code + released checkpoint ------------------
bifrost1-setup:    # pinned upstream code, its own env, checkpoint, DPG-Bench prompts
	sbatch --qos=gpulongjobs scripts/bifrost1/setup.sbatch
bifrost1-geneval:  # official generation on the 553 GenEval prompts
	sbatch --qos=gpulongjobs --export=ALL,BENCH=geneval scripts/bifrost1/generate.sbatch
bifrost1-dpg:      # official generation on the 1065 DPG-Bench prompts
	sbatch --qos=gpulongjobs --export=ALL,BENCH=dpgbench scripts/bifrost1/generate.sbatch

# --- benchmark scorers (calibrated before any model is scored) --------------------------
geneval-setup:      # official GenEval evaluator + Mask2Former, built for sm_90 (E1)
	sbatch --qos=gpulongjobs scripts/bench/geneval_setup.sbatch
geneval-calibrate:  # must reproduce SD v1.5 = 0.43 +- 0.02 before scoring any model
	sbatch --qos=gpulongjobs --export=ALL,CALIBRATE=1 scripts/bench/geneval_score.sbatch
dpg-setup:          # ELLA's DPG-Bench scorer + mPLUG-large, verified lock file (D1)
	sbatch --qos=gpulongjobs scripts/bench/dpg_setup.sbatch
dpg-calibrate:      # must reproduce SD v1.5 = 63.18 +- 2.0 before scoring any model
	sbatch --qos=gpulongjobs --export=ALL,CALIBRATE=1 scripts/bench/dpg_score.sbatch

# --- Level B data ---------------------------------------------------------------------
imagenet-prefetch:  # ILSVRC/imagenet-1k (gated; approve ambujm22 first) + ADM ref batch
	sbatch --qos=gpulongjobs scripts/imagenet/prefetch.sbatch
