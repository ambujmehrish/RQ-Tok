.PHONY: install lint type test check fmt clean ablations smoke prefetch e0 vhpc-setup vhpc-env vhpc-prefetch

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
		print(dump_ablation_configs('configs/ablations', base='base_gpu'))"

clean:
	rm -rf checkpoints/ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# End-to-end smoke test on REAL data (real COCO images -> real CLIP -> tokenizer -> E0).
smoke:
	bash scripts/smoke_test.sh

# Prefetch models/images to $ADARQ_CACHE (outside $HOME). RUN ON A LOGIN NODE.
prefetch:
	bash scripts/prefetch_assets.sh

# Submit the powered E0 downstream run (64 images x 3 seeds) on Leonardo.
e0:
	sbatch scripts/e0_downstream_64.sbatch

# --- vhpc01 (3x H200, submit-only SLURM) ---------------------------------------------
vhpc-setup:      # step 1: survey partitions / GPUs / egress from a compute node
	sbatch scripts/vhpc01_setup.sbatch
vhpc-env:        # step 2: build + verify the conda env on an H200
	sbatch scripts/vhpc01_env.sbatch
vhpc-prefetch:   # step 3: fetch models + data (needs ADARQ_CACHE and compute egress)
	sbatch scripts/vhpc01_prefetch.sbatch
