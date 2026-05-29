.PHONY: install lint type test check fmt clean ablations

install:
	pip install -e ".[dev,torch]"

lint:
	ruff check .

fmt:
	ruff check . --fix

type:
	mypy bifrost_flow/

test:
	pytest -q

# Full local gate, mirrors CI.
check: lint type test

ablations:
	python -c "from bifrost_flow.eval import dump_ablation_configs; \
		print(dump_ablation_configs('configs/ablations', base='base_gpu'))"

clean:
	rm -rf checkpoints/ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
