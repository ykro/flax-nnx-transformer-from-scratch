PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

.PHONY: all experiments verify test lint security clean

all: lint experiments verify test security

experiments:
	$(PYTHON) experiments/01_nnx_module_vs_torch.py
	$(PYTHON) experiments/02_graphdef_split_merge.py
	$(PYTHON) experiments/03_causal_attention_rope.py
	$(PYTHON) experiments/04_compiled_train_step.py
	$(PYTHON) experiments/05_inspect_stablehlo_recompiles.py

verify:
	$(PYTHON) verify_transformer.py

test:
	$(PYTEST) -v

lint:
	$(RUFF) check .

security:
	gitleaks detect --source . --no-git

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
