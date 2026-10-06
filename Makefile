.PHONY: setup test prepare-data tokenizer model-check train-smoke evaluate clean lint

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
setup:
	pip install -e ".[dev]"

setup-all:
	pip install -e ".[dev,serve,ui]"

# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src/tounsilm --cov-report=term-missing

# ──────────────────────────────────────────────
# Linting
# ──────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/

lint-fix:
	ruff check --fix src/ tests/ scripts/

# ──────────────────────────────────────────────
# Data Pipeline
# ──────────────────────────────────────────────
prepare-data:
	python scripts/prepare_data.py

clean-data:
	python scripts/clean_data.py

deduplicate:
	python scripts/deduplicate.py

# ──────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────
tokenizer:
	python scripts/train_tokenizer.py

inspect-tokenizer:
	python scripts/inspect_tokenizer.py

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
model-check:
	python scripts/estimate_parameters.py
	python scripts/initialize_model.py

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
train-smoke:
	python scripts/pretrain.py --config configs/pretraining_local.yaml --smoke-test

train-local:
	python scripts/pretrain.py --config configs/pretraining_local.yaml

train-cloud:
	python scripts/pretrain.py --config configs/pretraining_cloud.yaml

resume:
	python scripts/resume_training.py --checkpoint checkpoints/latest

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
evaluate:
	python scripts/evaluate.py

# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────
generate:
	python scripts/generate.py --prompt "شنوة تعمل توّا؟"

# ──────────────────────────────────────────────
# System Info
# ──────────────────────────────────────────────
system-info:
	python scripts/system_info.py

# ──────────────────────────────────────────────
# Clean
# ──────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
