# Arabic LLM - Makefile
# Common commands for development and testing

.PHONY: help install dev test lint format clean docs build

# Default target
help:
	@echo "Arabic LLM - Available Commands"
	@echo ""
	@echo "  make install      - Install package"
	@echo "  make dev          - Install development dependencies"
	@echo "  make test         - Run tests"
	@echo "  make test-verbose - Run tests with verbose output"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docs         - Generate documentation"
	@echo "  make build        - Build package"
	@echo "  make audit        - Audit datasets"
	@echo "  make process      - Process books"
	@echo "  make generate     - Generate dataset"
	@echo "  make train        - Train model"
	@echo "  make agent        - Run autonomous agent"
	@echo ""

# Installation
install:
	@echo "Installing Arabic LLM package..."
	pip install -e .

dev: install
	@echo "Installing development dependencies..."
	pip install -e ".[dev]"
	pip install pytest pytest-cov black flake8 mypy pre-commit

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v

test-verbose:
	@echo "Running tests with verbose output..."
	pytest tests/ -vv --tb=long

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=arabic_llm --cov-report=html

# Linting
lint:
	@echo "Running linters..."
	flake8 arabic_llm/ tests/
	mypy arabic_llm/

format:
	@echo "Formatting code..."
	black arabic_llm/ tests/ scripts/ examples/

# Cleaning
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Documentation files are in docs/ directory"
	@echo "See docs/COMPLETE_DOCUMENTATION.md"

# Building
build:
	@echo "Building package..."
	python -m build

# Dataset commands
audit:
	@echo "Auditing datasets..."
	python arabic_llm/scripts/audit_datasets.py

process:
	@echo "Processing books..."
	python arabic_llm/scripts/process_books.py \
		--books-dir ../datasets/extracted_books \
		--metadata-dir ../datasets/metadata \
		--output-dir data/processed

generate:
	@echo "Generating dataset..."
	python arabic_llm/scripts/generate_dataset.py \
		--input-dir data/processed \
		--output-dir data/jsonl \
		--target-examples 50000

train:
	@echo "Training model..."
	python arabic_llm/scripts/train_model.py \
		--dataset data/jsonl/train.jsonl \
		--output-dir models/arabic-linguist-v1

agent:
	@echo "Running autonomous agent..."
	python arabic_llm/scripts/agent.py \
		--experiments 100 \
		--time-per-exp 300

# Pre-commit
pre-commit-install:
	@echo "Installing pre-commit hooks..."
	pre-commit install

pre-commit-run:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files

# Quick commands
quick-test:
	pytest tests/test_schema.py -v

quick-lint:
	flake8 arabic_llm/core/ arabic_llm/utils/

# Development workflow
setup: install dev pre-commit-install
	@echo "Setup complete!"

check: lint test
	@echo "All checks passed!"

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t arabic-llm:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -it --rm arabic-llm:latest bash

# Help
.DEFAULT_GOAL := help
