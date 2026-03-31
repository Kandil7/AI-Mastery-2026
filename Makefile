# Makefile for AI-Mastery-2026
# =============================
# Common development tasks
#
# Usage:
#   make help          # Show this help
#   make install       # Install package
#   make test          # Run tests
#   make lint          # Run linters
#   make docs          # Build documentation
#   make clean         # Clean build artifacts

.PHONY: help install install-dev test test-cov lint format docs clean build release

# Default target
help:
	@echo "AI-Mastery-2026 Development Tasks"
	@echo "=================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install        Install package in editable mode"
	@echo "  make install-dev    Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-int       Run integration tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run all linters"
	@echo "  make format         Format code with black and isort"
	@echo "  make check          Run all checks (lint + test)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build documentation"
	@echo "  make docs-serve     Serve documentation locally"
	@echo ""
	@echo "Build & Release:"
	@echo "  make build          Build package distributions"
	@echo "  make clean          Clean build artifacts"
	@echo "  make release        Create new release (use with care)"
	@echo ""
	@echo "Utilities:"
	@echo "  make pre-commit     Install pre-commit hooks"
	@echo "  make env            Create .env from template"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit -v --tb=short

test-int:
	pytest tests/integration -v --tb=short

# Code Quality
lint:
	black --check src tests
	isort --check src tests
	flake8 src tests
	mypy src --ignore-missing-imports

format:
	black src tests
	isort src tests

check: lint test

# Documentation
docs:
	mkdocs build

docs-serve:
	mkdocs serve

# Build & Release
build:
	python -m build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type d -name .coverage -exec rm -rf {} +
	find . -type d -name dist -exec rm -rf {} +
	find . -type d -name build -exec rm -rf {} +
	find . -type d -name *.egg-info -exec rm -rf {} +
	rm -rf site/

release:
	@echo "Creating release..."
	@read -p "Enter version (e.g., 0.2.0): " version; \
	./scripts/release.sh $$version

# Utilities
pre-commit:
	pre-commit install

env:
	cp examples/.env.example examples/.env
	@echo "Created examples/.env from template"

# Docker
docker-build:
	docker build -t ai-mastery:latest .

docker-run:
	docker run -d -p 8000:8000 ai-mastery:latest

docker-clean:
	docker stop ai-mastery || true
	docker rm ai-mastery || true

# Full CI pipeline locally
ci: clean install-dev lint test-cov docs
	@echo "CI pipeline completed successfully!"
