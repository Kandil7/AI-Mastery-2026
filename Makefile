# AI Engineer Toolkit 2025 - Makefile
# ===================================
# Build automation for development and deployment

.PHONY: help install test lint format build run clean docker-build docker-run docs

# Default target
help:
	@echo "AI Engineer Toolkit 2025 - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Development:"
	@echo "  make install     Install dependencies"
	@echo "  make test        Run all tests"
	@echo "  make lint        Run linting checks"
	@echo "  make format      Format code"
	@echo "  make docs        Generate documentation"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-run     Run with docker-compose"
	@echo "  make docker-stop    Stop all containers"
	@echo ""
	@echo "Production:"
	@echo "  make build       Build production artifacts"
	@echo "  make run         Run API server locally"
	@echo "  make clean       Clean build artifacts"

# ============================================================
# DEVELOPMENT
# ============================================================

# Install all dependencies
install:
	pip install -r requirements.txt
	pip install -e .
	@echo "Dependencies installed successfully"

# Install development dependencies
install-dev: install
	pip install pytest pytest-cov black flake8 mypy isort
	@echo "Development dependencies installed"

# Run all tests
test:
	pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# Run specific test file
test-file:
	@test -n "$(FILE)" || (echo "Usage: make test-file FILE=tests/test_linear_algebra.py" && exit 1)
	pytest $(FILE) -v

# ============================================================
# CODE QUALITY
# ============================================================

# Lint code with flake8
lint:
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	mypy src/ --ignore-missing-imports

# Format code with black and isort
format:
	isort src/ tests/
	black src/ tests/ --line-length=100

# Check formatting without making changes
format-check:
	isort --check-only src/ tests/
	black --check src/ tests/ --line-length=100

# ============================================================
# DOCKER
# ============================================================

# Build Docker images
docker-build:
	docker-compose build

# Run all services
docker-run:
	docker-compose up -d
	@echo "Services started. API: http://localhost:8000, Jupyter: http://localhost:8888"

# Stop all services
docker-stop:
	docker-compose down

# View logs
docker-logs:
	docker-compose logs -f

# Clean Docker resources
docker-clean:
	docker-compose down -v --rmi local
	docker system prune -f

# ============================================================
# PRODUCTION
# ============================================================

# Build production artifacts
build:
	python setup.py sdist bdist_wheel
	@echo "Build artifacts created in dist/"

# Run API server locally
run:
	uvicorn src.production.api:app --host 0.0.0.0 --port 8000 --reload

# Run API server (production mode)
run-prod:
	uvicorn src.production.api:app --host 0.0.0.0 --port 8000 --workers 4

# Start Jupyter Lab
jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# ============================================================
# DOCUMENTATION
# ============================================================

# Generate documentation
docs:
	cd docs && make html
	@echo "Documentation generated in docs/_build/html/"

# Serve documentation locally
docs-serve:
	python -m http.server 8080 --directory docs/_build/html/

# ============================================================
# DATA & MODELS
# ============================================================

# Download sample datasets
download-data:
	python scripts/data_preprocessing/download_sample_datasets.py

# Generate synthetic data
generate-data:
	python scripts/data_preprocessing/generate_synthetic_data.py

# Train models
train:
	python scripts/model_training/train_model.py

# ============================================================
# CLEANUP
# ============================================================

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned build artifacts"

# Clean all (including Docker)
clean-all: clean docker-clean
	@echo "All artifacts cleaned"
