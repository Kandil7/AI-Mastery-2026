# AI-Mastery-2026 Makefile
# Comprehensive development commands
# ================================

.PHONY: help install setup clean test lint format docs build docker run

# =============================================================================
# HELP
# =============================================================================

help:
	@echo "AI-Mastery-2026 Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Setup:"
	@echo "  install              Install all dependencies"
	@echo "  setup-dev            Setup development environment"
	@echo "  setup-pre-commit     Install pre-commit hooks"
	@echo "  setup-jupyter        Register Jupyter kernel"
	@echo "  setup-full           Complete setup (install + pre-commit + jupyter)"
	@echo ""
	@echo "Testing:"
	@echo "  test                 Run all tests"
	@echo "  test-unit            Run unit tests"
	@echo "  test-integration     Run integration tests"
	@echo "  test-e2e             Run end-to-end tests"
	@echo "  test-cov             Run tests with coverage"
	@echo "  test-cov-strict      Run tests with 95% coverage requirement"
	@echo "  test-watch           Run tests in watch mode"
	@echo "  test-benchmarks      Run performance benchmarks"
	@echo "  test-file            Run specific test file (file=<path>)"
	@echo "  test-k               Run tests matching pattern (pattern=<name>)"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint                 Run all linters"
	@echo "  lint-quick           Quick lint check"
	@echo "  format               Format code"
	@echo "  format-check         Check formatting"
	@echo "  type-check           Run type checker"
	@echo "  type-check-strict    Run strict type checking"
	@echo "  security-check       Run security scanner"
	@echo "  check-all            Run all checks"
	@echo ""
	@echo "Documentation:"
	@echo "  docs                 Build documentation"
	@echo "  docs-serve           Serve documentation locally"
	@echo "  api-docs             Generate API documentation"
	@echo "  docs-clean           Clean documentation build"
	@echo ""
	@echo "Development:"
	@echo "  run-api              Run API server"
	@echo "  run-api-prod         Run API server in production mode"
	@echo "  run-streamlit        Run Streamlit app"
	@echo "  run-all              Run all services"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build         Build Docker images"
	@echo "  docker-run           Run with Docker Compose"
	@echo "  docker-stop          Stop Docker Compose"
	@echo "  docker-clean         Clean Docker resources"
	@echo "  docker-logs          View Docker logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean                Clean generated files"
	@echo "  clean-all            Clean everything including venv"
	@echo "  clean-pycache        Clean Python cache"
	@echo "  clean-test           Clean test artifacts"
	@echo ""
	@echo "Utilities:"
	@echo "  verify-install       Verify installation"
	@echo "  list-modules         List available modules"
	@echo "  check-deps           Check dependencies"
	@echo "  update-deps          Update dependencies"

# =============================================================================
# SETUP
# =============================================================================

install:
	@echo "Installing dependencies..."
	pip install -e ".[all]"
	pip install -r requirements-dev.txt 2>/dev/null || true
	@echo "Dependencies installed!"

setup-pre-commit:
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "Pre-commit hooks installed!"

setup-jupyter:
	@echo "Registering Jupyter kernel..."
	python -m ipykernel install --user --name ai-mastery-2026 --display-name "AI-Mastery-2026"
	@echo "Jupyter kernel registered!"

setup-dev: install setup-pre-commit
	@echo ""
	@echo "Development environment ready!"
	@echo "Run 'make setup-jupyter' to register Jupyter kernel"

setup-full: install setup-pre-commit setup-jupyter
	@echo ""
	@echo "Complete setup finished!"

# =============================================================================
# TESTING
# =============================================================================

test:
	@echo "Running all tests..."
	pytest tests/ -v --tb=short

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v --tb=short

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v --tb=short

test-e2e:
	@echo "Running end-to-end tests..."
	pytest tests/e2e/ -v --tb=short

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "Coverage report: htmlcov/index.html"

test-cov-strict:
	@echo "Running tests with strict coverage (95%)..."
	pytest tests/ -v --cov=src --cov-fail-under=95 --cov-report=term-missing

test-watch:
	@echo "Running tests in watch mode..."
	ptw -- --cov=src

test-benchmarks:
	@echo "Running performance benchmarks..."
	pytest tests/benchmarks/ -v --benchmark-only

test-file:
	@if [ -z "$(file)" ]; then \
		echo "Error: Please specify file=<path>"; \
		exit 1; \
	fi
	@echo "Running tests in $(file)..."
	pytest $(file) -v --tb=short

test-k:
	@if [ -z "$(pattern)" ]; then \
		echo "Error: Please specify pattern=<name>"; \
		exit 1; \
	fi
	@echo "Running tests matching $(pattern)..."
	pytest tests/ -v -k "$(pattern)" --tb=short

# =============================================================================
# CODE QUALITY
# =============================================================================

lint:
	@echo "Running linters..."
	@echo "Checking formatting..."
	black src/ tests/ --check
	isort src/ tests/ --check
	@echo ""
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203
	@echo ""
	@echo "Running mypy..."
	mypy src/ --ignore-missing-imports

lint-quick:
	@echo "Running quick lint check..."
	black src/ tests/ --check --quiet
	isort src/ tests/ --check --quiet

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "Code formatted!"

format-check:
	@echo "Checking code formatting..."
	black src/ tests/ --check
	isort src/ tests/ --check

type-check:
	@echo "Running type checker..."
	mypy src/ --ignore-missing-imports --pretty

type-check-strict:
	@echo "Running strict type checker..."
	mypy src/ --strict --ignore-missing-imports --pretty

security-check:
	@echo "Running security scanner..."
	bandit -r src/ -ll -ii
	@echo ""
	@echo "Checking for known vulnerabilities..."
	safety check --full-report 2>/dev/null || echo "Safety not installed, skipping"

check-all: format-check lint type-check security-check
	@echo ""
	@echo "All checks passed!"

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs:
	@echo "Building documentation..."
	mkdocs build

docs-serve:
	@echo "Serving documentation at http://localhost:8000..."
	mkdocs serve

api-docs:
	@echo "Generating API documentation..."
	pdoc --html --output-dir docs/api src/ 2>/dev/null || \
	pdoc3 --html --output-dir docs/api src/ 2>/dev/null || \
	echo "pdoc not installed, install with: pip install pdoc3"

docs-clean:
	@echo "Cleaning documentation build..."
	rm -rf site/
	rm -rf docs/_build/
	rm -rf docs/api/

# =============================================================================
# DEVELOPMENT
# =============================================================================

run-api:
	@echo "Starting API server..."
	uvicorn src.production.api:app --reload --host 0.0.0.0 --port 8000

run-api-prod:
	@echo "Starting API server in production mode..."
	uvicorn src.production.api:app --host 0.0.0.0 --port 8000 --workers 4

run-streamlit:
	@echo "Starting Streamlit app..."
	streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0

run-all:
	@echo "Starting all services..."
	docker-compose up

# =============================================================================
# DOCKER
# =============================================================================

docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-run:
	@echo "Starting Docker Compose..."
	docker-compose up

docker-stop:
	@echo "Stopping Docker Compose..."
	docker-compose down

docker-clean:
	@echo "Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f

docker-logs:
	@echo "Showing Docker logs..."
	docker-compose logs -f

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ site/ 2>/dev/null || true
	@echo "Cleanup complete!"

clean-all: clean
	@echo "Cleaning everything including virtual environment..."
	rm -rf .venv/ 2>/dev/null || true
	rm -rf venv/ 2>/dev/null || true
	@echo "Complete cleanup finished!"

clean-pycache:
	@echo "Cleaning Python cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Python cache cleaned!"

clean-test:
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache/ 2>/dev/null || true
	rm -rf .coverage 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf pytest_cache/ 2>/dev/null || true
	@echo "Test artifacts cleaned!"

# =============================================================================
# UTILITIES
# =============================================================================

verify-install:
	@echo "Verifying installation..."
	@echo ""
	python -c "import numpy; print('✓ numpy:', numpy.__version__)"
	python -c "import torch; print('✓ torch:', torch.__version__)"
	python -c "import fastapi; print('✓ fastapi:', fastapi.__version__)"
	python -c "import src.core; print('✓ src.core')"
	python -c "import src.ml; print('✓ src.ml')"
	python -c "import src.llm; print('✓ src.llm')"
	python -c "import src.rag; print('✓ src.rag')"
	python -c "import src.utils; print('✓ src.utils')"
	@echo ""
	@echo "Installation verified!"

list-modules:
	@echo "AI-Mastery-2026 Modules"
	@echo "======================="
	@echo ""
	python -c "from src import print_module_tree; print_module_tree()"

check-deps:
	@echo "Checking dependencies..."
	pip check
	pip list --outdated

update-deps:
	@echo "Updating dependencies..."
	pip install --upgrade -r requirements.txt
	pip install --upgrade -r requirements-dev.txt 2>/dev/null || true
	@echo "Dependencies updated!"

# =============================================================================
# CI/CD
# =============================================================================

ci: check-all test-cov-strict
	@echo ""
	@echo "CI checks passed!"

cd: ci docker-build
	@echo ""
	@echo "CD pipeline ready!"
