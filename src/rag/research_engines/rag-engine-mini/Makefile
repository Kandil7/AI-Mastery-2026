# =============================================================================
# RAG Engine Mini - Makefile
# =============================================================================
# Development commands for the RAG project
# أوامر التطوير لمشروع RAG
# =============================================================================

.PHONY: help install run worker test format lint typecheck migrate seed docker-up docker-down clean

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              RAG Engine Mini - Development Commands              ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  install      Install dependencies                               ║"
	@echo "║  run          Run FastAPI server (dev mode)                      ║"
	@echo "║  run-prod     Run FastAPI server (production mode)               ║"
	@echo "║  worker       Run Celery worker                                  ║"
	@echo "║  test         Run all tests                                      ║"
	@echo "║  demo         Start Gradio frontend demo                         ║"
	@echo "║  eval         Run retrieval evaluation script                     ║"
	@echo "║  format       Format code with black + isort                     ║"
	@echo "║  lint         Lint code with ruff                                ║"
	@echo "║  migrate      Run database migrations                            ║"
	@echo "║  seed         Seed demo user                                     ║"
	@echo "║  docker-up    Start all services with Docker                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"

# =============================================================================
# Installation / التثبيت
# =============================================================================

install:
	@echo "📦 Installing dependencies..."
	pip install -e ".[dev]"
	@echo "✅ Dependencies installed successfully"

install-local:
	@echo "📦 Installing with local LLM support..."
	pip install -e ".[dev,local]"
	@echo "✅ Dependencies installed successfully"

# =============================================================================
# Running / التشغيل
# =============================================================================

run:
	@echo "🚀 Starting FastAPI server (dev mode)..."
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	@echo "🚀 Starting FastAPI server (production mode)..."
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

worker:
	@echo "⚙️ Starting Celery worker..."
	celery -A src.workers.celery_app.celery_app worker -Q indexing --loglevel=INFO

worker-beat:
	@echo "⏰ Starting Celery beat (scheduler)..."
	celery -A src.workers.celery_app.celery_app beat --loglevel=INFO

flower:
	@echo "🌸 Starting Flower (Celery monitoring)..."
	celery -A src.workers.celery_app.celery_app flower --port=5555

# =============================================================================
# Testing / الاختبار
# =============================================================================

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated in htmlcov/"

test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v

test-integration:
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v

# =============================================================================
# Code Quality / جودة الكود
# =============================================================================

format:
	@echo "🎨 Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "✅ Code formatted"

format-check:
	@echo "🔍 Checking code format..."
	black --check src/ tests/
	isort --check-only src/ tests/

lint:
	@echo "🔍 Linting code..."
	ruff check src/ tests/
	@echo "✅ Linting complete"

lint-fix:
	@echo "🔧 Fixing lint issues..."
	ruff check --fix src/ tests/
	@echo "✅ Lint issues fixed"

typecheck:
	@echo "📝 Type checking..."
	mypy src/
	@echo "✅ Type check complete"

check-all: format-check lint typecheck
	@echo "✅ All checks passed"

# =============================================================================
# Database / قاعدة البيانات
# =============================================================================

migrate:
	@echo "📊 Running database migrations..."
	alembic upgrade head
	@echo "✅ Migrations complete"

migrate-rev:
	@echo "📝 Creating new migration revision..."
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

migrate-down:
	@echo "⬇️ Rolling back last migration..."
	alembic downgrade -1

migrate-history:
	@echo "📜 Migration history..."
	alembic history

seed:
	@echo "🌱 Seeding demo user..."
	python scripts/seed_user.py
	@echo "✅ Demo user created"

eval-ragas:
	@echo "🧪 Running RAGAS evaluation..."
	python scripts/evaluate_ragas.py

generate-testset:
	@echo "🔄 Generating synthetic testset..."
	python scripts/generate_synthetic_testset.py --input data/samples/sample_doc.txt

# =============================================================================
# Docker / دوكر
# =============================================================================

docker-down:
	@echo "🐳 Stopping Docker services..."
	docker compose -f docker/docker-compose.yml down
	@echo "✅ Services stopped"

## 🚀 Production Deployment
prod-check:
	@echo "🔍 Running production readiness audit (Inside Container)..."
	docker compose -f docker/docker-compose.prod.yml run --rm api python scripts/check_prod_readiness.py

prod-up:
	@echo "🚀 Deploying Production Stack..."
	docker compose -f docker/docker-compose.prod.yml up -d
	@echo "✅ Production stack is rising"

prod-down:
	@echo "🛑 Stopping Production Stack..."
	docker compose -f docker/docker-compose.prod.yml down
	@echo "✅ Production stack stopped"

docker-logs:
	@echo "📜 Docker logs..."
	docker compose -f docker/docker-compose.yml logs -f

docker-build:
	@echo "🏗️ Building Docker image..."
	docker build -f docker/Dockerfile -t rag-engine-mini .
	@echo "✅ Image built"

docker-clean:
	@echo "🧹 Cleaning Docker resources..."
	docker compose -f docker/docker-compose.yml down -v --remove-orphans
	@echo "✅ Docker resources cleaned"

# =============================================================================
# Development / التطوير
# =============================================================================

demo:
	@echo "🎨 Starting Gradio frontend demo..."
	python scripts/frontend_demo.py

eval:
	@echo "📊 Running retrieval evaluation..."
	python scripts/eval_retrieval.py

dev-setup: docker-up migrate seed
	@echo "🎉 Development environment ready!"
	@echo "   Run 'make run' in one terminal"
	@echo "   Run 'make worker' in another terminal"
	@echo "   Run 'make demo' in a third terminal"

shell:
	@echo "🐍 Starting Python shell..."
	python -i -c "from src.core.bootstrap import get_container; c = get_container(); print('Container loaded:', list(c.keys()))"

# =============================================================================
# Cleanup / التنظيف
# =============================================================================

clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✅ Cache cleaned"

clean-all: clean docker-clean
	@echo "🧹 Cleaning uploads..."
	rm -rf uploads/* 2>/dev/null || true
	@echo "✅ All cleaned"
