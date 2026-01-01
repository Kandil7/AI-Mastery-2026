# Makefile for AI-Mastery-2026

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Run linting
lint:
	black src/ tests/ --check
	mypy src/
	flake8 src/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Build Docker images
docker-build:
	docker build -t ai-mastery-2026 .

# Run with docker-compose
docker-run:
	docker-compose up --build

# Run API server locally
run:
	uvicorn src.production.api:app --reload --host 0.0.0.0 --port 8000

# Generate documentation
docs:
	sphinx-build -b html docs/ docs/_build/

# Clean generated files
clean:
	rm -rf docs/_build/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf results/

# Run performance benchmarks
benchmark:
	python scripts/run_benchmarks.py --iterations 100

# Run benchmarks and save results
benchmark-save:
	python scripts/run_benchmarks.py --iterations 100 --output results/benchmark_report.json

# Run production server (multiple workers)
run-prod:
	uvicorn src.production.api:app --host 0.0.0.0 --port 8000 --workers 4

# Check formatting without applying
format-check:
	black src/ tests/ --check
	isort src/ tests/ --check

# Stop docker services  
docker-stop:
	docker-compose down

# View docker logs
docker-logs:
	docker-compose logs -f

.PHONY: install test test-cov lint format docker-build docker-run run docs clean benchmark benchmark-save run-prod format-check docker-stop docker-logs