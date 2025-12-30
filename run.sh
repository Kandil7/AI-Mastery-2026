#!/bin/bash

# Run script for AI-Mastery-2026
# Provides convenient commands for common tasks

set -e  # Exit on any error

case "$1" in
    "api")
        echo "Starting FastAPI server..."
        uvicorn src.production.api:app --reload --host 0.0.0.0 --port 8000
        ;;
    "jupyter")
        echo "Starting Jupyter Lab..."
        jupyter lab
        ;;
    "test")
        echo "Running tests..."
        pytest tests/ -v
        ;;
    "test-cov")
        echo "Running tests with coverage..."
        pytest tests/ -v --cov=src --cov-report=html
        ;;
    "lint")
        echo "Running linters..."
        black src/ tests/ --check
        mypy src/
        flake8 src/
        ;;
    "format")
        echo "Formatting code..."
        black src/ tests/
        isort src/ tests/
        ;;
    "docker")
        echo "Starting services with Docker Compose..."
        docker-compose up --build
        ;;
    "docs")
        echo "Generating documentation..."
        make -C docs html
        ;;
    *)
        echo "Usage: $0 {api|jupyter|test|test-cov|lint|format|docker|docs}"
        echo "  api      - Start the FastAPI server"
        echo "  jupyter  - Start Jupyter Lab"
        echo "  test     - Run tests"
        echo "  test-cov - Run tests with coverage"
        echo "  lint     - Run linters"
        echo "  format   - Format code"
        echo "  docker   - Start services with Docker Compose"
        echo "  docs     - Generate documentation"
        exit 1
        ;;
esac