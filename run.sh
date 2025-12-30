#!/bin/bash
# =============================================================================
# Run Script for AI Engineer Toolkit
# =============================================================================
# Quick start script for common development tasks.

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[RUN]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# =============================================================================
# COMMANDS
# =============================================================================

help() {
    echo "AI Engineer Toolkit 2025 - Run Script"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  api         Start the FastAPI server"
    echo "  jupyter     Start Jupyter Lab"
    echo "  docker      Start all services with Docker Compose"
    echo "  test        Run all tests"
    echo "  lint        Run linting and formatting"
    echo "  generate    Generate synthetic data"
    echo "  train       Train a model"
    echo "  benchmark   Run inference benchmarks"
    echo "  help        Show this help message"
    echo ""
}

api() {
    log "Starting FastAPI server..."
    uvicorn src.production.api:app --reload --host 0.0.0.0 --port 8000
}

jupyter() {
    log "Starting Jupyter Lab..."
    jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
}

docker_up() {
    log "Starting Docker services..."
    docker-compose up -d
    log "Services started. API: http://localhost:8000, Jupyter: http://localhost:8888"
}

run_tests() {
    log "Running tests..."
    python -m pytest tests/ -v --tb=short
}

lint() {
    log "Running linting..."
    python -m flake8 src/ --count --select=E9,F63,F7,F82 --show-source
    python -m black src/ --check || true
    log "Lint complete."
}

generate_data() {
    log "Generating synthetic data..."
    python scripts/data_preprocessing/generate_synthetic_data.py --output-dir data/synthetic
}

train_model() {
    log "Training model..."
    python scripts/model_training/train_model.py "$@"
}

run_benchmark() {
    log "Running benchmarks..."
    python benchmarks/inference_optimization/vllm_vs_tgi.py
    python benchmarks/cost_performance_tradeoffs/model_size_vs_latency.py
}

# =============================================================================
# MAIN
# =============================================================================

case "${1:-help}" in
    api)        api ;;
    jupyter)    jupyter ;;
    docker)     docker_up ;;
    test)       run_tests ;;
    lint)       lint ;;
    generate)   generate_data ;;
    train)      shift; train_model "$@" ;;
    benchmark)  run_benchmark ;;
    help|*)     help ;;
esac
