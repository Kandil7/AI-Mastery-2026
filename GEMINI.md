# 🧠 AI-Mastery-2026 Project (`GEMINI.md`)

This document provides a comprehensive overview of the `AI-Mastery-2026` project, its structure, and key operational commands. It is intended to be used as a primary context file for AI-driven development and analysis.

---

## 1. Project Overview

**Purpose:** `AI-Mastery-2026` is a comprehensive AI Engineer Toolkit and learning repository. It is designed with a "White-Box Approach," emphasizing understanding fundamental mechanics before using high-level abstractions. The project covers a wide curriculum from mathematical foundations (linear algebra, optimization) to classical machine learning, deep learning, modern LLM engineering (Transformers, RAG, fine-tuning), and production deployment.

**Architecture:** The project follows a modular structure:
*   `src/core`: Implements mathematical and optimization algorithms from scratch using NumPy.
*   `src/ml`: Contains from-scratch implementations of classical and deep learning models.
*   `src/llm`: Focuses on LLM components like Attention mechanisms, RAG pipelines, and fine-tuning adapters (LoRA).
*   `src/production`: Handles real-world engineering concerns, including a FastAPI-based API for model serving, performance monitoring, and custom vector database implementations.
*   `research/`: A series of Jupyter notebooks organized into a 17-week learning plan.
*   `tests/`: Unit tests for all core modules, ensuring code correctness.
*   `scripts/`: Utility scripts for data preprocessing, training, and evaluation.

**Technologies:**
*   **Core:** Python 3.10+
*   **ML/DL:** PyTorch, Transformers, scikit-learn, NumPy
*   **Backend:** FastAPI, Uvicorn
*   **Vector DBs:** ChromaDB, Qdrant (with from-scratch implementations also available)
*   **Tooling:** Docker, pytest, Black, Ruff, MyPy, flake8, isort

---

## 2. Building and Running

The project uses a `Makefile` to streamline all common development and operational tasks.

### Installation

To set up the development environment and install all required dependencies:
```bash
make install
```

### Running the Application

The primary application is a FastAPI server that exposes the trained models and AI logic.

*   **Development Mode (with auto-reload):**
    ```bash
    make run
    ```
    The API will be available at `http://localhost:8000`.

*   **Production Mode (with multiple workers):**
    ```bash
    make run-prod
    ```

*   **Using Docker (Recommended):**
    ```bash
    # Build and start all services (API, Jupyter) in the background
    make docker-run

    # Stop all running services
    make docker-stop

    # View logs
    make docker-logs
    ```

---

## 3. Development Conventions

### Testing

The project uses `pytest` for testing.

*   **Run all tests:**
    ```bash
    make test
    ```

*   **Run tests with code coverage:**
    ```bash
    make test-cov
    ```
    A coverage report will be generated in the `htmlcov/` directory.

### Code Quality (Linting & Formatting)

The project enforces code quality using `flake8`, `mypy`, `isort`, and `black`.

*   **Format all code automatically:**
    ```bash

    make format
    ```

*   **Run linting and static type checks:**
    ```bash
    make lint
    ```
*   **Check formatting without applying changes:**
    ```bash
    make format-check
    ```

### Key Scripts

*   **Data Preprocessing:**
    ```bash
    # Download sample datasets
    make download-data

    # Generate synthetic data
    make generate-data
    ```

*   **Model Training:**
    ```bash
    make train
    ```
