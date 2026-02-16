# User Guide

Welcome to the User Guide for the AI-Mastery-2026 project! This guide will help you get started with the repository, understand its components, and navigate through the learning materials and code.

## 1. Project Overview

AI-Mastery-2026 is a comprehensive AI Engineer Toolkit and learning repository. It emphasizes a "White-Box Approach" to understanding fundamental AI mechanics before using high-level abstractions. The project covers a wide curriculum from mathematical foundations, classical machine learning, deep learning, modern LLM engineering, and production deployment.

## 2. Getting Started

### 2.1. Prerequisites

Before you begin, ensure you have the following installed on your system:

*   **Python 3.10+**: The primary language for this project, essential for running all AI/ML models and applications. You can download it from [python.org](https://www.python.org/).
*   **Git**: A version control system necessary for cloning the repository and managing code changes.
*   **Make**: A build automation tool that simplifies complex commands for setting up the environment, running tests, and managing applications.
*   **Docker (Optional but Recommended)**: A platform for developing, shipping, and running applications in containers, ensuring consistent environments across different machines.

### 2.2. Environment Setup

The project uses a `Makefile` to streamline environment setup.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kandil7/AI-Mastery-2026.git
    cd AI-Mastery-2026
    ```
2.  **Install dependencies:**
    ```bash
    make install
    ```
    This command performs the following actions:
    *   Creates a Python virtual environment (`.venv`).
    *   Installs all necessary Python packages listed in `requirements.txt`.
    *   Sets up `pre-commit` hooks for automatic code quality checks.

3.  **Verify installation:**
    ```bash
    make test
    ```
    This runs the unit tests and ensures your environment is correctly configured. A successful run confirms readiness.

### Troubleshooting `make install`

*   **`make` command not found:** If you see an error like "`make` is not recognized...", ensure that `make` is installed and added to your system's PATH. On Windows, you might need to install build tools like MinGW or Cygwin.
*   **Python package installation errors:** If `requirements.txt` packages fail to install, try updating `pip` (`python -m pip install --upgrade pip`) or resolving dependency conflicts manually. Ensure you have a stable internet connection.
*   **Virtual environment issues:** If `.venv` is not created or activated correctly, try deleting the `.venv` directory and running `make install` again.

### 2.3. Running the Application

The primary application is a FastAPI server, providing API endpoints for trained models and AI logic.

*   **Development Mode (with auto-reload):**
    ```bash
    make run
    ```
    This command starts the FastAPI server with auto-reload enabled, making development quicker. The API will be available at `http://localhost:8000`.

*   **Production Mode (with multiple workers):**
    ```bash
    make run-prod
    ```
    This command starts the FastAPI server using Uvicorn with multiple worker processes, suitable for production environments.

*   **Using Docker (Recommended for Consistency):**
    ```bash
    make docker-run  # Builds Docker images and starts all services via docker-compose
    make docker-stop # Stops all running Docker services
    make docker-logs # Views logs from all Docker services
    ```

### 2.4. Makefile Overview

The `Makefile` is central to managing this project. It encapsulates common commands for:
*   **`install`**: Environment setup.
*   **`test`**: Running tests and checking coverage.
*   **`lint` / `format`**: Maintaining code quality and style.
*   **`run`**: Starting the FastAPI server.
*   **`docker-*`**: Docker image and container management.
*   **`docs`**: Building documentation.

## 3. Repository Structure & Navigation

The repository is highly organized, with the `docs/` directory serving as the central hub for all documentation. It's structured into distinct, numbered sections to facilitate a structured learning and development path. To effectively navigate the documentation, start with the "Docs Entry Point" in the main `docs/README.md` file. Each numbered section below provides a focused area of learning or project aspect, guiding you progressively through the AI-Mastery-2026 curriculum.

*   **[Docs Entry Point](../README.md)**: Start here for an overview of the entire documentation suite.
*   **[00. Introduction](./README.md)**: Project overview, setup, contribution guide.
*   **[01. Learning Roadmap](../01_learning_roadmap/README.md)**: Detailed learning path, phase-by-phase curriculum, project roadmaps.
*   **[02. Core Concepts](../02_core_concepts/README.md)**: Mathematical foundations, ML/DL fundamentals, deep dives.
*   **[03. System Design](../03_system_design/README.md)**: Architectural patterns, deployment strategies, MLOps.
*   **[04. Tutorials](../04_tutorials/README.md)**: Practical guides, examples, troubleshooting.
*   **[05. Interview Preparation](../05_interview_prep/README.md)**: Coding, ML theory, system design questions.
*   **[06. Case Studies](../06_case_studies/README.md)**: Real-world AI engineering applications.

## 4. Key Modules (Top-Level Directories)

*   **`app/`**: Contains the main FastAPI application entry point (`main.py`).
*   **`benchmarks/`**: Scripts and results for performance and cost benchmarking.
*   **`config/`**: Configuration files for various tools and environments (e.g., Prometheus, Grafana).
*   **`models/`**: Stores trained machine learning model artifacts (e.g., `.joblib`, `.pt`).
*   **`notebooks/`**: Jupyter notebooks for hands-on learning, experimentation, and data exploration. Organized into numbered foundational stages.
*   **`research/`**: Dedicated area for independent research projects and longer-term development sprints (e.g., `rag_engine`).
*   **`scripts/`**: Utility scripts for common tasks like data preprocessing, training, evaluation, and system operations.
*   **`src/`**: The heart of the project, containing from-scratch implementations of core mathematical (`core/`), machine learning (`ml/`), deep learning/LLM (`llm/`), and production-related (`production/`) components.
*   **`tests/`**: Comprehensive unit, integration, and end-to-end tests for validating the codebase.
*   **`templates/`**: Reusable project templates or documentation templates.

## 5. Contributing

We welcome contributions! Please refer to the [Contribution Guidelines](../00_introduction/CONTRIBUTING.md) for details on how to contribute to this project, including code standards, testing procedures, and submission processes.

## 6. Further Assistance & Troubleshooting

If you encounter any issues, have questions, or need further assistance:

*   **Check the Documentation:** Always start by searching within the relevant sections of this `docs/` directory.
*   Review the [Troubleshooting Guides](../04_tutorials/troubleshooting/README.md): This section contains common failure modes and troubleshooting guides.
*   **Verify Environment:** Ensure your environment is correctly set up as per Section 2.2.
*   **Create a GitHub Issue:** If you cannot find a solution, please open an issue on the project's GitHub page, providing as much detail as possible.
