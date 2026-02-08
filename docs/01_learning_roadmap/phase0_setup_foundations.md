# Phase 0: Project Setup & Foundational Readiness

**Objective:** Establish development environment, understand repository structure, and review essential mathematical foundations.

## Key Topics:

*   **Environment Setup:** Setting up your local development environment, including Docker, Git/GitHub workflow, Python virtual environments, and VS Code configurations. Refer to the [User Guide](../00_introduction/01_user_guide.md) for detailed step-by-step instructions, especially the `make install` command for quick setup.
*   **Repository Navigation:** Understanding the project's modular structure and how to navigate key directories like:
        *   `src/`: Core implementations (algorithms, utilities).
        *   `notebooks/`: Jupyter notebooks for learning and experimentation.
        *   `docs/`: All project documentation.
        *   `research/`: Independent research projects and sprints.
        *   `scripts/`: Utility and automation scripts.
        *   `tests/`: Unit and integration tests.
        *   `models/`: Saved model artifacts.
        *   `app/`: FastAPI application entry point.
        *   `config/`: Configuration files.
        *   `templates/`: Reusable code templates.
        *   `benchmarks/`: Performance and cost benchmarking.
        Familiarize yourself with the `Makefile` commands for common operations (e.g., `make install`, `make test`, `make run`).
*   **Math Refresher:** Reviewing essential mathematical foundations for AI, including:
        *   **Linear Algebra:** Vectors, matrices, common operations (dot product, matrix multiplication).
        *   **Calculus:** Gradients, derivatives, partial derivatives, chain rule (crucial for backpropagation).
        *   **Probability & Statistics:** Basic distributions (normal, binomial), hypothesis testing, descriptive statistics.
        Consider working through problems that apply these concepts, e.g., implementing a simple dot product from scratch.

## Deliverables:

*   Working development environment.
*   Familiarity with project navigation and key tools.
*   Review of [`docs/02_core_concepts/fundamentals/math_notes.md`](../02_core_concepts/fundamentals/math_notes.md) and understanding of core mathematical concepts.
*   Completion of key notebooks from [`notebooks/01_foundations/`](../../notebooks/01_foundations/), such as those covering basic NumPy operations, vector manipulations, and simple function plotting.
