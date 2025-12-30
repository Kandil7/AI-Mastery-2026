# Guide: Research Notebooks

The `research/` directory contains a comprehensive set of Jupyter notebooks that form a 17-week (Week 0 to Week 16) practical learning path. These notebooks are designed to be hands-on labs where you can experiment with the concepts implemented in the `src/` modules.

Each weekly directory contains a notebook that aligns with the project's "White-Box Approach." You will often find mathematical derivations, from-scratch coding exercises, and comparisons with production-level libraries.

## How to Use the Notebooks

1.  **Activate Your Environment:** Make sure you have followed the [Getting Started](./01_getting_started.md) guide and have activated your virtual environment (`.venv`).
2.  **Start Jupyter Lab:** The easiest way to run the notebooks is by using the Makefile command, which starts a Jupyter Lab instance.
    ```bash
    make jupyter
    ```
    Alternatively, if you used `make docker-run`, the Jupyter Lab instance is already running and accessible at `http://localhost:8888`.
3.  **Navigate and Explore:** Once Jupyter Lab is open, navigate to the `research/` directory in the file browser on the left and open any of the weekly notebooks to get started.

## Weekly Breakdown

This is a high-level overview of the topics covered in the research notebooks.

*   **Week 0: Math (`research/week0-math`)**
    *   Focuses on the mathematical foundations of AI, likely covering linear algebra concepts like matrix multiplication.

*   **Week 1: Similarity (`research/week1-similarity`)**
    *   Explores vector similarity metrics, a core concept for search and retrieval.

*   **Week 2: Probability (`research/week2-probability`)**
    *   Covers fundamental probability theory necessary for understanding machine learning models.

*   **Week 3: ML Core (`research/week3-ml-core`)**
    *   Dives into core machine learning concepts, with a focus on embeddings.

*   **Week 4: Transformers (`research/week4-transformers`)**
    *   A deep dive into the Transformer architecture, likely complementing the code in `src/llm/attention.py`.

*   **Week 5: Backend (`research/week5-backend`)**
    *   Focuses on backend development, likely showing how to serve models using FastAPI, connecting with `src/production/api.py`.

*   **Week 6: VectorDB (`research/week6-vectordb`)**
    *   Hands-on experiments with vector databases, likely using the implementations from `src/production/vector_db.py`.

*   **Week 7: Retrieval (`research/week7-retrieval`)**
    *   Covers advanced retrieval techniques, such as hybrid search.

*   **Week 8: Reranking (`research/week8-reranking`)**
    *   Explores the reranking stage of a RAG pipeline.

*   **Week 9: Orchestration (`research/week9-orchestration`)**
    *   Focuses on orchestrating complex AI workflows, possibly with tools like LangChain or custom pipelines.

*   **Week 10: Evaluation (`research/week10-evaluation`)**
    *   Covers methods for evaluating model performance and the financial costs associated with different models.

*   **Week 11: Flutter UI (`research/week11-flutter-ui`)**
    *   Explores building a user interface for the AI backend, possibly with Flutter.

*   **Week 12: Flutter Voice (`research/week12-flutter-voice`)**
    *   Focuses on integrating voice capabilities into a UI.

*   **Week 13: Deployment (`research/week13-deployment`)**
    *   Practical guide to deploying the models and API.

*   **Week 14: Monitoring (`research/week14-monitoring`)**
    *   Hands-on with the concepts from `src/production/monitoring.py`, such as setting up drift detection.

*   **Week 15: Feedback (`research/week15-feedback`)**
    *   Covers implementing user feedback loops to improve model performance over time.

*   **Week 16: Documentation (`research/week16-documentation`)**
    *   Focuses on the importance and practice of documenting AI/ML projects.

This structured, week-by-week guide provides a clear path from fundamental theory to a fully deployed and monitored production system.
