
# AI-Mastery-2026: The Complete Guide

Welcome to the complete guide for the `AI-Mastery-2026` toolkit. This documentation provides a deep dive into the architecture, modules, and workflows of the project.

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-repo/AI-Mastery-2026.git
cd AI-Mastery-2026
make install

# Train models and start API
python scripts/train_save_models.py
make run

# Or use Docker
make docker-run
```

## Table of Contents

1.  **[Getting Started](./01_getting_started.md)**
    *   A detailed guide on installation, environment setup, and verification.

2.  **[Core Concepts](./02_core_concepts.md)**
    *   An explanation of the "White-Box Approach" and the project's educational philosophy.

3.  **Module Documentation**
    *   **[Module: Core (`src/core`)](./03_module_core.md)**
        *   In-depth look at the mathematical and optimization foundations implemented from scratch.
    *   **[Module: Machine Learning (`src/ml`)](./04_module_ml.md)**
        *   Documentation for the from-scratch classical and deep learning algorithms.
    *   **[Module: LLM Engineering (`src/llm`)](./05_module_llm.md)**
        *   Details on the Transformer, RAG, and fine-tuning components.
    *   **[Module: Production (`src/production`)](./06_module_production.md)**
        *   Guide to the production API, monitoring tools, and vector database implementations.

4.  **[Research Notebooks](./07_research_notebooks.md)**
    *   An overview of the 17-week learning path and how to use the Jupyter notebooks.

5.  **[Contribution & Development](./08_contribution_guide.md)**
    *   A complete guide for developers on coding standards, testing, and contributing to the project.

6.  **[Deployment Guide](./09_deployment_guide.md)**
    *   Instructions for deploying the application using Docker and `docker-compose`.

7.  **[Capstone Project: GitHub Issue Classifier](./10_capstone_project.md)** ⭐ NEW
    *   Complete end-to-end project building a production ML application.

---

## Architecture Overview

```
AI-Mastery-2026/
├── src/
│   ├── core/          # Mathematical foundations
│   ├── ml/            # Classical & deep learning
│   ├── llm/           # LLM, RAG, fine-tuning
│   └── production/    # API, monitoring, vector DB
├── research/          # 17-week learning notebooks
├── scripts/           # Utility scripts
├── tests/             # Unit tests
├── config/            # Prometheus, Grafana configs
└── app/               # Streamlit web UI
```

## Key Components

| Component | File | Description |
|-----------|------|-------------|
| **API** | `src/production/api.py` | FastAPI with real model inference |
| **RAG** | `src/llm/rag.py` | Dense/Sparse/Hybrid retrieval |
| **Neural Net** | `src/ml/deep_learning.py` | Dense, LSTM, Conv2D layers |
| **Classical ML** | `src/ml/classical.py` | LR, SVM, Tree, RandomForest |
| **Web UI** | `app/main.py` | Streamlit chat interface |

