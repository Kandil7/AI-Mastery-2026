# AI-Mastery-2026: The Ultimate AI Engineering Toolkit

<div align="center">

![CI](https://github.com/Kandil7/AI-Mastery-2026/workflows/Main%20Branch%20CI%2FCD/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.78+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![Coverage](https://img.shields.io/badge/Coverage-85%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Week%202%3A%20COMPLETE%20%E2%9C%85-brightgreen)

**From First Principles to Production Scale.**
*A 17-week roadmap to master the mathematics, algorithms, and engineering of modern AI.*

[Quick Start](#-quick-start) • [Roadmap](#-17-week-roadmap) • [Modules](#-module-deep-dive) • [Benchmarks](#-performance-benchmarks) • [Documentation](#-documentation)

</div>

---

## 🎯 The "White-Box" Philosophy

**Stop treating AI as a Black Box.** AI-Mastery-2026 is designed to build "Senior AI Engineer" intuition by forcing you to implement core systems from scratch before using production libraries.

1.  **Math First** → Derive gradients, proofs, and update rules on paper.
2.  **Code Second** → Implement algorithms in **Pure Python `src/core`** (No NumPy allowed initially).
3.  **Libraries Third** → Switch to `src/ml` (NumPy/PyTorch) for performance.
4.  **Production Always** → Deploy with FastAPI, Docker, and Prometheus.

---

## 🗺️ 17-Week Roadmap

| Phase | Weeks | Focus Area | Status | Notebooks |
|:---|:---|:---|:---|:---|
| **I. Foundations** | 1-2 | **Mathematics & Core ML** | ✅ **COMPLETE** | 38 notebooks |
| **II. Deep Learning** | 3 | **CNNs & Vision** | ✅ **COMPLETE** | 5 notebooks |
| **III. LLMs & RAG** | 9 | **Modern AI Applications** | ✅ **COMPLETE** | 4 notebooks |
| **IV. Production** | 13 | **MLOps & Deployment** | ✅ **COMPLETE** | 4 notebooks |
| **V. Advanced** | 4-8, 10-12, 14-17 | **Optional Extensions** | ⏳ Available | - |

**Total Progress**: 47 comprehensive notebooks (~220 KB) covering critical path from foundations to production.

---

## 🧩 Module Deep Dive

### 1. The Core (`src/core/`)
*The mathematical heart of the system. Implementations in Pure Python.*

*   **Linear Algebra**: `linear_algebra.py` - Matrix/Vector operations, Eigenvalues (Power Iteration).
*   **Optimization**: `optimization_whitebox.py` - SGD, Adam, Finite Difference Gradients.
*   **Probability**: `probability_whitebox.py` - Gaussian, Metropolis-Hastings MCMC.
*   **Causal Inference**: `causal_whitebox.py` - ATE, Inverse Probability Weighting (IPW).
*   **Explainable AI**: `explainable_ai.py` - SHAP, Lime, Integrated Gradients from scratch.
*   **Time Series**: `time_series.py` - Kalman Filters (Extended/Unscented), Particle Filters.

### 2. Machine Learning (`src/ml/`)
*Algorithms built from the ground up to understand internal mechanics.*

*   **Classical**: `classical.py` - Decision Trees, SVMs (SMO), Naive Bayes.
*   **Deep Learning**: `deep_learning.py` - Autograd engine, Dense/Conv2D/LSTM layers.
*   **Graph Neural Networks**: `gnn_integration.py` - Message passing frameworks.
*   **Reinforcement Learning**: `rl_integration.py` - PPO, DQN, Policy Gradients.

### 3. Production Engineering (`src/production/`)
*Enterprise-grade infrastructure for real-world deployment.*

*   **RAG Pipeline**: `query_enhancement.py` (HyDE, Multi-Query), `data_pipeline.py`.
*   **Performance**: `caching.py` (Semantic Cache with Redis), `vector_db.py` (HNSW).
*   **Observability**: `monitoring.py` (Prometheus/Grafana), `observability.py`.
*   **Serving**: `api.py` (FastAPI), Docker containerization.

---

## 📊 Performance Benchmarks

> "Don't just say it's fast—prove it with numbers."

### Core & Algorithms
| Component | Metric | Result | vs Baseline |
|:---|:---|:---|:---|
| **Matrix Mul** | 50x50 Latency | **0.1ms (NumPy)** | 73x faster than Python |
| **Matrix Inv** | 30x30 Latency | **0.2ms (NumPy)** | 18x faster than Python |
| **SVM Inference** | Latency (p50) | **2.3ms** | Optimized from scratch implementation |
| **LSTM Inference** | Latency (p50) | **8.4ms** | Sequence length = 20 |

### Production RAG System
| Metric | Value | Notes |
|--------|-------|-------|
| Retrieval Latency (p95) | **580ms** | Hybrid dense+sparse retrieval |
| Retrieval Faithfulness | **92%** | Measured on internal benchmark |
| Embedding Throughput | 1,200 docs/min | Using all-MiniLM-L6-v2 |
| Vector Search (10K docs) | 12ms | HNSW index, top-5 |

---

## 🚀 Quick Start

### Installation

We use a modular `setup.py` for flexibility.

```bash
# 1. Clone & Venv
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# 2. Install (Choose your flavor)
pip install -e .          # Core only (Foundations)
pip install -e ".[dev]"   # + Testing tools
pip install -e ".[llm]"   # + Heavy AI libraries (Torch, Transformers)
pip install -e ".[all]"   # Everything
```

### Running the System

```bash
# Run Week 1 Benchmarks (Pure Python vs NumPy)
python scripts/benchmark_week1.py

# Launch Full Stack App (API + Streamlit + Grafana)
docker-compose up -d --build
```
*   **Web UI**: [http://localhost:8501](http://localhost:8501)
*   **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
*   **Grafana**: [http://localhost:3000](http://localhost:3000) (admin/admin)

---

## 📖 Documentation

*   **[User Guide](docs/USER_GUIDE.md)**: The bible of this project. Steps, params, and usage.
*   **[Math Notes](docs/math_notes.md)**: Handwritten derivations of key algorithms.
*   **[Interview Prep](docs/INTERVIEW_PREP.md)**: "How would you design a recommendation system?"

---

## ✨ Feature Highlights

### Mathematical Core
```python
# Causal Inference (Inverse Probability Weighting)
from src.core.causal_whitebox import estimate_ate_ipw
ate = estimate_ate_ipw(data, propensity_score_func)

# Bayesian Inference (Metropolis-Hastings)
from src.core.probability_whitebox import metropolis_hastings
samples = metropolis_hastings(log_prob_func, n_samples=1000)
```

### Deep Learning (From Scratch)
```python
from src.ml.deep_learning import NeuralNetwork, Dense, Conv2D

model = NeuralNetwork()
model.add(Conv2D(1, 32, kernel_size=3))
model.add(Dense(32*14*14, 10))
model.fit(X_train, y_train)
```

### Production RAG
```python
from src.production.query_enhancement import HyDEGenerator
hyde = HyDEGenerator()
enhanced_query = hyde.generate("What is the capital of Mars?")
```

---

<div align="center">
    <b>Built for the 1% of Engineers who want to understand EVERYTHING down to the metal.</b>
</div>