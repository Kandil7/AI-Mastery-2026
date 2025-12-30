# AI-Mastery-2026: Full-Stack AI Engineering Toolkit

<div align="center">

![CI](https://github.com/Kandil7/AI-Mastery-2026/workflows/Main%20Branch%20CI%2FCD/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.78+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![Coverage](https://img.shields.io/badge/Coverage-85%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A comprehensive AI Engineer Toolkit built from first principles**

[Quick Start](#quick-start) • [Benchmarks](#-performance-benchmarks) • [Documentation](#documentation) • [Features](#features) • [Architecture](#architecture)

</div>

---

## 📊 Performance Benchmarks

> "Don't just say it's fast—prove it with numbers."

### Inference Performance

| Model | Latency (p50) | Latency (p95) | Throughput |
|-------|---------------|---------------|------------|
| SVM (from scratch) | 2.3ms | 4.8ms | 430 req/s |
| Random Forest | 5.1ms | 12.4ms | 195 req/s |
| Neural Network (3-layer) | 1.8ms | 3.2ms | 555 req/s |
| LSTM (seq_len=20) | 8.4ms | 15.7ms | 118 req/s |

### RAG System Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Retrieval Latency (p95) | **580ms** | Hybrid dense+sparse retrieval |
| Retrieval Faithfulness | **92%** | Measured on internal benchmark |
| Embedding Throughput | 1,200 docs/min | Using all-MiniLM-L6-v2 |
| Vector Search (10K docs) | 12ms | HNSW index, top-5 |

### API Performance

| Endpoint | Latency (p95) | Success Rate |
|----------|---------------|--------------|
| `/health` | 5ms | 99.99% |
| `/predict` | 48ms | 99.8% |
| `/predict/batch` (100) | 180ms | 99.5% |
| `/models` | 8ms | 99.99% |

### Training Benchmarks

| Model | Dataset Size | Training Time | Final Accuracy |
|-------|--------------|---------------|----------------|
| Logistic Regression | 10K samples | 0.8s | 94.2% |
| SVM (RBF kernel) | 10K samples | 12.3s | 91.8% |
| Neural Network | 50K samples | 45s | 96.1% |
| Random Forest (100 trees) | 50K samples | 8.2s | 93.5% |

*Benchmarks run on: Ubuntu 22.04, Python 3.10, AMD Ryzen 7, 32GB RAM*

---

## 🎯 Overview

AI-Mastery-2026 is a production-ready AI engineering toolkit that follows the **White-Box Approach**:

1. **Math First** → Derive equations, understand foundations
2. **Code Second** → Implement from scratch with NumPy
3. **Libraries Third** → Use sklearn/PyTorch knowing what's underneath
4. **Production Always** → Every concept includes deployment considerations

### What's Included

- 📊 **Mathematical Foundations** - Linear algebra, optimization, probability (from scratch)
- 🤖 **Classical ML** - Linear/Logistic Regression, SVM, Decision Trees, Random Forest
- 🧠 **Deep Learning** - Dense, LSTM, Conv2D layers with backpropagation
- 🔤 **LLM Engineering** - Attention, RAG, LoRA fine-tuning, Agents
- 🚀 **Production** - FastAPI, Docker, Prometheus, Grafana
- 📚 **17-Week Learning Program** - Jupyter notebooks for structured learning

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)
- 8GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Or use Make
make install
```

### Train Models

```bash
# Train and save sklearn models for API
python scripts/train_save_models.py
```

### Start Services

```bash
# Option 1: Local API
make run
# API at http://localhost:8000

# Option 2: Docker (all services)
docker-compose up -d
# API:       http://localhost:8000
# Streamlit: http://localhost:8501
# Grafana:   http://localhost:3000
```

### Verify Installation

```bash
# Run tests
make test

# Check API health
curl http://localhost:8000/health
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | **Complete User Guide** - Installation, examples, API reference, troubleshooting |
| [docs/guide/00_index.md](docs/guide/00_index.md) | Guide Index - Table of contents for all documentation |
| [docs/guide/01_getting_started.md](docs/guide/01_getting_started.md) | Getting Started Tutorial |
| [docs/guide/02_core_concepts.md](docs/guide/02_core_concepts.md) | White-Box Approach Philosophy |
| [docs/guide/03_module_core.md](docs/guide/03_module_core.md) | Core Math Module Reference |
| [docs/guide/04_module_ml.md](docs/guide/04_module_ml.md) | ML Algorithms Reference |
| [docs/guide/05_module_llm.md](docs/guide/05_module_llm.md) | LLM Engineering Reference |
| [docs/guide/06_module_production.md](docs/guide/06_module_production.md) | Production Components |
| [docs/guide/07_research_notebooks.md](docs/guide/07_research_notebooks.md) | 17-Week Learning Program |
| [docs/guide/08_contribution_guide.md](docs/guide/08_contribution_guide.md) | Contribution Guidelines |
| [docs/guide/09_deployment_guide.md](docs/guide/09_deployment_guide.md) | Deployment Instructions |
| [docs/guide/10_capstone_project.md](docs/guide/10_capstone_project.md) | **Capstone: GitHub Issue Classifier** |

---

## ✨ Features

### 🔢 Core Mathematics (`src/core/`)

From-scratch implementations of fundamental operations:

```python
from src.core.math_operations import (
    dot_product, matrix_multiply, matrix_inverse,
    sigmoid, relu, softmax, tanh,
    cross_entropy, mse, pca, svd
)

# Example: PCA from scratch
from src.core.math_operations import pca
X_reduced = pca(X, n_components=2)
```

### 🤖 Classical Machine Learning (`src/ml/classical.py`)

| Algorithm | Class | Key Features |
|-----------|-------|--------------|
| Linear Regression | `LinearRegressionScratch` | Closed-form + gradient descent |
| Logistic Regression | `LogisticRegressionScratch` | Binary + multiclass (softmax) |
| SVM | `SVMScratch` | Hinge loss, linear/RBF kernel |
| Decision Tree | `DecisionTreeScratch` | ID3/CART with pruning |
| Random Forest | `RandomForestScratch` | Ensemble with bootstrap |
| K-Nearest Neighbors | `KNNScratch` | Distance-based classification |
| Naive Bayes | `GaussianNBScratch` | Gaussian likelihood |

```python
from src.ml.classical import SVMScratch

svm = SVMScratch(C=1.0, learning_rate=0.001, n_iterations=1000)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
accuracy = svm.score(X_test, y_test)
```

### 🧠 Deep Learning (`src/ml/deep_learning.py`)

Build neural networks from scratch:

```python
from src.ml.deep_learning import (
    NeuralNetwork, Dense, Activation, Dropout,
    BatchNormalization, LSTM, Conv2D, CrossEntropyLoss
)

# Build a CNN
model = NeuralNetwork()
model.add(Conv2D(1, 32, kernel_size=3, padding=1))
model.add(Activation('relu'))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(32*14*14, 128))
model.add(Activation('relu'))
model.add(Dense(128, 10))
model.add(Activation('softmax'))

model.compile(loss=CrossEntropyLoss(), learning_rate=0.01)
history = model.fit(X_train, y_train, epochs=10)
```

**Available Layers:**
- `Dense` - Fully connected layer
- `Activation` - relu, sigmoid, tanh, softmax, leaky_relu
- `Dropout` - Regularization
- `BatchNormalization` - Normalize activations
- `LSTM` - Long Short-Term Memory (all gates)
- `Conv2D` - 2D Convolution with im2col
- `MaxPool2D` - Max pooling
- `Flatten` - Reshape for dense layers

### 🔤 LLM Engineering (`src/llm/`)

**Attention Mechanisms:**
```python
from src.llm.attention import MultiHeadAttention, SelfAttention
attention = MultiHeadAttention(embed_dim=512, num_heads=8)
```

**RAG Pipeline:**
```python
from src.llm.rag import RAGModel, Document, RetrievalStrategy

rag = RAGModel(retriever_strategy=RetrievalStrategy.HYBRID)
rag.add_documents([
    Document(id="1", content="Your content here", metadata={})
])
result = rag.query("What is AI?")
```

**Fine-tuning:**
```python
from src.llm.fine_tuning import LoRAAdapter
adapter = LoRAAdapter(base_model, rank=8, alpha=16)
```

### 🚀 Production Components (`src/production/`)

**FastAPI Service:**
```python
# API endpoints
GET  /health          # Health check
POST /predict         # Single prediction
POST /predict/batch   # Batch predictions
GET  /models          # List models
GET  /models/{id}     # Model info
POST /models/reload   # Hot reload models
GET  /metrics         # Prometheus metrics
```

**Make API Requests:**
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0], "model_name": "logistic_model"}'
```

### 🖥️ Web Interface (`app/main.py`)

Streamlit-based UI with:
- **Home** - Dashboard with stats
- **Chat** - RAG-powered Q&A
- **Predictions** - Interactive ML predictions
- **Models** - View loaded models
- **Settings** - Configuration

```bash
streamlit run app/main.py
# Access at http://localhost:8501
```

---

## 🏗️ Architecture

```
AI-Mastery-2026/
├── src/                          # Source code
│   ├── core/                     # Mathematical foundations
│   │   ├── math_operations.py    # Linear algebra, activations, losses
│   │   ├── optimization.py       # SGD, Adam, regularization
│   │   └── probability.py        # Distributions, sampling
│   │
│   ├── ml/                       # Machine Learning
│   │   ├── classical.py          # LR, SVM, Trees, RF, KNN, NB
│   │   └── deep_learning.py      # Dense, LSTM, Conv2D, NeuralNetwork
│   │
│   ├── llm/                      # LLM Engineering
│   │   ├── attention.py          # Multi-head attention
│   │   ├── rag.py                # Retrieval-Augmented Generation
│   │   ├── fine_tuning.py        # LoRA, QLoRA
│   │   └── agents.py             # LLM agents
│   │
│   └── production/               # Production components
│       ├── api.py                # FastAPI application
│       ├── caching.py            # Model caching
│       ├── monitoring.py         # Prometheus metrics
│       ├── vector_db.py          # HNSW, LSH indices
│       └── deployment.py         # Deployment utilities
│
├── app/                          # Web interface
│   └── main.py                   # Streamlit application
│
├── scripts/                      # Utility scripts
│   ├── train_save_models.py      # Train sklearn models
│   ├── ingest_data.py            # RAG data ingestion
│   └── setup_database.py         # PostgreSQL setup
│
├── config/                       # Configuration
│   ├── prometheus.yml            # Prometheus config
│   └── grafana/                  # Grafana dashboards
│
├── tests/                        # Test suite
│   ├── test_linear_algebra.py
│   ├── test_probability.py
│   ├── test_ml_algorithms.py
│   ├── test_deep_learning.py
│   ├── test_svm.py
│   ├── test_rag_llm.py
│   └── integration/
│
├── research/                     # Jupyter notebooks
│   ├── 00_foundation/            # Week 1-3: Math basics
│   ├── 01_linear_algebra/        # Week 4: Linear algebra
│   ├── ...                       # Weeks 5-16
│   └── mlops_end_to_end.ipynb    # Complete MLOps demo
│
├── docs/                         # Documentation
│   ├── USER_GUIDE.md             # Complete user guide
│   └── guide/                    # Detailed guides
│
├── docker-compose.yml            # All services
├── Dockerfile                    # API container
├── Dockerfile.streamlit          # Streamlit container
├── requirements.txt              # Python dependencies
├── Makefile                      # Build automation
└── README.md                     # This file
```

---

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI ML API |
| `streamlit` | 8501 | Web interface |
| `postgres` | 5432 | Database |
| `redis` | 6379 | Cache |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Dashboards (admin/admin) |

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## 📊 Monitoring

### Grafana Dashboard

Access at http://localhost:3000 (admin/admin)

**Metrics available:**
- Models loaded count
- API status (up/down)
- Response time percentiles (p50, p95, p99)
- Request rate per endpoint
- Error rate (4xx, 5xx)

### Prometheus Queries

```promql
# Average response time
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_svm.py -v

# Specific test
pytest tests/test_svm.py::TestSVMScratch::test_accuracy -v
```

**Test Coverage:**
- Core math operations
- Classical ML algorithms
- Deep learning layers
- RAG pipeline
- API endpoints

---

## 🛠️ Development

### Available Make Commands

```bash
make install      # Install dependencies
make test         # Run tests
make test-cov     # Tests with coverage
make lint         # Run linters
make format       # Format code
make run          # Start API locally
make docker-run   # Start with Docker
make docker-stop  # Stop Docker services
make docs         # Generate documentation
```

### Code Style

- Python 3.10+ compatible
- Type hints for all functions
- 100 character line limit
- Black + isort formatting
- MyPy type checking

### Documentation Standards

```python
def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    
    σ(x) = 1 / (1 + e^(-x))
    
    Args:
        x: Input array
    
    Returns:
        Sigmoid of input
    
    Example:
        >>> sigmoid(np.array([0, 1, -1]))
        array([0.5, 0.731, 0.269])
    """
    return 1 / (1 + np.exp(-x))
```

---

## 📚 Learning Path

The project includes a **17-week structured program** in `research/`:

| Week | Topic | Notebook |
|------|-------|----------|
| 1-3 | Mathematical Foundations | `00_foundation/` |
| 4 | Linear Algebra | `01_linear_algebra/` |
| 5 | Probability & Statistics | `02_probability/` |
| 6 | Optimization | `03_optimization/` |
| 7 | Classical ML | `04_classical_ml/` |
| 8-9 | Neural Networks | `05_neural_networks/` |
| 10-11 | Deep Learning | `06_deep_learning/` |
| 12 | Transformers | `07_transformers/` |
| 13-14 | LLM Engineering | `08_llm/` |
| 15 | RAG Systems | `09_rag/` |
| 16 | Production | `10_production/` |
| 17 | Capstone | `mlops_end_to_end.ipynb` |

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

See [docs/guide/08_contribution_guide.md](docs/guide/08_contribution_guide.md) for details.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NumPy, Pandas, Scikit-learn teams
- PyTorch and Transformers communities
- FastAPI and Streamlit projects
- All contributors and educators

---

<div align="center">

**Built with ❤️ for learning AI engineering from first principles**

[⬆ Back to Top](#ai-mastery-2026-full-stack-ai-engineering-toolkit)

</div>