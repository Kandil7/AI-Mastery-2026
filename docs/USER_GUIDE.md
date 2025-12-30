# AI-Mastery-2026: Complete User Guide

This comprehensive guide covers everything you need to know to use the AI-Mastery-2026 toolkit effectively.

---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Quick Start Examples](#quick-start-examples)
4. [Core Modules](#core-modules)
5. [API Reference](#api-reference)
6. [Web Interface](#web-interface)
7. [Docker Deployment](#docker-deployment)
8. [Monitoring](#monitoring)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.10+
- pip or conda
- Docker (optional)
- 8GB+ RAM recommended
- GPU (optional, for faster LLM inference)

### Standard Installation

```bash
# Clone repository
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or use make
make install
```

### Docker Installation

```bash
# Build images
make docker-build

# Start all services
make docker-run

# Check logs
make docker-logs
```

### Verify Installation

```bash
# Run tests
make test

# Train sample models
python scripts/train_save_models.py

# Start API
make run
# API available at http://localhost:8000
```

---

## Project Structure

```
AI-Mastery-2026/
├── src/                          # Source code
│   ├── core/                     # Mathematical foundations
│   │   ├── math_operations.py    # Linear algebra, activation functions
│   │   ├── optimization.py       # SGD, Adam, gradients
│   │   └── probability.py        # Distributions, sampling
│   ├── ml/                       # Machine learning
│   │   ├── classical.py          # LinearRegression, LogisticRegression, SVM, etc.
│   │   └── deep_learning.py      # Dense, LSTM, Conv2D, NeuralNetwork
│   ├── llm/                      # LLM engineering
│   │   ├── attention.py          # Multi-head attention
│   │   ├── rag.py                # RAG pipeline
│   │   ├── fine_tuning.py        # LoRA, QLoRA
│   │   └── agents.py             # LLM agents
│   └── production/               # Production components
│       ├── api.py                # FastAPI application
│       ├── caching.py            # Model caching
│       ├── monitoring.py         # Prometheus metrics
│       └── vector_db.py          # HNSW, LSH
├── app/                          # Web interface
│   └── main.py                   # Streamlit app
├── scripts/                      # Utility scripts
│   ├── train_save_models.py      # Train sklearn models
│   ├── ingest_data.py            # RAG data ingestion
│   └── setup_database.py         # PostgreSQL setup
├── config/                       # Configuration files
│   ├── prometheus.yml            # Prometheus config
│   └── grafana/                  # Grafana dashboards
├── tests/                        # Test suite
├── research/                     # Jupyter notebooks (17-week program)
└── docs/                         # Documentation
```

---

## Quick Start Examples

### Example 1: Train a Neural Network from Scratch

```python
from src.ml.deep_learning import (
    NeuralNetwork, Dense, Activation, Dropout, CrossEntropyLoss
)
import numpy as np

# Create sample data
X = np.random.randn(1000, 10)
y = (X.sum(axis=1) > 0).astype(int)

# Build model
model = NeuralNetwork()
model.add(Dense(10, 32, weight_init='he'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32, 2))
model.add(Activation('softmax'))

# Compile and train
model.compile(loss=CrossEntropyLoss(), learning_rate=0.01)
history = model.fit(X, y, epochs=50, batch_size=32, verbose=True)

# Evaluate
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2%}")
```

### Example 2: Use SVM Classifier

```python
from src.ml.classical import SVMScratch
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# Train SVM
svm = SVMScratch(C=1.0, learning_rate=0.001, n_iterations=1000)
svm.fit(X, y)

# Predict
predictions = svm.predict(X)
accuracy = svm.score(X, y)
print(f"SVM Accuracy: {accuracy:.2%}")
```

### Example 3: Use RAG System

```python
from src.llm.rag import RAGModel, Document, RetrievalStrategy

# Initialize RAG
rag = RAGModel(retriever_strategy=RetrievalStrategy.HYBRID)

# Add documents
documents = [
    Document(id="1", content="Python is a programming language.", metadata={}),
    Document(id="2", content="Machine learning uses algorithms.", metadata={}),
    Document(id="3", content="Neural networks have layers.", metadata={}),
]
rag.add_documents(documents)

# Query
result = rag.query("What is Python?")
print(f"Response: {result['response']}")
print(f"Sources: {[doc.id for doc in result['documents']]}")
```

### Example 4: Make API Predictions

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [1.0, 2.0, 3.0, 4.0, 5.0], "model_name": "logistic_model"}
)
print(response.json())
```

---

## Core Modules

### 1. Core (`src/core`)

**Math Operations** (`math_operations.py`)
- `dot_product()`, `matrix_multiply()`, `matrix_inverse()`
- `sigmoid()`, `relu()`, `softmax()`, `tanh()`
- `cross_entropy()`, `mse()`, `log_loss()`
- `pca()`, `svd()`, `eigen_decomposition()`

**Optimization** (`optimization.py`)
- `SGD`, `Adam`, `RMSprop` optimizers
- `gradient_descent()`, `newton_raphson()`
- Regularization: L1, L2, Elastic Net

**Probability** (`probability.py`)
- `gaussian_pdf()`, `bernoulli()`, `multinomial()`
- `kl_divergence()`, `entropy()`, `mutual_information()`

### 2. Machine Learning (`src/ml`)

**Classical ML** (`classical.py`)
- `LinearRegressionScratch` - closed form + gradient descent
- `LogisticRegressionScratch` - binary + multiclass
- `SVMScratch` - hinge loss, linear/RBF kernel
- `DecisionTreeScratch` - ID3/CART algorithms
- `RandomForestScratch` - ensemble of trees
- `KNNScratch` - k-nearest neighbors
- `GaussianNBScratch` - naive bayes

**Deep Learning** (`deep_learning.py`)
- Layers: `Dense`, `Activation`, `Dropout`, `BatchNormalization`
- Advanced: `LSTM`, `Conv2D`, `MaxPool2D`, `Flatten`
- Losses: `MSELoss`, `CrossEntropyLoss`, `BinaryCrossEntropyLoss`
- Model: `NeuralNetwork` - sequential model with auto backprop

### 3. LLM Engineering (`src/llm`)

**Attention** (`attention.py`)
- `ScaledDotProductAttention`
- `MultiHeadAttention`
- `SelfAttention`, `CrossAttention`

**RAG** (`rag.py`)
- `DenseRetriever` - sentence-transformers + FAISS
- `SparseRetriever` - TF-IDF
- `HybridRetriever` - combined dense + sparse
- `RAGModel` - full retrieval + generation

**Fine-tuning** (`fine_tuning.py`)
- `LoRAAdapter` - low-rank adaptation
- `QLoRATrainer` - quantized training

### 4. Production (`src/production`)

**API** (`api.py`)
- `ModelCache` - efficient model loading
- `/predict` - single prediction
- `/predict/batch` - batch predictions
- `/models` - list available models
- `/metrics` - Prometheus metrics

**Monitoring** (`monitoring.py`)
- Request counting
- Latency histograms
- Error tracking

**Vector DB** (`vector_db.py`)
- `HNSWIndex` - approximate nearest neighbors
- `LSHIndex` - locality-sensitive hashing

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/models` | List all models |
| GET | `/models/{id}` | Get model info |
| POST | `/models/reload` | Reload models |
| GET | `/metrics` | Prometheus metrics |

### Request/Response Examples

**Health Check**
```bash
curl http://localhost:8000/health
```
```json
{"status": "healthy", "models_loaded": 3, "timestamp": 1704067200.0}
```

**Prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0], "model_name": "logistic_model"}'
```
```json
{
  "predictions": [1],
  "probabilities": [[0.15, 0.85]],
  "model_type": "LogisticRegression",
  "processing_time": 0.002
}
```

---

## Web Interface

### Streamlit App

Start the web interface:

```bash
streamlit run app/main.py
# Available at http://localhost:8501
```

### Pages

1. **Home** - Dashboard with stats
2. **Chat (RAG)** - Interactive Q&A with document retrieval
3. **Predictions** - Make ML predictions with UI
4. **Models** - View loaded models and metadata
5. **Settings** - Configure API URL and preferences

---

## Docker Deployment

### Services

| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI server |
| streamlit | 8501 | Web UI |
| postgres | 5432 | Database |
| redis | 6379 | Cache |
| prometheus | 9090 | Metrics |
| grafana | 3000 | Dashboards |

### Commands

```bash
# Start all
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop all
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

---

## Monitoring

### Grafana Dashboard

Access at http://localhost:3000 (admin/admin)

**Available Metrics:**
- Models loaded
- API status
- Response time (p50, p95, p99)
- Request rate
- Error rate

### Prometheus Queries

```promql
# Request rate (5m average)
rate(http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
```

---

## Testing

### Run Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_svm.py -v

# Specific test
pytest tests/test_svm.py::TestSVMScratch::test_accuracy_linearly_separable -v
```

### Test Categories

- `tests/test_linear_algebra.py` - Core math operations
- `tests/test_probability.py` - Probability functions
- `tests/test_ml_algorithms.py` - Classical ML
- `tests/test_deep_learning.py` - Neural networks
- `tests/test_svm.py` - SVM + advanced layers
- `tests/test_rag_llm.py` - RAG pipeline
- `tests/test_caching.py` - Caching layer
- `tests/integration/` - API integration tests

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**2. Model Not Found**
```bash
# Train and save models first
python scripts/train_save_models.py
```

**3. Port Already in Use**
```bash
# Kill process on port 8000
kill $(lsof -t -i:8000)
```

**4. Docker Memory Issues**
```bash
# Increase Docker memory limit to 4GB+
# Docker Desktop > Settings > Resources
```

**5. GPU Not Detected**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Getting Help

1. Check [docs/guide/](docs/guide/) for detailed documentation
2. Run `make test` to verify installation
3. Check GitHub Issues for known problems
4. Review logs: `docker-compose logs -f`

---

## Next Steps

1. **Learn Fundamentals**: Work through `research/` notebooks
2. **Build Capstone**: Follow [10_capstone_project.md](docs/guide/10_capstone_project.md)
3. **Deploy Production**: Use Docker Compose for full stack
4. **Contribute**: See [08_contribution_guide.md](docs/guide/08_contribution_guide.md)

---

*Generated by AI-Mastery-2026 | Full-Stack AI Engineering Toolkit*
