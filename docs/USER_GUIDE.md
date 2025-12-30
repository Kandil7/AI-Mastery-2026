# AI-Mastery-2026: Complete User Guide

This comprehensive guide covers everything you need to know to use the AI-Mastery-2026 toolkit effectively.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation Guide](#2-installation-guide)
3. [Project Structure](#3-project-structure)
4. [Core Modules](#4-core-modules)
5. [Classical Machine Learning](#5-classical-machine-learning)
6. [Deep Learning](#6-deep-learning)
7. [LLM Engineering](#7-llm-engineering)
8. [Production API](#8-production-api)
9. [Web Interface](#9-web-interface)
10. [Docker Deployment](#10-docker-deployment)
11. [Monitoring & Observability](#11-monitoring--observability)
12. [Testing](#12-testing)
13. [Troubleshooting](#13-troubleshooting)
14. [FAQ](#14-faq)

---

## 1. Introduction

### What is AI-Mastery-2026?

AI-Mastery-2026 is a **full-stack AI engineering toolkit** designed for learning and building production AI applications. It follows the "White-Box Approach":

1. **Math First** - Understand the mathematical foundations
2. **Code Second** - Implement algorithms from scratch using NumPy
3. **Libraries Third** - Use frameworks knowing what's happening underneath
4. **Production Always** - Every concept includes real-world deployment

### Who Is This For?

- **Students** learning ML/AI fundamentals
- **Engineers** transitioning into AI roles
- **Researchers** needing reference implementations
- **Teams** building production ML systems

### What You'll Learn

- Linear algebra, calculus, optimization
- Classical ML algorithms (from scratch)
- Neural networks and deep learning
- Transformer architecture and attention
- RAG systems and LLM engineering
- Production deployment with Docker
- Monitoring with Prometheus/Grafana

---

## 2. Installation Guide

### 2.1 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Required |
| pip | Latest | Package manager |
| Docker | 20.10+ | Optional, for deployment |
| Docker Compose | 2.0+ | Optional |
| RAM | 8GB+ | Recommended |
| GPU | CUDA 11+ | Optional, for faster training |

### 2.2 Standard Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026

# Step 2: Create virtual environment
python -m venv .venv

# Step 3: Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Step 4: Install dependencies
pip install -r requirements.txt

# Step 5: Verify installation
python -c "from src.ml.classical import SVMScratch; print('✅ Installation successful!')"
```

### 2.3 Using Make

```bash
# Install everything with one command
make install

# Or step by step:
make venv           # Create virtual environment
make install-deps   # Install dependencies
make install-dev    # Install dev dependencies
```

### 2.4 Docker Installation

```bash
# Build all images
docker-compose build

# Or use make
make docker-build
```

### 2.5 Verify Installation

```bash
# Run test suite
make test

# Expected output:
# ========================= test session starts =========================
# collected XX items
# tests/test_linear_algebra.py ....                               [  5%]
# tests/test_probability.py ....                                  [ 10%]
# ...
# ========================= XX passed in X.XXs =========================
```

---

## 3. Project Structure

```
AI-Mastery-2026/
│
├── src/                              # Core source code
│   ├── __init__.py
│   │
│   ├── core/                         # Mathematical foundations
│   │   ├── __init__.py
│   │   ├── math_operations.py        # Vector ops, matrix ops, activations
│   │   ├── optimization.py           # SGD, Adam, gradient methods
│   │   └── probability.py            # Distributions, sampling, info theory
│   │
│   ├── ml/                           # Machine Learning
│   │   ├── __init__.py
│   │   ├── classical.py              # LR, SVM, Trees, RF, KNN, NB
│   │   └── deep_learning.py          # Dense, LSTM, Conv2D, Networks
│   │
│   ├── llm/                          # LLM Engineering
│   │   ├── __init__.py
│   │   ├── attention.py              # Scaled dot-product, multi-head
│   │   ├── rag.py                    # RAG pipeline components
│   │   ├── fine_tuning.py            # LoRA adapters
│   │   └── agents.py                 # LLM agents
│   │
│   └── production/                   # Production components
│       ├── __init__.py
│       ├── api.py                    # FastAPI application
│       ├── caching.py                # Model caching
│       ├── monitoring.py             # Prometheus metrics
│       └── vector_db.py              # HNSW, LSH indices
│
├── app/                              # Web interface
│   └── main.py                       # Streamlit application
│
├── scripts/                          # Utility scripts
│   ├── train_save_models.py          # Train sklearn models for API
│   ├── ingest_data.py                # RAG data ingestion pipeline
│   └── setup_database.py             # PostgreSQL schema setup
│
├── config/                           # Configuration files
│   ├── prometheus.yml                # Prometheus scrape config
│   └── grafana/
│       ├── provisioning/
│       │   └── datasources/
│       └── dashboards/
│           └── ml_api.json           # Pre-built dashboard
│
├── tests/                            # Test suite
│   ├── test_linear_algebra.py
│   ├── test_probability.py
│   ├── test_ml_algorithms.py
│   ├── test_deep_learning.py
│   ├── test_svm.py
│   ├── test_rag_llm.py
│   └── integration/
│
├── research/                         # Jupyter notebooks
│   ├── 00_foundation/                # Weeks 1-3
│   ├── 01_linear_algebra/            # Week 4
│   ├── ...                           # Weeks 5-16
│   └── mlops_end_to_end.ipynb        # Complete MLOps demo
│
├── models/                           # Saved models directory
│   ├── classification_model.joblib
│   ├── regression_model.joblib
│   ├── logistic_model.joblib
│   └── models_metadata.json
│
├── docs/                             # Documentation
│   ├── USER_GUIDE.md                 # This file
│   └── guide/                        # Detailed guides
│
├── docker-compose.yml                # Docker services config
├── Dockerfile                        # API container
├── Dockerfile.streamlit              # Web UI container
├── requirements.txt                  # Python dependencies
├── Makefile                          # Build automation
├── setup.sh                          # Setup script
└── README.md                         # Project overview
```

---

## 4. Core Modules

### 4.1 Math Operations (`src/core/math_operations.py`)

**Vector Operations:**

```python
from src.core.math_operations import dot_product, cosine_similarity

# Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = dot_product(a, b)  # 32

# Cosine similarity
sim = cosine_similarity(a, b)  # 0.974
```

**Matrix Operations:**

```python
from src.core.math_operations import matrix_multiply, matrix_inverse, matrix_transpose

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Multiply
C = matrix_multiply(A, B)

# Inverse (if exists)
A_inv = matrix_inverse(A)

# Transpose
A_T = matrix_transpose(A)
```

**Activation Functions:**

```python
from src.core.math_operations import sigmoid, relu, softmax, tanh

# Sigmoid: σ(x) = 1 / (1 + e^(-x))
sigmoid(np.array([-1, 0, 1]))  # [0.269, 0.5, 0.731]

# ReLU: max(0, x)
relu(np.array([-1, 0, 1]))  # [0, 0, 1]

# Softmax: e^xi / Σe^xj
softmax(np.array([1, 2, 3]))  # [0.09, 0.24, 0.67]

# Tanh: (e^x - e^-x) / (e^x + e^-x)
tanh(np.array([-1, 0, 1]))  # [-0.76, 0, 0.76]
```

**Dimensionality Reduction:**

```python
from src.core.math_operations import pca, svd

# PCA
X_reduced = pca(X, n_components=2)

# SVD: A = UΣV^T
U, S, Vt = svd(A)
```

### 4.2 Optimization (`src/core/optimization.py`)

**Gradient Descent:**

```python
from src.core.optimization import SGD, Adam

# Stochastic Gradient Descent
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# Adam optimizer
optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

# Update weights
weights = optimizer.update(weights, gradients)
```

**Regularization:**

```python
from src.core.optimization import l1_regularization, l2_regularization

# L1 (Lasso): λ Σ|w|
l1_penalty = l1_regularization(weights, lambda_=0.01)

# L2 (Ridge): λ Σw²
l2_penalty = l2_regularization(weights, lambda_=0.01)
```

### 4.3 Probability (`src/core/probability.py`)

**Distributions:**

```python
from src.core.probability import gaussian_pdf, bernoulli, multinomial

# Gaussian PDF
p = gaussian_pdf(x, mean=0, std=1)

# Sample from distributions
samples = bernoulli(p=0.5, size=100)
```

**Information Theory:**

```python
from src.core.probability import entropy, kl_divergence, mutual_information

# Shannon entropy: H(X) = -Σ p(x) log p(x)
H = entropy(probabilities)

# KL Divergence: KL(P||Q) = Σ P(x) log(P(x)/Q(x))
kl = kl_divergence(P, Q)
```

---

## 5. Classical Machine Learning

### 5.1 Linear Regression

```python
from src.ml.classical import LinearRegressionScratch

# Create model
model = LinearRegressionScratch(method='gradient_descent', learning_rate=0.01)

# Fit
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Coefficients
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias}")

# R² Score
score = model.score(X_test, y_test)
```

### 5.2 Logistic Regression

```python
from src.ml.classical import LogisticRegressionScratch

# Binary classification
model = LogisticRegressionScratch(
    learning_rate=0.01,
    n_iterations=1000,
    regularization='l2',
    lambda_=0.1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### 5.3 Support Vector Machine

```python
from src.ml.classical import SVMScratch

# Create SVM
svm = SVMScratch(
    C=1.0,                # Regularization (higher = less regularization)
    learning_rate=0.001,   # Step size
    n_iterations=1000,     # Training iterations
    kernel='linear'        # 'linear' or 'rbf'
)

# Train
svm.fit(X_train, y_train)

# Predict
predictions = svm.predict(X_test)

# Decision function (distance from hyperplane)
decision_values = svm.decision_function(X_test)

# Probability estimates (Platt scaling approximation)
probabilities = svm.predict_proba(X_test)

# Get support vectors
print(f"Support vectors: {len(svm.support_vectors_)}")
print(f"Training loss: {svm.loss_history[-1]:.4f}")
```

### 5.4 Decision Tree

```python
from src.ml.classical import DecisionTreeScratch

# Create tree
tree = DecisionTreeScratch(
    max_depth=5,
    min_samples_split=10,
    criterion='gini'  # or 'entropy'
)

tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

# Feature importance
importances = tree.feature_importances_
```

### 5.5 Random Forest

```python
from src.ml.classical import RandomForestScratch

# Create forest
forest = RandomForestScratch(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    bootstrap=True
)

forest.fit(X_train, y_train)
predictions = forest.predict(X_test)
probabilities = forest.predict_proba(X_test)
```

### 5.6 K-Nearest Neighbors

```python
from src.ml.classical import KNNScratch

knn = KNNScratch(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

### 5.7 Naive Bayes

```python
from src.ml.classical import GaussianNBScratch

nb = GaussianNBScratch()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
log_probs = nb.predict_log_proba(X_test)
```

---

## 6. Deep Learning

### 6.1 Basic Neural Network

```python
from src.ml.deep_learning import (
    NeuralNetwork, Dense, Activation, Dropout,
    BatchNormalization, CrossEntropyLoss
)

# Build model
model = NeuralNetwork()
model.add(Dense(input_size=784, output_size=256, weight_init='he'))
model.add(BatchNormalization(256))
model.add(Activation('relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(256, 128, weight_init='he'))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(128, 10, weight_init='xavier'))
model.add(Activation('softmax'))

# Compile
model.compile(
    loss=CrossEntropyLoss(),
    learning_rate=0.01
)

# Summary
model.summary()
# Output:
# ============================================================
# Model Summary
# ============================================================
# Layer 1: Dense                  | Params: 200,960
# Layer 2: BatchNormalization     | Params: 512
# Layer 3: Activation             | Params: 0
# ...
# ============================================================
# Total Parameters: 235,146
# ============================================================

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=True
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### 6.2 LSTM for Sequences

```python
from src.ml.deep_learning import LSTM, Dense, Activation, NeuralNetwork

# For sequence data: (batch_size, timesteps, features)
model = NeuralNetwork()
model.add(LSTM(input_size=10, hidden_size=64, return_sequences=False))
model.add(Dense(64, 32))
model.add(Activation('relu'))
model.add(Dense(32, 2))
model.add(Activation('softmax'))

# Input shape: (batch, timesteps, features)
X_seq = np.random.randn(100, 20, 10)  # 100 samples, 20 steps, 10 features
y_seq = np.random.randint(0, 2, 100)

model.compile(loss=CrossEntropyLoss(), learning_rate=0.01)
model.fit(X_seq, y_seq, epochs=10)
```

### 6.3 Convolutional Neural Network

```python
from src.ml.deep_learning import (
    Conv2D, MaxPool2D, Flatten, Dense, Activation, NeuralNetwork
)

# Build CNN for image classification
model = NeuralNetwork()

# Conv block 1
model.add(Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2))

# Conv block 2
model.add(Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2))

# Classifier
model.add(Flatten())
model.add(Dense(64 * 7 * 7, 256))
model.add(Activation('relu'))
model.add(Dense(256, 10))
model.add(Activation('softmax'))

# Input shape: (batch, channels, height, width)
X_images = np.random.randn(100, 1, 28, 28)  # MNIST-like
y_labels = np.random.randint(0, 10, 100)

model.compile(loss=CrossEntropyLoss(), learning_rate=0.001)
model.fit(X_images, y_labels, epochs=5, batch_size=16)
```

### 6.4 Available Layers

| Layer | Description | Parameters |
|-------|-------------|------------|
| `Dense` | Fully connected | input_size, output_size, weight_init |
| `Activation` | Activation function | 'relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu' |
| `Dropout` | Regularization | rate (0-1) |
| `BatchNormalization` | Normalize activations | n_features, momentum, epsilon |
| `LSTM` | Recurrent layer | input_size, hidden_size, return_sequences |
| `Conv2D` | 2D Convolution | in_channels, out_channels, kernel_size, stride, padding |
| `MaxPool2D` | Max pooling | pool_size, stride |
| `Flatten` | Reshape to 1D | - |

### 6.5 Loss Functions

| Loss | Class | Use Case |
|------|-------|----------|
| MSE | `MSELoss` | Regression |
| Cross-Entropy | `CrossEntropyLoss` | Multi-class classification |
| Binary Cross-Entropy | `BinaryCrossEntropyLoss` | Binary classification |

---

## 7. LLM Engineering

### 7.1 Attention Mechanisms

```python
from src.llm.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention
)

# Scaled dot-product attention
# Attention(Q, K, V) = softmax(QK^T / √d_k)V
attention = ScaledDotProductAttention()
output, weights = attention(query, key, value)

# Multi-head attention with 8 heads
mha = MultiHeadAttention(embed_dim=512, num_heads=8)
output = mha(query, key, value)
```

### 7.2 RAG Pipeline

```python
from src.llm.rag import RAGModel, Document, RetrievalStrategy

# Initialize RAG with different strategies
# Options: DENSE, SPARSE, HYBRID
rag = RAGModel(retriever_strategy=RetrievalStrategy.HYBRID)

# Add documents
documents = [
    Document(
        id="doc1",
        content="Machine learning uses algorithms to learn from data.",
        metadata={"source": "ml_intro.txt", "topic": "ML"}
    ),
    Document(
        id="doc2",
        content="Neural networks are inspired by biological neurons.",
        metadata={"source": "dl_intro.txt", "topic": "DL"}
    ),
]
rag.add_documents(documents)

# Query
result = rag.query(
    "What is machine learning?",
    k=3  # Number of documents to retrieve
)

print(f"Response: {result['response']}")
print(f"Retrieved documents: {len(result['documents'])}")
for doc in result['documents']:
    print(f"  - {doc.id}: {doc.content[:50]}...")
```

### 7.3 Data Ingestion

```python
from scripts.ingest_data import DataIngestionPipeline

# Initialize pipeline
pipeline = DataIngestionPipeline(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=512,
    chunk_overlap=50
)

# Ingest documents from directory
num_docs = pipeline.ingest_directory(
    directory="./data/documents",
    extensions=[".txt", ".md", ".pdf"],
    recursive=True
)

# Process (chunk and embed)
num_chunks = pipeline.process_documents()

# Save index
pipeline.save("./data/rag_index")

# Search
results = pipeline.search("How do neural networks work?", k=5)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:100]}...")
```

### 7.4 Fine-tuning with LoRA

```python
from src.llm.fine_tuning import LoRAAdapter

# Apply LoRA to a base model
adapter = LoRAAdapter(
    base_model=model,
    rank=8,          # Rank of low-rank matrices
    alpha=16,        # Scaling factor
    target_modules=['query', 'value']  # Layers to adapt
)

# Train only LoRA parameters (much smaller)
adapter.train(train_data, epochs=3)

# Save adapter
adapter.save("./adapters/my_lora")
```

---

## 8. Production API

### 8.1 Starting the API

```bash
# Option 1: Direct
uvicorn src.production.api:app --host 0.0.0.0 --port 8000 --reload

# Option 2: Make command
make run

# Option 3: Docker
docker-compose up -d api
```

### 8.2 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check with model count |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/models` | List all loaded models |
| GET | `/models/{model_id}` | Get specific model info |
| POST | `/models/reload` | Hot reload models |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | OpenAPI documentation |

### 8.3 Example Requests

**Health Check:**
```bash
curl http://localhost:8000/health
```
```json
{
    "status": "healthy",
    "models_loaded": 3,
    "timestamp": 1704067200.0
}
```

**List Models:**
```bash
curl http://localhost:8000/models
```
```json
{
    "models": [
        {
            "model_id": "classification_model",
            "model_type": "RandomForestClassifier",
            "n_features": 10,
            "n_classes": 2,
            "metadata": {"accuracy": 0.92}
        },
        {
            "model_id": "regression_model",
            "model_type": "GradientBoostingRegressor",
            "n_features": 10
        }
    ]
}
```

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "model_name": "classification_model"
    }'
```
```json
{
    "predictions": [1],
    "probabilities": [[0.15, 0.85]],
    "model_type": "RandomForestClassifier",
    "processing_time": 0.002
}
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:8000/predict/batch \
    -H "Content-Type: application/json" \
    -d '{
        "features": [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        ],
        "model_id": "classification_model"
    }'
```

### 8.4 Python Client

```python
import requests

API_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{API_URL}/health")
print(response.json())

# Prediction
features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
response = requests.post(
    f"{API_URL}/predict",
    json={"features": features, "model_name": "classification_model"}
)
result = response.json()
print(f"Prediction: {result['predictions'][0]}")
print(f"Confidence: {max(result['probabilities'][0]):.2%}")
```

---

## 9. Web Interface

### 9.1 Starting Streamlit

```bash
# Direct
streamlit run app/main.py

# Docker
docker-compose up -d streamlit

# Access at http://localhost:8501
```

### 9.2 Pages

**Home Dashboard:**
- Models loaded count
- API status indicator
- Response time metrics
- Uptime statistics

**Chat (RAG):**
- Interactive Q&A interface
- Document retrieval visualization
- Source citations
- Conversation history

**Predictions:**
- Model selection dropdown
- Feature input (manual/JSON/random)
- Real-time prediction results
- Probability visualization

**Models:**
- List all registered models
- Model metadata display
- Feature count, type, metrics

**Settings:**
- API URL configuration
- Theme selection
- RAG parameters

---

## 10. Docker Deployment

### 10.1 Services

```yaml
# docker-compose.yml services:

api:          # FastAPI server (port 8000)
streamlit:    # Web UI (port 8501)
postgres:     # Database (port 5432)
redis:        # Cache (port 6379)
prometheus:   # Metrics (port 9090)
grafana:      # Dashboards (port 3000)
```

### 10.2 Commands

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d api

# View logs
docker-compose logs -f api

# Stop all
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Remove volumes (data reset)
docker-compose down -v
```

### 10.3 Environment Variables

```bash
# Database
DB_HOST=postgres
DB_PORT=5432
DB_NAME=ai_mastery
DB_USER=postgres
DB_PASSWORD=password

# API
API_URL=http://api:8000
PROMETHEUS_URL=http://prometheus:9090

# Optional
LOG_LEVEL=INFO
WORKERS=4
```

---

## 11. Monitoring & Observability

### 11.1 Prometheus Metrics

Built-in metrics at `/metrics`:

```
# Request counts
http_requests_total{method="POST", endpoint="/predict", status="200"}

# Latency histogram
http_request_duration_seconds_bucket{le="0.1"}
http_request_duration_seconds_sum
http_request_duration_seconds_count

# Custom ML metrics
ml_models_loaded_total
ml_predictions_total{model="classification_model"}
```

### 11.2 Grafana Dashboard

Access at http://localhost:3000 (admin/admin)

**Pre-built panels:**
- Models Loaded (gauge)
- API Status (status indicator)
- Response Time (time series)
- Request Rate (graph)
- Error Rate (percentage)

### 11.3 Example Prometheus Queries

```promql
# Average response time (last 5 min)
rate(http_request_duration_seconds_sum[5m]) 
  / rate(http_request_duration_seconds_count[5m])

# 95th percentile latency
histogram_quantile(0.95, 
  rate(http_request_duration_seconds_bucket[5m])
)

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) 
  / sum(rate(http_requests_total[5m])) * 100

# Request rate by endpoint
sum by (endpoint) (rate(http_requests_total[5m]))
```

---

## 12. Testing

### 12.1 Running Tests

```bash
# All tests
make test

# With coverage report
make test-cov
# Opens htmlcov/index.html

# Specific file
pytest tests/test_svm.py -v

# Specific test
pytest tests/test_svm.py::TestSVMScratch::test_accuracy_linearly_separable -v

# With print output
pytest tests/test_svm.py -v -s
```

### 12.2 Test Structure

```python
# tests/test_example.py
import pytest
import numpy as np
from src.ml.classical import SVMScratch

class TestSVMScratch:
    
    @pytest.fixture
    def sample_data(self):
        """Generate test data."""
        X = np.random.randn(100, 5)
        y = (X.sum(axis=1) > 0).astype(int)
        return X, y
    
    def test_fit_predict(self, sample_data):
        X, y = sample_data
        svm = SVMScratch(n_iterations=100)
        svm.fit(X, y)
        predictions = svm.predict(X)
        assert predictions.shape == y.shape
    
    def test_accuracy(self, sample_data):
        X, y = sample_data
        svm = SVMScratch()
        svm.fit(X, y)
        accuracy = svm.score(X, y)
        assert accuracy > 0.5  # Better than random
```

### 12.3 Test Categories

| File | Tests |
|------|-------|
| `test_linear_algebra.py` | Vector/matrix operations |
| `test_probability.py` | Distributions, entropy |
| `test_ml_algorithms.py` | Classical ML models |
| `test_deep_learning.py` | Neural network layers |
| `test_svm.py` | SVM + LSTM + Conv2D |
| `test_rag_llm.py` | RAG pipeline |
| `integration/test_api.py` | API endpoints |

---

## 13. Troubleshooting

### Common Issues

**1. ImportError: No module named 'src'**
```bash
# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or in Windows PowerShell
$env:PYTHONPATH = "$env:PYTHONPATH;$(pwd)"
```

**2. Models not found in API**
```bash
# Train and save models first
python scripts/train_save_models.py

# Check models directory
ls models/
# Should see: classification_model.joblib, etc.
```

**3. Port already in use**
```bash
# Kill process on port 8000
kill $(lsof -t -i:8000)

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**4. Docker memory issues**
```bash
# Increase Docker memory (4GB+ recommended)
# Docker Desktop > Settings > Resources > Memory

# Or in docker-compose.yml:
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
```

**5. Slow training**
```bash
# Use GPU if available
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**6. Database connection failed**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# View logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

---

## 14. FAQ

**Q: Can I use this for production?**
A: Yes! The toolkit includes production-ready components (FastAPI, Docker, monitoring). However, review security settings before deploying to public environments.

**Q: How do I add a new ML model?**
A: Add to `src/ml/classical.py` or `deep_learning.py`, then:
1. Inherit from `BaseEstimator` or `Layer`
2. Implement `fit()`, `predict()`, `forward()`, `backward()`
3. Add tests in `tests/`
4. Export in `__all__`

**Q: Can I use GPU?**
A: The from-scratch implementations use NumPy (CPU). For GPU, leverage PyTorch/TensorFlow versions in production code.

**Q: How do I contribute?**
A: See [docs/guide/08_contribution_guide.md](docs/guide/08_contribution_guide.md). Fork, create branch, submit PR.

**Q: Where are the Jupyter notebooks?**
A: In `research/` directory, organized by week (1-17).

---

## Need Help?

1. 📖 Check the [Documentation](docs/guide/)
2. 🧪 Run tests: `make test`
3. 📋 Check [GitHub Issues](https://github.com/Kandil7/AI-Mastery-2026/issues)
4. 📬 Open a new issue with:
   - Python version
   - OS
   - Error message
   - Steps to reproduce

---

*Built with ❤️ for learning AI engineering from first principles*
