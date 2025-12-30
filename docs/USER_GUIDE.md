# AI Engineer Toolkit: User Guide

This guide provides a step-by-step walkthrough of using the AI Engineer Toolkit for various tasks, ranging from basic mathematical operations to deploying a full-stack RAG application.

## Table of Contents
1. [Installation & Setup](#1-installation--setup)
2. [Core Utilities](#2-core-utilities)
3. [Machine Learning Workflow](#3-machine-learning-workflow)
4. [Deep Learning & Neural Networks](#4-deep-learning--neural-networks)
5. [LLM & RAG Pipelines](#5-llm--rag-pipelines)
6. [Production Deployment](#6-production-deployment)

---

## 1. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kandil7/AI-Mastery-2026.git
   cd AI-Mastery-2026
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # OR
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation:**
   Run the verification script to ensure all modules are correctly installed:
   ```bash
   python scripts/verify_toolkit.py
   ```
   *Expected Output:* `[SUCCESS] AI Engineer Toolkit verification complete!`

---

## 2. Core Utilities

The `src.core` package provides the mathematical building blocks.

### Linear Algebra & Statistics
```python
import numpy as np
from src.core.math_operations import dot_product, cosine_similarity, PCA
from src.core.probability import Gaussian

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
similarity = cosine_similarity(v1, v2)
print(f"Cosine Similarity: {similarity:.4f}")

# Dimensionality Reduction (PCA)
X = np.random.rand(100, 10)  # 100 samples, 10 features
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Reduced shape: {X_reduced.shape}")  # (100, 2)

# Probability Distributions
dist = Gaussian(mean=0, std=1)
samples = dist.sample(5)
print(f"Samples from Normal(0,1): {samples}")
```

### Optimization
Use standard optimizers (SGD, Adam) for your custom loops.
```python
from src.core.optimization import Adam

param = np.array([10.0])  # Initial parameter
optimizer = Adam(learning_rate=0.1)

def loss_fn(p):
    return (p[0] ** 2), 2 * p  # Loss: p^2, Grad: 2p

for i in range(50):
    loss, grad = loss_fn(param)
    param = optimizer.step(param, grad)

print(f"Optimized parameter: {param[0]:.4f} (Expected: ~0.0)")
```

---

## 3. Machine Learning Workflow

Train classical ML models using `src.ml.classical`.

### Linear Regression (From Scratch)
```python
from src.ml.classical import LinearRegressionScratch
from src.ml.evaluation import RegressionMetrics
import numpy as np

# Generate synthetic data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) * 0.1

# Train
model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

# Predict
preds = model.predict(X)

# Evaluate
metrics = RegressionMetrics(y, preds)
print(f"MSE: {metrics.mse():.4f}")
print(f"R2 Score: {metrics.r2_score():.4f}")
```

---

## 4. Deep Learning & Neural Networks

Build neural networks using the `src.ml.deep_learning` module (API similar to Keras/PyTorch).

```python
from src.ml.deep_learning import Sequential, Dense, Activation
from src.ml.loss import CrossEntropyLoss
from src.ml.optimizers import Adam

# Define Architecture
model = Sequential()
model.add(Dense(input_size=784, output_size=128))
model.add(Activation('relu'))
model.add(Dense(input_size=128, output_size=10))
model.add(Activation('softmax'))

# Compile
model.compile(
    loss=CrossEntropyLoss(),
    optimizer=Adam(learning_rate=0.001)
)

# Train
loss_history = model.fit(X_train, y_train, epochs=10, batch_size=32)
```

---

## 5. LLM & RAG Pipelines

Build advanced LLM applications using `src.llm`.

### RAG Pipeline (Retrieval Augmented Generation)
```python
from src.llm.rag import RAGPipeline
from src.llm.models import MockLLM  # Or connect to OpenAI/Local

# Initialize
rag = RAGPipeline(
    llm=MockLLM(),
    embedding_dim=384
)

# Add Knowledge Base
documents = [
    "The AI Toolkit was built in 2024.",
    "It emphasizes a White-Box approach to learning."
]
rag.index_documents(documents)

# Query
response = rag.query("What represents the core philosophy?")
print(response)
```

### Agents (ReAct Pattern)
```python
from src.llm.agents import ReActAgent, Tool

def calculator(expression):
    return eval(expression)

tools = [
    Tool(name="calc", func=calculator, description="Useful for math")
]

agent = ReActAgent(
    llm=MockLLM(),
    tools=tools,
    verbose=True
)

agent.run("What is 123 * 45?")
```

---

## 6. Production Deployment

Deploy your models using `src.production`.

### Caching
Optimize performance with cached predictions.
```python
from src.production.caching import LRUCache, cached

cache = LRUCache(max_size=1000)

@cached(cache)
def slow_inference(input_data):
    # Simulate heavy compute
    return model.predict(input_data)
```

### Model Serving (FastAPI)
Using the provided deployment script or creating a custom server.

1. **Define the API (`app.py`):**
   ```python
   from fastapi import FastAPI
   from src.production.api import ModelServer
   
   app = FastAPI()
   server = ModelServer(model)
   
   @app.post("/predict")
   async def predict(data: dict):
       return server.predict(data)
   ```

2. **Run the Server:**
   ```bash
   uvicorn app:app --reload
   ```

### Docker Deployment
Build and run the containerized application.
```bash
# Build
docker-compose build

# Run
docker-compose up -d
```

---

## 7. Case Studies

The toolkit includes full implementations of real-world systems.

### Legal Document RAG System
Located in `case_studies/legal_document_rag_system/`.
*   **Architecture**: Hybrid search (Keyword + Vector) with citation tracking.
*   **Key Components**: `DocumentProcessor`, `VectorStore`, `QueryEngine`.
*   **Run Evaluation**:
    ```bash
    python case_studies/legal_document_rag_system/run_evaluation.py
    ```

### Medical Diagnosis Agent
Located in `case_studies/medical_diagnosis_agent/`.
*   **Architecture**: Privacy-first agent with PII stripping and clinical validation.
*   **Key Components**: `PIIFilter`, `DiagnosticEngine`, `SafetyValidator`.
*   **Run Demo**:
    ```bash
    python case_studies/medical_diagnosis_agent/run_demo.py
    ```

---

## Educational Resources
Explore the `notebooks/` directory for weekly learning materials:
- **Week 1**: Mathematical Foundations (`notebooks/week_01/`)
- **Week 2**: Neural Networks from Scratch (`notebooks/week_02/`)
- **Week 3**: Classical ML Algorithms (`notebooks/week_03/`)
