# 🧠 AI-Mastery-2026: From Math to Production

> A comprehensive AI Engineer Toolkit built from first principles following the **White-Box Approach**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

---

## 🎯 Philosophy

**White-Box Approach**: Understand mechanics before using abstractions.

1. **Math First** → Derive equations, understand foundations
2. **Code Second** → Implement from scratch with NumPy
3. **Libraries Third** → Use sklearn/PyTorch knowing what happens underneath
4. **Production Always** → Every concept includes deployment considerations

---

## 📁 Repository Structure

```
AI-Mastery-2026/
├── src/
│   ├── core/                 # Mathematical foundations
│   │   ├── math_operations.py    # Linear algebra, PCA, activations
│   │   └── optimization.py       # SGD, Adam, schedulers, regularization
│   ├── ml/                   # Machine Learning
│   │   ├── classical.py          # Linear/Logistic regression, trees, SVM
│   │   └── deep_learning.py      # Neural networks, layers, backprop
│   ├── production/           # Production Engineering
│   │   ├── api.py                # FastAPI model serving
│   │   ├── monitoring.py         # Drift detection, metrics
│   │   └── vector_db.py          # HNSW, LSH vector search
│   └── llm/                  # LLM Engineering
│       ├── attention.py          # Transformers, multi-head attention
│       ├── fine_tuning.py        # LoRA, QLoRA, adapters
│       └── rag.py                # RAG pipeline components
├── research/                 # Jupyter notebooks (17 weeks)
├── tests/                    # Unit tests
├── docs/                     # Technical documentation
└── scripts/                  # Automation tools
```

---

## 📚 Documentation

For a complete breakdown of the architecture, modules, development workflows, and deployment, please see the **[Complete User Guide](./docs/guide/00_index.md)**.

---

## 🚀 Quick Start

```bash
# Clone the repository
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

# Run tests
pytest tests/ -v
```

---

## 📚 Core Modules

### `src/core/` - Mathematical Foundations

```python
from src.core.math_operations import cosine_similarity, PCA, softmax
from src.core.optimization import Adam, CosineAnnealingLR

# PCA from scratch
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Adam optimizer
optimizer = Adam(learning_rate=0.001)
params = optimizer.step(params, gradients)
```

### `src/ml/` - Machine Learning Algorithms

```python
from src.ml.classical import LinearRegressionScratch, RandomForestScratch
from src.ml.deep_learning import NeuralNetwork, Dense, Activation

# Train linear regression
model = LinearRegressionScratch(regularization='l2')
model.fit(X_train, y_train)

# Build neural network
nn = NeuralNetwork()
nn.add(Dense(784, 256, weight_init='he'))
nn.add(Activation('relu'))
nn.add(Dense(256, 10))
nn.add(Activation('softmax'))
nn.compile(loss=CrossEntropyLoss(), learning_rate=0.001)
nn.fit(X_train, y_train, epochs=100)
```

### `src/production/` - Production Engineering

```python
from src.production.api import create_app, model_cache
from src.production.monitoring import DriftDetector, PerformanceMonitor
from src.production.vector_db import HNSW, VectorIndex

# Drift detection
detector = DriftDetector(method='ks')
detector.set_reference(X_train)
results = detector.detect_drift(X_production)

# Vector search
index = HNSW(dim=384, M=16)
index.add_items(embeddings, ids)
results = index.search(query_embedding, k=10)
```

### `src/llm/` - LLM Engineering

```python
from src.llm.attention import MultiHeadAttention, TransformerBlock
from src.llm.fine_tuning import LoRALayer, LinearWithLoRA
from src.llm.rag import RAGPipeline, TextChunker

# Transformer attention
mha = MultiHeadAttention(d_model=512, num_heads=8)
output = mha(Q, K, V)

# LoRA fine-tuning
lora = LoRALayer(in_features=768, out_features=768, r=8)
adapted_output = base_output + lora.forward(x)

# RAG pipeline
rag = RAGPipeline()
rag.add_documents(documents)
response = rag.query("What is transformer attention?")
```

---

## 🗺️ Learning Roadmap

| Phase | Focus | Key Topics |
|-------|-------|------------|
| 1 | **Math Foundations** | Linear Algebra, Calculus, Probability |
| 2 | **Classical ML** | Linear Models, Trees, SVM, Ensemble |
| 3 | **Deep Learning** | Neural Networks, CNN, RNN, Backprop |
| 4 | **Transformers** | Attention, BERT, GPT Architecture |
| 5 | **LLM Engineering** | Fine-tuning, RAG, Agents |
| 6 | **Production** | API, Monitoring, Deployment, Scale |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific module tests
pytest tests/test_linear_algebra.py -v
```

---

## 📖 Key Implementations

### From-Scratch Algorithms

- **Linear Regression** - Closed-form + Gradient Descent
- **Logistic Regression** - Binary + Multiclass (Softmax/OvR)
- **Decision Trees** - ID3/CART with Gini/Entropy
- **Neural Networks** - Full backpropagation
- **Transformers** - Scaled dot-product attention
- **HNSW** - Approximate nearest neighbor search
- **LoRA** - Low-rank adaptation for fine-tuning

### Production Components

- **FastAPI** - Model serving with Pydantic validation
- **SSE Streaming** - Real-time LLM responses
- **Drift Detection** - KS test, PSI monitoring
- **Vector DB** - HNSW, LSH implementations

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'feat: add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- "Attention Is All You Need" (Vaswani et al., 2017)
- "LoRA: Low-Rank Adaptation" (Hu et al., 2021)
- Fast.ai, Andrej Karpathy, 3Blue1Brown

---

*Started: December 2024 | Goal: Full-Stack AI Engineer*
