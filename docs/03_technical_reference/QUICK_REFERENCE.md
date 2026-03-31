# Quick Reference Cards

Quick reference for common tasks in AI-Mastery-2026.

---

## 📦 Installation

```bash
# Clone and install
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026
pip install -e ".[dev]"

# Verify installation
python -c "from src.core import Adam; print('OK')"
```

---

## ⚙️ Configuration

```python
from src.config import get_settings, TransformerConfig, TrainingConfig

# Get settings
settings = get_settings()
print(settings.environment)

# Model config
model_cfg = TransformerConfig(
    hidden_dim=768,
    num_heads=12,
    num_layers=12
)

# Training config
train_cfg = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=10
)
```

---

## 🔧 Common Imports

```python
# Core
from src.core import Adam, ExtendedKalmanFilter, metropolis_hastings

# ML
from src.ml import NeuralNetwork, DecisionTree, ResNet18

# LLM
from src.llm import MultiHeadAttention, BERT, FineTuner

# RAG
from src.rag import Document, RAGPipeline, SemanticChunker

# Agents
from src.agents import SupportAgent, MultiAgent, ReActAgent

# Types
from src.types import DocumentProtocol, EmbeddingVector, ModelOutput
```

---

## 📊 Data Loading

```python
from src.config import DataConfig

config = DataConfig(
    data_dir="data/",
    train_file="train.csv",
    batch_size=32
)

# Check if files exist
if config.has_train:
    print("Training data available")
```

---

## 🤖 Model Training

```python
from src.ml import NeuralNetwork
from src.config import TrainingConfig

# Create model
model = NeuralNetwork(
    input_dim=10,
    hidden_dims=[64, 32],
    output_dim=4
)

# Training config
config = TrainingConfig(
    learning_rate=0.001,
    batch_size=32
)

# Train (pseudo-code)
# model.fit(X_train, y_train, config=config)
```

---

## 📝 RAG Pipeline

```python
from src.rag import RAGPipeline, Document, SemanticChunker

# Create pipeline
pipeline = RAGPipeline(
    chunker=SemanticChunker(chunk_size=512),
    # ... other components
)

# Add documents
docs = [
    Document(id="1", content="AI is transforming industries."),
    Document(id="2", content="Machine learning powers modern AI."),
]
pipeline.add_documents(docs)

# Query
results = pipeline.query("How does AI work?", k=5)
```

---

## 🧪 Testing

```python
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/unit/core/test_optimization.py -v

# With coverage
pytest --cov=src --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

---

## 🔍 Debugging

```python
from src.config import Settings, Environment

# Enable debug mode
settings = Settings(
    environment=Environment.DEVELOPMENT,
    debug=True,
    log_level="DEBUG"
)

# Check device
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📈 Performance

```python
# GPU acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Mixed precision
from torch.cuda.amp import autocast

with autocast():
    output = model(input)

# Clear cache
import torch
torch.cuda.empty_cache()
```

---

## 🚀 Deployment

```bash
# Run API
uvicorn src.production.api:app --host 0.0.0.0 --port 8000

# Docker
docker build -t ai-mastery .
docker run -p 8000:8000 ai-mastery

# Production
gunicorn src.production.api:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 📚 Documentation

```bash
# Build docs
mkdocs build

# Serve locally
mkdocs serve

# Deploy
mkdocs gh-deploy --force
```

---

## 🔧 Code Quality

```bash
# Format
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/

# All checks
./scripts/lint.sh
```

---

## 📋 Common Patterns

### Custom Model

```python
from src.types import Trainable
import numpy as np

class MyModel(Trainable):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MyModel":
        # Training logic
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Prediction logic
        pass
```

### Custom Chunker

```python
from src.rag.chunking import BaseChunker, Chunk

class MyChunker(BaseChunker):
    def chunk(self, document: dict) -> list[Chunk]:
        # Chunking logic
        pass
```

### Custom Agent

```python
from src.agents.orchestration import Agent

class MyAgent(Agent):
    def act(self, observation) -> str:
        # Agent logic
        pass
```

---

## 🆘 Help

```bash
# Check version
python -c "import src; print(src.__version__)"

# List available modules
python -c "import src; print(dir(src))"

# Get help
python -c "from src.ml import NeuralNetwork; help(NeuralNetwork)"
```

---

## 📞 Contact

- **Issues:** https://github.com/Kandil7/AI-Mastery-2026/issues
- **Discussions:** https://github.com/Kandil7/AI-Mastery-2026/discussions
- **Email:** medokandeal7@gmail.com

---

**Last Updated:** March 31, 2026
