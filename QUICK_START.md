# AI-Mastery-2026 Quick Start Guide

Get up and running with AI-Mastery-2026 in 5 minutes!

---

## ⚡ 1-Minute Installation

```bash
# Clone the repository
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026

# Install with pip
pip install ai-mastery-2026

# Verify installation
python -c "from src.core import Adam; print('✅ Installation successful!')"
```

---

## 🚀 5-Minute Quick Start

### Step 1: Install Development Version

```bash
# Clone and install in editable mode
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026
pip install -e ".[dev]"
```

### Step 2: Verify Installation

```bash
# Run tests
pytest tests/unit -v --tb=short

# Run linters
make lint

# Check imports
python -c "
from src.config import get_settings
from src.ml import NeuralNetwork
from src.llm import MultiHeadAttention
from src.rag import RAGPipeline
print('✅ All imports working!')
"
```

### Step 3: Try Your First Example

```python
# Create a simple neural network
from src.ml import NeuralNetwork
import numpy as np

# Generate sample data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

# Create and train model
model = NeuralNetwork(input_dim=10, hidden_dims=[32, 16], output_dim=2)
model.fit(X, y, epochs=10)

# Make predictions
predictions = model.predict(X[:5])
print(f"Predictions: {predictions}")
```

---

## 📚 Next Steps

### Learn the Fundamentals

1. **Mathematics** - Start with [Linear Algebra](notebooks/tier1_fundamentals/week_01/01_linear_algebra_from_scratch.ipynb)
2. **Classical ML** - Try [Decision Trees](notebooks/tier1_fundamentals/week_02/05_decision_trees_complete.ipynb)
3. **Deep Learning** - Explore [Neural Networks](notebooks/tier1_fundamentals/week_02/02_neural_networks.ipynb)

### Build Something

- **RAG System** - [Complete RAG Pipeline](notebooks/tier3_llm_engineer/week_09/02_rag_pipeline.ipynb)
- **LLM Fine-tuning** - [LoRA Implementation](notebooks/tier3_llm_engineer/week_09/01_lora_implementation.ipynb)
- **Production API** - [FastAPI Deployment](notebooks/tier4_production/week_13/01_fastapi_serving.ipynb)

### Join the Community

- 📖 [Read Documentation](https://kandil7.github.io/AI-Mastery-2026/)
- 💬 [Join Discussions](https://github.com/Kandil7/AI-Mastery-2026/discussions)
- 🐛 [Report Issues](https://github.com/Kandil7/AI-Mastery-2026/issues)
- ⭐ [Star the Project](https://github.com/Kandil7/AI-Mastery-2026)

---

## 🆘 Troubleshooting

### Common Issues

**Import Error:**
```bash
# Ensure package is installed
pip install -e .

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
```

**Test Failures:**
```bash
# Install test dependencies
pip install -e ".[dev]"

# Run specific test
pytest tests/unit/core/test_optimization.py -v
```

**Documentation Build:**
```bash
# Install docs dependencies
pip install mkdocs mkdocs-material

# Build locally
mkdocs build

# Serve locally
mkdocs serve
```

---

## 📞 Need Help?

- **Documentation:** https://kandil7.github.io/AI-Mastery-2026/
- **Issues:** https://github.com/Kandil7/AI-Mastery-2026/issues
- **Discussions:** https://github.com/Kandil7/AI-Mastery-2026/discussions
- **Email:** medokandeal7@gmail.com

---

**Last Updated:** March 31, 2026
