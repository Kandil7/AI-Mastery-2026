# Getting Started with AI-Mastery-2026

**Last Updated:** March 28, 2026  
**Level:** Beginner  
**Estimated Time:** 30 minutes

---

## 🎯 Overview

This guide will help you get started with AI-Mastery-2026, the comprehensive AI engineering toolkit that takes you from first principles to production-scale systems.

### What You'll Learn

- Project overview and philosophy
- System requirements
- Installation steps
- Verification and testing
- Your first steps

---

## 📋 Prerequisites

### Required Knowledge

| Topic | Level | Why It's Needed |
|-------|-------|-----------------|
| Python Programming | Intermediate | All code examples are in Python |
| Basic Mathematics | High School | Linear algebra, calculus fundamentals |
| Command Line | Basic | Running scripts and tools |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 / macOS 10.15 / Linux | Windows 11 / macOS 12+ / Ubuntu 22.04+ |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 10 GB free | 50+ GB SSD |
| GPU | Optional | NVIDIA GPU with 8GB+ VRAM |

### Software Requirements

```bash
# Required
Python 3.10+
Git
pip or conda

# Recommended
CUDA 11.8+ (for GPU acceleration)
Docker (for containerized deployment)
Make (for running commands)
```

---

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Kandil7/AI-Mastery-2026.git

# Navigate to project directory
cd AI-Mastery-2026
```

### Step 2: Set Up Virtual Environment

#### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

#### Option B: Using conda

```bash
# Create conda environment
conda create -n ai-mastery python=3.10

# Activate environment
conda activate ai-mastery
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or use make (if available)
make install
```

### Step 4: Verify Installation

```bash
# Run tests to verify installation
pytest tests/ -v

# Or use make
make test
```

**Expected Output:**
```
============================= test session starts ==============================
platform win32 -- Python 3.10.0, pytest-7.0.0
collected 150 items

tests/test_core.py ..................................................... [ 35%]
tests/test_ml.py ....................................................... [ 74%]
tests/test_llm.py ........................................               [100%]

============================= 150 passed in 45.2s ==============================
```

---

## 🎓 Your First Steps

### 1. Explore the Learning Roadmap

Start with the [Learning Roadmap](../01_learning_roadmap/README.md) to understand your journey:

```
Phase 1: Foundations (Weeks 1-4)
├── Mathematical Foundations
├── Python for AI
└── Core Concepts

Phase 2: Machine Learning (Weeks 5-10)
├── Classical ML Algorithms
├── Deep Learning Basics
└── Neural Networks

Phase 3: Advanced Topics (Weeks 11-18)
├── LLM Engineering
├── RAG Systems
└── AI Agents

Phase 4: Production (Weeks 19-24)
├── Deployment Strategies
├── Monitoring & Observability
└── Scale & Optimization
```

### 2. Run Your First Notebook

```bash
# Navigate to notebooks
cd notebooks/01_foundations

# Start Jupyter
jupyter notebook

# Open: 01_introduction_to_python_for_ai.ipynb
```

### 3. Explore Core Implementations

```bash
# Explore pure Python implementations
cd src/core

# View linear algebra implementation
cat linear_algebra.py
```

### 4. Build Your First Model

```python
# Create a simple test script
cat > test_model.py << 'EOF'
from src.ml.classical import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2
)

# Train model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
EOF

# Run it
python test_model.py
```

---

## 📚 Learning Resources

### Documentation Sections

| Section | Description | Link |
|---------|-------------|------|
| User Guides | How-to documentation | [guides/](guides/) |
| Tutorials | Step-by-step lessons | [tutorials/](tutorials/) |
| API Reference | Complete API docs | [api/](api/) |
| Knowledge Base | Concepts & best practices | [kb/](kb/) |

### Recommended Learning Path

1. **Week 1-2:** [Mathematical Foundations](../notebooks/01_mathematical_foundations/)
2. **Week 3-4:** [Classical ML](../notebooks/02_classical_ml/)
3. **Week 5-8:** [Deep Learning](../notebooks/03_deep_learning/)
4. **Week 9-12:** [LLM Engineering](../notebooks/04_llm/)
5. **Week 13-16:** [RAG Systems](../notebooks/RAG/)

---

## 🛠️ Development Setup

### IDE Configuration

#### VS Code (Recommended)

1. Install extensions:
   - Python
   - Jupyter
   - Pylance
   - Black Formatter

2. Configure settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

### Git Configuration

```bash
# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set up pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## ✅ Installation Checklist

Use this checklist to ensure everything is set up correctly:

- [ ] Repository cloned successfully
- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] Tests pass (`make test` or `pytest`)
- [ ] Jupyter notebooks working
- [ ] GPU acceleration working (if applicable)
- [ ] Docker installed (optional)
- [ ] Pre-commit hooks installed

---

## 🐛 Common Issues

### Issue: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: CUDA Not Found

**Problem:** `CUDA not available` warnings

**Solution:**
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Memory Errors

**Problem:** Out of memory during training

**Solution:**
```python
# Reduce batch size
batch_size = 8  # Instead of 32

# Use gradient accumulation
# See: tutorials/intermediate/gradient-accumulation.md
```

---

## 📞 Getting Help

### Resources

| Resource | Purpose | Link |
|----------|---------|------|
| Documentation | Comprehensive guides | [docs/](docs/) |
| FAQ | Common questions | [faq/](faq/) |
| GitHub Issues | Bug reports | [Issues](https://github.com/Kandil7/AI-Mastery-2026/issues) |
| Discussions | Questions & ideas | [Discussions](https://github.com/Kandil7/AI-Mastery-2026/discussions) |

### Support Channels

1. **Check Documentation:** Search [docs/](docs/) first
2. **Review FAQ:** See [faq/](faq/) for common questions
3. **Search Issues:** Check existing GitHub issues
4. **Ask in Discussions:** Post in GitHub Discussions
5. **Report Bugs:** Create a detailed issue report

---

## 🎯 Next Steps

Now that you're set up, continue your journey:

1. **[Installation Guide](installation.md)** - Detailed setup options
2. **[Learning Roadmap](../01_learning_roadmap/README.md)** - Your learning path
3. **[First Tutorial](../tutorials/beginner/first-neural-network.md)** - Hands-on practice
4. **[API Reference](api/overview.md)** - Explore the API

---

## 📝 Quick Reference

### Essential Commands

```bash
# Install dependencies
make install

# Run tests
make test

# Run linter
make lint

# Build documentation
make docs

# Clean build artifacts
make clean

# Run all checks
make all
```

### Project Structure

```
AI-Mastery-2026/
├── src/              # Source code
├── tests/            # Test suite
├── notebooks/        # Jupyter notebooks
├── docs/             # Documentation
├── scripts/          # Utility scripts
└── config/           # Configuration files
```

---

**Congratulations!** You're now ready to start your AI engineering journey with AI-Mastery-2026.

For detailed module-specific guides, see the [User Guides](guides/) section.
