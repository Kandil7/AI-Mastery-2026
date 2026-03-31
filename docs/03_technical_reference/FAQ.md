# Frequently Asked Questions (FAQ)

Common questions about AI-Mastery-2026.

---

## General Questions

### What is AI-Mastery-2026?

AI-Mastery-2026 is a comprehensive AI engineering toolkit designed to teach AI from first principles. It includes:
- Mathematical foundations implemented from scratch
- Classical and deep learning algorithms
- LLM engineering components
- RAG systems
- Production-ready infrastructure

### Who is this project for?

- **Students** learning AI/ML fundamentals
- **Engineers** wanting to understand internals
- **Researchers** needing reference implementations
- **Educators** teaching AI courses

### What Python version is required?

Python 3.10 or higher is required. We recommend Python 3.11 for best compatibility.

### Is this project production-ready?

The core components are stable and tested. However, this is primarily an educational project. For production use, we recommend:
- Thorough testing in your environment
- Security review
- Performance benchmarking

---

## Installation

### How do I install AI-Mastery-2026?

```bash
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026
pip install -e ".[dev]"
```

### Can I install only specific components?

Yes! Use optional dependencies:
```bash
# ML only
pip install ai-mastery-2026[ml]

# LLM only
pip install ai-mastery-2026[llm]

# Vector databases
pip install ai-mastery-2026[vector]
```

### Does this work on Windows/Mac/Linux?

Yes! AI-Mastery-2026 is cross-platform compatible.

---

## Usage

### How do I get started?

1. See the [Quick Start Guide](../00_introduction/QUICK_START.md)
2. Run the example notebooks in `notebooks/`
3. Check the [Learning Roadmap](../01_learning_roadmap/README.md)

### Can I use this in my commercial project?

Yes! The project is licensed under MIT License, which allows commercial use. See [LICENSE](../LICENSE) for details.

### How do I cite this project?

See [CITATION.cff](../CITATION.cff) for citation formats.

---

## Technical Questions

### What's the difference between `classical_scratch.py` and `classical/`?

- `classical_scratch.py`: From-scratch implementations for learning
- `classical/`: Organized module with individual algorithm files

Both implement the same algorithms; the scratch version is for educational purposes.

### How do I use the configuration system?

```python
from src.config import get_settings, TransformerConfig

# Get global settings
settings = get_settings()

# Create model config
config = TransformerConfig(hidden_dim=768, num_heads=12)
```

See [Configuration Guide](./configuration.md) for details.

### Can I use my own models with this framework?

Yes! The framework is designed to be extensible:
```python
from src.types import Trainable

class MyModel(Trainable):
    def fit(self, X, y):
        # Your implementation
        pass
```

---

## Development

### How do I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

### How do I run tests?

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit -v

# With coverage
pytest --cov=src
```

### How do I build documentation?

```bash
pip install mkdocs mkdocs-material
mkdocs serve  # Local development
mkdocs build  # Static site
```

---

## Troubleshooting

### Where can I find help?

1. [Troubleshooting Guide](./TROUBLESHOOTING.md)
2. [GitHub Issues](https://github.com/Kandil7/AI-Mastery-2026/issues)
3. [GitHub Discussions](https://github.com/Kandil7/AI-Mastery-2026/discussions)

### I found a bug. What should I do?

Please report it on [GitHub Issues](https://github.com/Kandil7/AI-Mastery-2026/issues) using the bug report template.

### How do I report a security vulnerability?

See [SECURITY.md](../SECURITY.md) for responsible disclosure guidelines.

---

## Learning Path

### Where should I start learning?

Follow the [Learning Roadmap](../01_learning_roadmap/README.md):
1. Core mathematics
2. Classical ML
3. Deep Learning
4. LLM Engineering
5. RAG Systems
6. Production

### Are there video tutorials?

Video tutorials are planned. Check the [curriculum](../curriculum/) for updates.

### How long does it take to complete?

The full curriculum is designed for 16-20 weeks of study (approximately 10-15 hours/week).

---

## Performance

### How does this compare to production libraries?

Our from-scratch implementations are **educational** and not optimized for production. Expect:
- **Slower** than optimized libraries (PyTorch, TensorFlow)
- **Better** for understanding internals
- **Comparable** API design for easy transition

### Can I use GPU acceleration?

Yes! Many components support GPU acceleration via PyTorch:
```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Licensing

### What license is this project under?

MIT License - see [LICENSE](../LICENSE)

### Can I use this in my thesis/dissertation?

Yes! Please cite the project as described in [CITATION.cff](../CITATION.cff).

### Are there any restrictions on use?

No significant restrictions. The MIT License allows:
- Commercial use
- Modification
- Distribution
- Private use

---

## Contact

### How can I reach the maintainers?

- **Email:** medokandeal7@gmail.com
- **GitHub:** @Kandil7
- **Discussions:** [GitHub Discussions](https://github.com/Kandil7/AI-Mastery-2026/discussions)

### Do you offer support?

Community support is available through GitHub Issues and Discussions. For enterprise support, please contact us directly.

---

**Last Updated:** March 31, 2026
