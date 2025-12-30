# Contributing to AI Engineer Toolkit 2025

Thank you for your interest in contributing! This project follows the **White-Box Approach** to AI/ML engineering.

## Philosophy

Our core principles:
1. **Math First** - Understand the mathematical foundations before implementation
2. **Code Second** - Implement from scratch using pure NumPy before using libraries
3. **Production Always** - Consider deployment, monitoring, and scaling from day one

## How to Contribute

### 1. Fork & Clone

```bash
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026
```

### 2. Set Up Development Environment

```bash
# Create conda environment
conda create -n ai-mastery python=3.10
conda activate ai-mastery

# Install dependencies
make install-dev
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes

Follow these guidelines:
- Add mathematical derivations in docstrings
- Include unit tests for new functions
- Follow the existing code style

### 5. Test Your Changes

```bash
make test
make lint
```

### 6. Submit Pull Request

- Write clear PR description
- Reference any related issues
- Ensure CI passes

## Code Style

- **Python 3.10+** compatible
- **Type hints** for all functions
- **Docstrings** with mathematical notation
- **100 character** line limit

## Documentation Standards

Each function should include:
1. Brief description
2. Mathematical definition (using Unicode symbols)
3. Args and Returns sections
4. Example usage

```python
def example_function(x: np.ndarray) -> np.ndarray:
    """
    Brief description of what this does.
    
    Mathematical Definition:
        f(x) = σ(Wx + b)
    
    Args:
        x: Input array of shape (n, m)
    
    Returns:
        Transformed array
    
    Example:
        >>> result = example_function(np.array([1, 2, 3]))
    """
    pass
```

## Questions?

Open an issue or reach out to the maintainers.

---

Thank you for contributing! 🚀
