# Core Mathematics Module

**Mathematical foundations implemented from first principles.**

This module contains pure Python implementations of mathematical concepts without relying on NumPy for the core algorithms.

## Topics Covered

### Linear Algebra
- Vector and Matrix operations
- Matrix decompositions (SVD, QR, Cholesky)
- Eigenvalue problems

### Calculus
- Numerical differentiation
- Integration (Newton-Cotes, Gaussian quadrature)
- Gradient computation

### Optimization
- Gradient descent variants (SGD, Adam, RMSprop)
- Constrained optimization
- Learning rate schedules

### Probability & Statistics
- Distributions
- Hypothesis testing
- Bayesian inference

### Advanced Topics
- MCMC methods
- Variational inference
- Causal inference
- Differential privacy

## Usage

```python
from src.core import Adam, Matrix, Vector
from src.core.optimization import GradientDescent
from src.core.linear_algebra import dot_product, matrix_multiply

# Create vectors
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

# Compute dot product
result = dot_product(v1, v2)

# Initialize optimizer
optimizer = Adam(learning_rate=0.001)
```

## Implementation Philosophy

1. **Math First** - Derive formulas on paper
2. **Code Second** - Implement in pure Python
3. **Optimize Third** - Add NumPy/PyTorch for performance

## Related Modules

- [`src/ml`](../ml/) - Machine learning algorithms
- [`src/llm`](../llm/) - Transformer architectures
