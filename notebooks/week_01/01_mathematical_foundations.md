# Week 01: Mathematical Foundations for ML

This notebook covers the essential mathematical concepts for machine learning:
- Linear Algebra (vectors, matrices, eigenvalues)
- Calculus (gradients, chain rule)
- Probability (distributions, Bayes theorem)

## Learning Objectives
After completing this notebook, you will:
1. Implement vector/matrix operations from scratch
2. Understand gradient computation
3. Apply probability distributions to ML problems

---

## 1. Linear Algebra Foundations

### 1.1 Vector Operations

```python
import numpy as np
from typing import List, Tuple

# Type aliases
Vector = np.ndarray
Matrix = np.ndarray


def dot_product(v1: Vector, v2: Vector) -> float:
    """
    Compute dot product of two vectors.
    
    Mathematical definition:
    v1 · v2 = Σ(v1[i] * v2[i]) for i in range(n)
    
    Properties:
    - Commutative: v1 · v2 = v2 · v1
    - Distributive: a · (b + c) = a·b + a·c
    - Scalar multiplication: (ca) · b = c(a · b)
    
    Example:
    >>> dot_product(np.array([1, 2, 3]), np.array([4, 5, 6]))
    32
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same length")
    
    result = 0.0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result


def vector_norm(v: Vector, p: int = 2) -> float:
    """
    Compute Lp norm of a vector.
    
    L1 norm: ||v||_1 = Σ|v[i]|
    L2 norm: ||v||_2 = √(Σv[i]²)
    L∞ norm: ||v||_∞ = max|v[i]|
    """
    if p == 1:
        return np.sum(np.abs(v))
    elif p == 2:
        return np.sqrt(np.sum(v ** 2))
    elif p == np.inf:
        return np.max(np.abs(v))
    else:
        return np.sum(np.abs(v) ** p) ** (1/p)


def cosine_similarity(v1: Vector, v2: Vector) -> float:
    """
    Cosine similarity: measures angle between vectors.
    
    cos(θ) = (v1 · v2) / (||v1|| * ||v2||)
    
    Range: [-1, 1]
    - 1: vectors point same direction
    - 0: orthogonal
    - -1: opposite directions
    """
    dot = dot_product(v1, v2)
    norm1 = vector_norm(v1)
    norm2 = vector_norm(v2)
    return dot / (norm1 * norm2)


# Test
v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([4.0, 5.0, 6.0])

print(f"Dot product: {dot_product(v1, v2)}")
print(f"L2 norm of v1: {vector_norm(v1):.4f}")
print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")
```

### 1.2 Matrix Operations

```python
def matrix_multiply(A: Matrix, B: Matrix) -> Matrix:
    """
    Matrix multiplication from scratch.
    
    C[i,j] = Σ(A[i,k] * B[k,j]) for k in range(n)
    
    Requirements:
    - A: (m, n) matrix
    - B: (n, p) matrix
    - Result: (m, p) matrix
    """
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError(f"Matrix dimensions don't match: {A.shape} vs {B.shape}")
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matrix_transpose(A: Matrix) -> Matrix:
    """Transpose: swap rows and columns."""
    m, n = A.shape
    result = np.zeros((n, m))
    for i in range(m):
        for j in range(n):
            result[j, i] = A[i, j]
    return result


# Test
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[7, 8, 9], [10, 11, 12]])

print("A @ B =")
print(matrix_multiply(A, B))
print("\nA.T =")
print(matrix_transpose(A))
```

### 1.3 Eigenvalues and Eigenvectors

```python
def power_iteration(A: Matrix, num_iterations: int = 100) -> Tuple[float, Vector]:
    """
    Find dominant eigenvalue and eigenvector using power iteration.
    
    Algorithm:
    1. Start with random vector v
    2. Multiply: v = A @ v
    3. Normalize: v = v / ||v||
    4. Repeat until convergence
    
    The eigenvalue is the Rayleigh quotient: λ = (v.T @ A @ v) / (v.T @ v)
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(num_iterations):
        w = A @ v
        v = w / np.linalg.norm(w)
    
    # Rayleigh quotient for eigenvalue
    eigenvalue = (v.T @ A @ v) / (v.T @ v)
    
    return eigenvalue, v


# Test with symmetric matrix
A = np.array([[4.0, 2.0], [2.0, 3.0]])
eigenvalue, eigenvector = power_iteration(A)

print(f"Dominant eigenvalue: {eigenvalue:.4f}")
print(f"Corresponding eigenvector: {eigenvector}")

# Verify
print(f"A @ v: {A @ eigenvector}")
print(f"λ * v: {eigenvalue * eigenvector}")
```

---

## 2. Calculus for Machine Learning

### 2.1 Numerical Gradients

```python
def numerical_gradient(f, x: Vector, h: float = 1e-5) -> Vector:
    """
    Compute gradient using central difference.
    
    ∂f/∂x_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)
    
    More accurate than forward difference.
    """
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad


# Test: gradient of f(x,y) = x² + 2xy + y²
def f(x):
    return x[0]**2 + 2*x[0]*x[1] + x[1]**2

point = np.array([1.0, 2.0])
grad = numerical_gradient(f, point)
print(f"Numerical gradient at {point}: {grad}")

# Analytical gradient: [2x + 2y, 2x + 2y]
analytical = np.array([2*1 + 2*2, 2*1 + 2*2])
print(f"Analytical gradient: {analytical}")
```

### 2.2 Chain Rule

```python
def chain_rule_example():
    """
    Demonstrate chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)
    
    Example: z = (x² + y²)³
    Let u = x² + y², then z = u³
    
    ∂z/∂x = ∂z/∂u * ∂u/∂x = 3u² * 2x = 6x(x² + y²)²
    """
    x, y = 2.0, 3.0
    
    # Forward pass
    u = x**2 + y**2  # u = 13
    z = u**3         # z = 2197
    
    # Backward pass (chain rule)
    dz_du = 3 * u**2    # = 3 * 169 = 507
    du_dx = 2 * x       # = 4
    du_dy = 2 * y       # = 6
    
    dz_dx = dz_du * du_dx  # = 507 * 4 = 2028
    dz_dy = dz_du * du_dy  # = 507 * 6 = 3042
    
    print(f"z = (x² + y²)³ at x={x}, y={y}")
    print(f"∂z/∂x = {dz_dx}")
    print(f"∂z/∂y = {dz_dy}")
    
    return dz_dx, dz_dy

chain_rule_example()
```

---

## 3. Probability Foundations

### 3.1 Probability Distributions

```python
class Gaussian:
    """
    Gaussian (Normal) distribution.
    
    PDF: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
    """
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
        self.variance = std ** 2
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function."""
        coef = 1 / np.sqrt(2 * np.pi * self.variance)
        exp_term = np.exp(-((x - self.mean) ** 2) / (2 * self.variance))
        return coef * exp_term
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Generate samples using Box-Muller transform."""
        u1 = np.random.uniform(0, 1, n)
        u2 = np.random.uniform(0, 1, n)
        
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return self.mean + self.std * z


# Test
dist = Gaussian(mean=5, std=2)
samples = dist.sample(1000)
print(f"Sample mean: {np.mean(samples):.2f} (expected: 5)")
print(f"Sample std: {np.std(samples):.2f} (expected: 2)")
```

### 3.2 Bayes Theorem

```python
def bayes_theorem(prior: float, likelihood: float, evidence: float) -> float:
    """
    Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
    
    Args:
        prior: P(A) - prior probability
        likelihood: P(B|A) - probability of evidence given hypothesis
        evidence: P(B) - total probability of evidence
    
    Returns:
        posterior: P(A|B) - updated probability after evidence
    """
    return (likelihood * prior) / evidence


# Example: Medical test
# P(disease) = 0.01 (1% of population has disease)
# P(positive|disease) = 0.99 (99% sensitivity)
# P(positive|no disease) = 0.05 (5% false positive)

prior_disease = 0.01
sensitivity = 0.99
false_positive = 0.05

# P(positive) = P(positive|disease)*P(disease) + P(positive|no disease)*P(no disease)
p_positive = sensitivity * prior_disease + false_positive * (1 - prior_disease)

posterior = bayes_theorem(prior_disease, sensitivity, p_positive)

print("Medical Test Example:")
print(f"Prior P(disease) = {prior_disease}")
print(f"P(positive|disease) = {sensitivity}")
print(f"P(positive|no disease) = {false_positive}")
print(f"\nPosterior P(disease|positive) = {posterior:.4f}")
print(f"Even with positive test, only {posterior*100:.1f}% chance of disease!")
```

---

## 4. Exercises

### Exercise 1: Implement PCA from Scratch
```python
def pca_scratch(X: Matrix, n_components: int = 2) -> Tuple[Matrix, Vector]:
    """
    Implement Principal Component Analysis.
    
    Steps:
    1. Center the data (subtract mean)
    2. Compute covariance matrix
    3. Find eigenvalues and eigenvectors
    4. Project data onto top eigenvectors
    
    Returns:
        transformed_data, explained_variance_ratio
    """
    # YOUR CODE HERE
    # 1. Center data
    X_centered = X - np.mean(X, axis=0)
    
    # 2. Covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    
    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. Project
    components = eigenvectors[:, :n_components]
    transformed = X_centered @ components
    
    # Explained variance
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return transformed, explained_variance


# Test
np.random.seed(42)
X = np.random.randn(100, 5)
X_pca, var_ratio = pca_scratch(X, n_components=2)
print(f"Transformed shape: {X_pca.shape}")
print(f"Explained variance: {var_ratio}")
```

### Exercise 2: Gradient Descent Implementation
```python
def gradient_descent(
    f, grad_f, x0: Vector, 
    learning_rate: float = 0.01,
    max_iters: int = 1000
) -> Tuple[Vector, List[float]]:
    """
    Implement vanilla gradient descent.
    
    Returns:
        optimal_x, loss_history
    """
    x = x0.copy()
    history = []
    
    for _ in range(max_iters):
        loss = f(x)
        history.append(loss)
        x = x - learning_rate * grad_f(x)
    
    return x, history


# Test on quadratic: f(x) = (x-3)² + (y-4)²
def f(x): return (x[0] - 3)**2 + (x[1] - 4)**2
def grad_f(x): return np.array([2*(x[0] - 3), 2*(x[1] - 4)])

x_opt, history = gradient_descent(f, grad_f, np.array([0.0, 0.0]))
print(f"Optimal x: {x_opt}")  # Should be close to [3, 4]
print(f"Final loss: {history[-1]:.6f}")
```

---

## Summary

| Topic | Key Concept | ML Application |
|-------|-------------|----------------|
| Dot Product | Similarity measure | Attention, embeddings |
| Matrix Multiply | Linear transform | Neural network layers |
| Eigenvalues | Principal directions | PCA, spectral methods |
| Gradient | Direction of steepest ascent | Optimization |
| Chain Rule | Composite derivatives | Backpropagation |
| Bayes Theorem | Updating beliefs | Bayesian inference |

---

## Next Week Preview
Week 02 will cover:
- Neural Networks from Scratch
- Backpropagation Algorithm
- Activation Functions
