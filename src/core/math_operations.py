"""
Mathematical Operations Module
==============================
Linear Algebra, Matrix Calculus, and Decomposition algorithms
implemented from scratch following the White-Box Approach.

Mathematical Foundations:
- Vector spaces and operations
- Matrix decomposition (SVD, Eigendecomposition)
- Numerical stability considerations

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Tuple, Optional, Union

# Type aliases
Vector = Union[List[float], np.ndarray]
Matrix = Union[List[List[float]], np.ndarray]


# ============================================================
# VECTOR OPERATIONS
# ============================================================

def dot_product(v1: Vector, v2: Vector) -> float:
    """
    Compute dot product of two vectors.
    
    Mathematical Definition:
        a · b = Σ(aᵢ × bᵢ) = ||a|| ||b|| cos(θ)
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Scalar dot product
    
    Example:
        >>> dot_product([1, 2, 3], [4, 5, 6])
        32
    """
    v1, v2 = np.asarray(v1), np.asarray(v2)
    return float(np.sum(v1 * v2))


def magnitude(v: Vector) -> float:
    """
    Compute L2 norm (Euclidean magnitude) of a vector.
    
    Mathematical Definition:
        ||v|| = √(Σvᵢ²)
    
    Args:
        v: Input vector
    
    Returns:
        Scalar magnitude
    """
    v = np.asarray(v)
    return float(np.sqrt(np.sum(v ** 2)))


def normalize(v: Vector) -> np.ndarray:
    """
    Normalize vector to unit length.
    
    Mathematical Definition:
        v̂ = v / ||v||
    
    Args:
        v: Input vector
    
    Returns:
        Unit vector in same direction
    """
    v = np.asarray(v)
    mag = magnitude(v)
    if mag == 0:
        return np.zeros_like(v)
    return v / mag


def cosine_similarity(v1: Vector, v2: Vector) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Mathematical Definition:
        cos(θ) = (a · b) / (||a|| × ||b||)
    
    Range: [-1, 1]
        1: Same direction
        0: Orthogonal
       -1: Opposite direction
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Cosine similarity score
    """
    v1, v2 = np.asarray(v1), np.asarray(v2)
    dot = dot_product(v1, v2)
    mag1, mag2 = magnitude(v1), magnitude(v2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot / (mag1 * mag2)


def euclidean_distance(v1: Vector, v2: Vector) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Mathematical Definition:
        d(a, b) = √(Σ(aᵢ - bᵢ)²)
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Euclidean distance
    """
    v1, v2 = np.asarray(v1), np.asarray(v2)
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def manhattan_distance(v1: Vector, v2: Vector) -> float:
    """
    Compute Manhattan (L1) distance between two vectors.
    
    Mathematical Definition:
        d(a, b) = Σ|aᵢ - bᵢ|
    
    Better for high-dimensional sparse data.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Manhattan distance
    """
    v1, v2 = np.asarray(v1), np.asarray(v2)
    return float(np.sum(np.abs(v1 - v2)))


# ============================================================
# MATRIX OPERATIONS
# ============================================================

def matrix_multiply(A: Matrix, B: Matrix) -> np.ndarray:
    """
    Matrix multiplication from scratch (no np.matmul).
    
    Mathematical Definition:
        C[i,j] = Σₖ A[i,k] × B[k,j]
    
    Complexity: O(n³) for square matrices
    
    Args:
        A: Matrix of shape (m, n)
        B: Matrix of shape (n, p)
    
    Returns:
        Result matrix of shape (m, p)
    """
    A, B = np.asarray(A), np.asarray(B)
    
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    
    m, n = A.shape
    n2, p = B.shape
    
    assert n == n2, f"Incompatible shapes: {A.shape} @ {B.shape}"
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C


def transpose(A: Matrix) -> np.ndarray:
    """
    Matrix transpose.
    
    Mathematical Definition:
        B[i,j] = A[j,i]
    
    Args:
        A: Input matrix
    
    Returns:
        Transposed matrix
    """
    A = np.asarray(A)
    m, n = A.shape
    B = np.zeros((n, m))
    
    for i in range(m):
        for j in range(n):
            B[j, i] = A[i, j]
    
    return B


def identity_matrix(n: int) -> np.ndarray:
    """
    Create n×n identity matrix.
    
    Args:
        n: Matrix dimension
    
    Returns:
        Identity matrix
    """
    I = np.zeros((n, n))
    for i in range(n):
        I[i, i] = 1.0
    return I


def trace(A: Matrix) -> float:
    """
    Compute trace of a square matrix (sum of diagonal elements).
    
    Mathematical Definition:
        tr(A) = Σ A[i,i]
    
    Args:
        A: Square matrix
    
    Returns:
        Trace value
    """
    A = np.asarray(A)
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    return float(sum(A[i, i] for i in range(A.shape[0])))


def frobenius_norm(A: Matrix) -> float:
    """
    Compute Frobenius norm of a matrix.
    
    Mathematical Definition:
        ||A||_F = √(Σᵢⱼ A[i,j]²)
    
    Args:
        A: Input matrix
    
    Returns:
        Frobenius norm
    """
    A = np.asarray(A)
    return float(np.sqrt(np.sum(A ** 2)))


# ============================================================
# MATRIX DECOMPOSITION
# ============================================================

def power_iteration(A: Matrix, num_iterations: int = 100, 
                    tolerance: float = 1e-10) -> Tuple[float, np.ndarray]:
    """
    Power iteration method to find dominant eigenvalue/eigenvector.
    
    Algorithm:
        1. Start with random vector b
        2. Repeat: b = Ab / ||Ab||
        3. Eigenvalue: λ = bᵀAb
    
    Args:
        A: Square matrix
        num_iterations: Maximum iterations
        tolerance: Convergence threshold
    
    Returns:
        Tuple of (eigenvalue, eigenvector)
    """
    A = np.asarray(A)
    n = A.shape[0]
    
    # Random initial vector
    b = np.random.rand(n)
    b = b / magnitude(b)
    
    for _ in range(num_iterations):
        # Multiply by matrix
        Ab = A @ b
        
        # Normalize
        b_new = Ab / magnitude(Ab)
        
        # Check convergence
        if magnitude(b_new - b) < tolerance:
            break
        
        b = b_new
    
    # Rayleigh quotient for eigenvalue
    eigenvalue = float(b @ A @ b)
    
    return eigenvalue, b


def gram_schmidt(vectors: Matrix) -> np.ndarray:
    """
    Gram-Schmidt orthogonalization.
    
    Produces orthonormal basis from input vectors.
    
    Args:
        vectors: Matrix where rows are vectors
    
    Returns:
        Orthonormal basis vectors
    """
    V = np.asarray(vectors, dtype=float)
    n = V.shape[0]
    
    U = np.zeros_like(V)
    
    for i in range(n):
        U[i] = V[i].copy()
        
        # Subtract projections onto previous vectors
        for j in range(i):
            proj = dot_product(V[i], U[j]) * U[j]
            U[i] = U[i] - proj
        
        # Normalize
        norm = magnitude(U[i])
        if norm > 1e-10:
            U[i] = U[i] / norm
    
    return U


def qr_decomposition(A: Matrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    QR decomposition using Gram-Schmidt.
    
    Decomposes A = QR where:
        Q: Orthogonal matrix
        R: Upper triangular matrix
    
    Args:
        A: Input matrix (m × n)
    
    Returns:
        Tuple of (Q, R)
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        
        for i in range(j):
            R[i, j] = float(Q[:, i] @ A[:, j])
            v = v - R[i, j] * Q[:, i]
        
        R[j, j] = magnitude(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
    
    return Q, R


def eigendecomposition(A: Matrix, num_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition using QR algorithm.
    
    Finds eigenvalues and eigenvectors: A = VΛV⁻¹
    
    Args:
        A: Square symmetric matrix
        num_iterations: QR iterations
    
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    
    # Accumulate eigenvectors
    V = identity_matrix(n)
    Ak = A.copy()
    
    for _ in range(num_iterations):
        Q, R = qr_decomposition(Ak)
        Ak = R @ Q
        V = V @ Q
    
    # Eigenvalues on diagonal
    eigenvalues = np.diag(Ak)
    
    return eigenvalues, V


def svd_simple(A: Matrix, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplified SVD using eigendecomposition.
    
    Decomposes A = UΣVᵀ
    
    For recommendation systems: Low-rank approximation
    
    Args:
        A: Input matrix (m × n)
        k: Number of singular values to keep (optional)
    
    Returns:
        Tuple of (U, Sigma, Vt)
    
    Note: For production, use np.linalg.svd which uses LAPACK.
    """
    A = np.asarray(A, dtype=float)
    
    # Use numpy SVD for numerical stability
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    if k is not None:
        U = U[:, :k]
        s = s[:k]
        Vt = Vt[:k, :]
    
    return U, np.diag(s), Vt


def low_rank_approximation(A: Matrix, k: int) -> np.ndarray:
    """
    Low-rank matrix approximation using truncated SVD.
    
    Used in:
        - Recommendation systems (Netflix Prize)
        - Dimensionality reduction
        - Noise reduction
    
    Args:
        A: Input matrix
        k: Rank of approximation
    
    Returns:
        Approximated matrix of rank k
    """
    U, S, Vt = svd_simple(A, k)
    return U @ S @ Vt


# ============================================================
# COVARIANCE AND CORRELATION
# ============================================================

def covariance_matrix(X: Matrix) -> np.ndarray:
    """
    Compute covariance matrix.
    
    Mathematical Definition:
        Cov(X) = (1/n) × (X - μ)ᵀ(X - μ)
    
    Used in PCA for finding principal components.
    
    Args:
        X: Data matrix (n_samples × n_features)
    
    Returns:
        Covariance matrix (n_features × n_features)
    """
    X = np.asarray(X)
    n = X.shape[0]
    
    # Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # Covariance
    return (X_centered.T @ X_centered) / n


def correlation_matrix(X: Matrix) -> np.ndarray:
    """
    Compute Pearson correlation matrix.
    
    Args:
        X: Data matrix (n_samples × n_features)
    
    Returns:
        Correlation matrix (n_features × n_features)
    """
    X = np.asarray(X)
    
    # Standardize
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    X_standardized = (X - mean) / std
    
    n = X.shape[0]
    return (X_standardized.T @ X_standardized) / n


# ============================================================
# PCA (Principal Component Analysis)
# ============================================================

class PCA:
    """
    Principal Component Analysis from scratch.
    
    Mathematical Foundation:
        1. Center data: X_c = X - μ
        2. Covariance: Σ = (1/n) X_cᵀ X_c
        3. Eigendecomposition: Σ = VΛVᵀ
        4. Project: X_new = X_c × V[:, :k]
    
    Example:
        >>> pca = PCA(n_components=2)
        >>> X_reduced = pca.fit_transform(X)
    """
    
    def __init__(self, n_components: int):
        """
        Args:
            n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X: Matrix) -> 'PCA':
        """
        Fit PCA model.
        
        Args:
            X: Data matrix (n_samples × n_features)
        
        Returns:
            Self for chaining
        """
        X = np.asarray(X)
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov = covariance_matrix(X)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store top k components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X: Matrix) -> np.ndarray:
        """
        Apply dimensionality reduction.
        
        Args:
            X: Data matrix
        
        Returns:
            Transformed data
        """
        X = np.asarray(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: Matrix) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: Matrix) -> np.ndarray:
        """
        Reconstruct data from reduced dimensions.
        
        Args:
            X_transformed: Reduced data
        
        Returns:
            Reconstructed data (with information loss)
        """
        X_transformed = np.asarray(X_transformed)
        return X_transformed @ self.components_ + self.mean_


# ============================================================
# NUMERICAL UTILITIES
# ============================================================

def softmax(x: Vector, axis: int = -1) -> np.ndarray:
    """
    Softmax function (numerically stable).
    
    Mathematical Definition:
        softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
    
    Used in:
        - Attention mechanisms (Transformers)
        - Multiclass classification output
    
    Args:
        x: Input logits
        axis: Axis to apply softmax
    
    Returns:
        Probability distribution
    """
    x = np.asarray(x)
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: Vector, axis: int = -1) -> np.ndarray:
    """
    Log-softmax (numerically stable).
    
    More stable than log(softmax(x)) for gradient computation.
    """
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return x - log_sum_exp


def sigmoid(x: Vector) -> np.ndarray:
    """
    Sigmoid activation function.
    
    Mathematical Definition:
        σ(x) = 1 / (1 + exp(-x))
    
    Properties:
        - Output in (0, 1)
        - σ'(x) = σ(x)(1 - σ(x))
    """
    x = np.asarray(x)
    # Numerically stable implementation
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def relu(x: Vector) -> np.ndarray:
    """
    ReLU activation function.
    
    Mathematical Definition:
        ReLU(x) = max(0, x)
    """
    return np.maximum(0, np.asarray(x))


def relu_derivative(x: Vector) -> np.ndarray:
    """Derivative of ReLU."""
    return np.where(np.asarray(x) > 0, 1.0, 0.0)


def tanh(x: Vector) -> np.ndarray:
    """Hyperbolic tangent activation."""
    return np.tanh(np.asarray(x))


def gelu(x: Vector) -> np.ndarray:
    """
    Gaussian Error Linear Unit.
    
    Used in BERT, GPT and modern Transformers.
    
    Approximation:
        GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    """
    x = np.asarray(x)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Vector operations
    'dot_product', 'magnitude', 'normalize', 'cosine_similarity',
    'euclidean_distance', 'manhattan_distance',
    # Matrix operations
    'matrix_multiply', 'transpose', 'identity_matrix', 'trace', 'frobenius_norm',
    # Decomposition
    'power_iteration', 'gram_schmidt', 'qr_decomposition', 
    'eigendecomposition', 'svd_simple', 'low_rank_approximation',
    # Covariance
    'covariance_matrix', 'correlation_matrix', 'PCA',
    # Activations
    'softmax', 'log_softmax', 'sigmoid', 'relu', 'relu_derivative', 'tanh', 'gelu',
]
