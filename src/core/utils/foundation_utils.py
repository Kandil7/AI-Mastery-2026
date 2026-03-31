"""
Foundation Utilities - Week 0
Core mathematical and statistical functions built from scratch.
"""

import numpy as np
from typing import List, Callable


# ============================================================
# LINEAR ALGEBRA
# ============================================================

def dot(v1: List[float], v2: List[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def magnitude(v: List[float]) -> float:
    """Compute the magnitude (L2 norm) of a vector."""
    return np.sqrt(sum(x**2 for x in v))


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    return dot(v1, v2) / (magnitude(v1) * magnitude(v2))


def normalize(v: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    mag = magnitude(v)
    return [x / mag for x in v]


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Manual matrix multiplication: C = A @ B"""
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, f"Incompatible dimensions: {n} vs {n2}"
    
    C = [[0.0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


# ============================================================
# STATISTICS
# ============================================================

def mean(arr: List[float]) -> float:
    """Compute arithmetic mean."""
    return sum(arr) / len(arr)


def variance(arr: List[float]) -> float:
    """Compute population variance."""
    mu = mean(arr)
    return sum((x - mu)**2 for x in arr) / len(arr)


def std(arr: List[float]) -> float:
    """Compute population standard deviation."""
    return np.sqrt(variance(arr))


def covariance(x: List[float], y: List[float]) -> float:
    """Compute covariance between two variables."""
    mu_x, mu_y = mean(x), mean(y)
    return sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)) / len(x)


def correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    return covariance(x, y) / (std(x) * std(y))


# ============================================================
# OPTIMIZATION
# ============================================================

def gradient_descent(
    f_prime: Callable[[float], float],
    lr: float = 0.01,
    start: float = 0.0,
    steps: int = 100
) -> float:
    """
    Simple 1D gradient descent.
    
    Args:
        f_prime: Derivative of the function to minimize
        lr: Learning rate
        start: Starting point
        steps: Number of iterations
    
    Returns:
        Final optimized value
    """
    x = start
    for _ in range(steps):
        x -= lr * f_prime(x)
    return x


def gradient_descent_with_history(
    f_prime: Callable[[float], float],
    lr: float = 0.01,
    start: float = 0.0,
    steps: int = 100
) -> tuple:
    """Gradient descent with history for visualization."""
    x = start
    history = [x]
    for _ in range(steps):
        x -= lr * f_prime(x)
        history.append(x)
    return x, history


# ============================================================
# LINEAR REGRESSION
# ============================================================

def linear_regression_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form solution for linear regression.
    w = (X^T X)^{-1} X^T y
    """
    X = np.array(X)
    y = np.array(y)
    return np.linalg.inv(X.T @ X) @ X.T @ y


def linear_regression_gradient(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    epochs: int = 100
) -> tuple:
    """
    Linear regression using gradient descent.
    
    Returns:
        weights, loss_history
    """
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape
    
    # Initialize weights
    w = np.zeros(n_features)
    loss_history = []
    
    for _ in range(epochs):
        # Predictions
        y_pred = X @ w
        
        # MSE Loss
        loss = np.mean((y_pred - y)**2)
        loss_history.append(loss)
        
        # Gradient: dL/dw = (2/n) * X^T @ (y_pred - y)
        gradient = (2 / n_samples) * X.T @ (y_pred - y)
        
        # Update weights
        w -= lr * gradient
    
    return w, loss_history


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error loss."""
    return np.mean((y_true - y_pred)**2)


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
    """Binary Cross-Entropy loss."""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()
