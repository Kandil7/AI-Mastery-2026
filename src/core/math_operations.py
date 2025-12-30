"""
Mathematical Operations Module

This module implements fundamental mathematical operations from scratch using NumPy,
focusing on linear algebra, calculus, and other mathematical foundations needed for AI.
"""

import numpy as np
from typing import Union, List, Tuple


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Perform matrix multiplication from scratch without using np.dot.
    
    Args:
        A: First matrix of shape (m, n)
        B: Second matrix of shape (n, p)
        
    Returns:
        Product matrix of shape (m, p)
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Cannot multiply matrices with shapes {A.shape} and {B.shape}")
    
    m, n = A.shape
    _, p = B.shape
    
    result = np.zeros((m, p))
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += A[i, k] * B[k, j]
    
    return result


def matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    """
    Compute matrix inverse using Gauss-Jordan elimination.
    
    Args:
        matrix: Square matrix to invert
        
    Returns:
        Inverted matrix
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square to compute inverse")
    
    n = matrix.shape[0]
    # Create augmented matrix [A|I]
    augmented = np.hstack([matrix.astype(float), np.eye(n)])
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = np.argmax(np.abs(augmented[i:, i])) + i
        augmented[[i, max_row]] = augmented[[max_row, i]]
        
        # Check for singular matrix
        if abs(augmented[i, i]) < 1e-10:
            raise ValueError("Matrix is singular and cannot be inverted")
        
        # Scale pivot row
        augmented[i] = augmented[i] / augmented[i, i]
        
        # Eliminate column
        for j in range(n):
            if i != j:
                augmented[j] = augmented[j] - augmented[j, i] * augmented[i]
    
    return augmented[:, n:]


def eigenvalues_eigenvectors(matrix: np.ndarray, max_iterations: int = 1000, tolerance: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors using the power iteration method for symmetric matrices.
    
    Args:
        matrix: Square matrix
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of eigenvalues and eigenvectors
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    
    n = matrix.shape[0]
    eigenvals = np.zeros(n)
    eigenvecs = np.zeros((n, n))
    
    A = matrix.copy().astype(float)
    
    for i in range(n):
        # Power iteration to find the largest eigenvalue/eigenvector
        x = np.random.rand(n)
        
        for _ in range(max_iterations):
            x_new = A @ x
            eigenval = x.T @ A @ x  # Rayleigh quotient
            x_new_norm = np.linalg.norm(x_new)
            
            if x_new_norm < 1e-12:
                break
                
            x_new = x_new / x_new_norm
            if np.allclose(x, x_new, rtol=tolerance):
                eigenval = x.T @ A @ x
                break
                
            x = x_new
        else:
            # If not converged, use the last computed values
            eigenval = x.T @ A @ x
        
        eigenvals[i] = eigenval
        eigenvecs[:, i] = x
        
        # Deflate matrix to find next eigenvalue
        A = A - eigenval * np.outer(x, x)
    
    return eigenvals, eigenvecs


def svd(matrix: np.ndarray, max_iterations: int = 1000, tolerance: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Singular Value Decomposition using the power method approach.
    
    Args:
        matrix: Input matrix to decompose
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of U, S, Vt matrices such that A = U @ S @ Vt
    """
    A = matrix.astype(float)
    m, n = A.shape
    
    # Compute A^T * A for right singular vectors
    ATA = A.T @ A
    
    # Compute eigenvalues and eigenvectors of A^T * A
    eigenvals, V = eigenvalues_eigenvectors(ATA)
    
    # Singular values are square roots of eigenvalues
    singular_vals = np.sqrt(np.abs(eigenvals))
    
    # Sort in descending order
    idx = np.argsort(singular_vals)[::-1]
    singular_vals = singular_vals[idx]
    V = V[:, idx]
    
    # Compute left singular vectors
    U = np.zeros((m, m))
    for i in range(min(m, n)):
        if singular_vals[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / singular_vals[i]
        else:
            # For zero singular values, use random orthogonal vector
            u = np.random.rand(m)
            u = u - U[:, :i] @ (U[:, :i].T @ u)  # Orthogonalize
            U[:, i] = u / np.linalg.norm(u)
    
    # Handle remaining columns of U if m > n
    if m > n:
        for i in range(n, m):
            u = np.random.rand(m)
            # Orthogonalize with all previous vectors
            for j in range(i):
                u = u - U[:, j] @ (U[:, j].T @ u)
            U[:, i] = u / np.linalg.norm(u)
    
    # Create diagonal matrix of singular values
    S = np.zeros((m, n))
    np.fill_diagonal(S, singular_vals)
    
    return U, S, V.T


def gradient_descent(
    func, 
    grad, 
    initial_params: np.ndarray, 
    learning_rate: float = 0.01, 
    max_iterations: int = 1000, 
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    Perform gradient descent optimization.
    
    Args:
        func: Function to minimize
        grad: Gradient function
        initial_params: Initial parameters
        learning_rate: Learning rate
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of optimized parameters and loss history
    """
    params = initial_params.astype(float)
    loss_history = []
    
    for i in range(max_iterations):
        loss = func(params)
        loss_history.append(loss)
        
        grad_val = grad(params)
        
        # Update parameters
        new_params = params - learning_rate * grad_val
        
        # Check for convergence
        if np.linalg.norm(new_params - params) < tolerance:
            params = new_params
            break
            
        params = new_params
    
    return params, loss_history


def numerical_derivative(func, x: Union[float, np.ndarray], h: float = 1e-7) -> Union[float, np.ndarray]:
    """
    Compute numerical derivative using central difference method.
    
    Args:
        func: Function to differentiate
        x: Point at which to compute derivative
        h: Small value for finite difference
        
    Returns:
        Derivative at point x
    """
    x = np.asarray(x, dtype=float)
    
    if x.ndim == 0:  # Scalar
        return (func(x + h) - func(x - h)) / (2 * h)
    else:  # Vector
        result = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            result[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return result