"""
Interview Coding Challenges - Optimization Problems
====================================================
ML optimization problems commonly asked in interviews.

Each challenge includes:
- Problem statement
- Mathematical background
- Implementation
- Complexity analysis

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# CHALLENGE 1: Implement Gradient Descent
# ============================================================

def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    learning_rate: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    Problem: Implement vanilla gradient descent.
    
    Update rule: x_{t+1} = x_t - lr * ∇f(x_t)
    
    Args:
        f: Objective function
        grad_f: Gradient function
        x0: Initial point
        learning_rate: Step size
        max_iters: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        Tuple of (optimal_x, loss_history)
    
    Time: O(max_iters * cost(grad_f))
    """
    x = x0.copy()
    history = [f(x)]
    
    for i in range(max_iters):
        grad = grad_f(x)
        x = x - learning_rate * grad
        
        loss = f(x)
        history.append(loss)
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
    
    return x, history


# ============================================================
# CHALLENGE 2: Implement Adam Optimizer
# ============================================================

@dataclass
class AdamState:
    """Adam optimizer state."""
    m: np.ndarray  # First moment
    v: np.ndarray  # Second moment
    t: int = 0     # Time step


def adam_step(
    grad: np.ndarray,
    state: AdamState,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8
) -> Tuple[np.ndarray, AdamState]:
    """
    Problem: Implement one step of Adam optimizer.
    
    Adam combines momentum and RMSprop:
    - m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    - v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    - Bias correction: m̂_t = m_t / (1 - β₁^t)
    - Update: x_t = x_{t-1} - lr * m̂_t / (√v̂_t + ε)
    
    Returns:
        Tuple of (update_vector, new_state)
    """
    state.t += 1
    
    # Update biased first moment estimate
    state.m = beta1 * state.m + (1 - beta1) * grad
    
    # Update biased second raw moment estimate
    state.v = beta2 * state.v + (1 - beta2) * (grad ** 2)
    
    # Compute bias-corrected first moment estimate
    m_hat = state.m / (1 - beta1 ** state.t)
    
    # Compute bias-corrected second raw moment estimate
    v_hat = state.v / (1 - beta2 ** state.t)
    
    # Compute update
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return update, state


# ============================================================
# CHALLENGE 3: Line Search (Backtracking)
# ============================================================

def backtracking_line_search(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    grad: np.ndarray,
    direction: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4
) -> float:
    """
    Problem: Implement Armijo backtracking line search.
    
    Find step size α such that:
    f(x + α*d) ≤ f(x) + c * α * ∇f(x)ᵀd
    
    This is the Armijo condition (sufficient decrease).
    
    Returns:
        Optimal step size
    """
    fx = f(x)
    slope = np.dot(grad, direction)
    
    while f(x + alpha * direction) > fx + c * alpha * slope:
        alpha *= beta
        
        if alpha < 1e-10:
            break
    
    return alpha


# ============================================================
# CHALLENGE 4: Newton's Method
# ============================================================

def newtons_method(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    Problem: Implement Newton's method for optimization.
    
    Update rule: x_{t+1} = x_t - H⁻¹ * ∇f(x_t)
    
    Pros: Quadratic convergence near optimum
    Cons: Requires Hessian, expensive for high dimensions
    
    Time: O(max_iters * n³) where n³ is for matrix inversion
    """
    x = x0.copy()
    history = [f(x)]
    
    for i in range(max_iters):
        grad = grad_f(x)
        hess = hess_f(x)
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
        
        # Newton direction: solve H * d = -grad
        try:
            direction = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            # Hessian singular, fall back to gradient descent
            direction = -grad
        
        # Line search for step size
        alpha = backtracking_line_search(f, x, grad, direction)
        
        x = x + alpha * direction
        history.append(f(x))
    
    return x, history


# ============================================================
# CHALLENGE 5: Coordinate Descent
# ============================================================

def coordinate_descent(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    learning_rate: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    Problem: Implement coordinate descent.
    
    Optimize one variable at a time, holding others fixed.
    Useful for separable objectives like LASSO.
    
    Time: O(max_iters * n * cost(f))
    """
    x = x0.copy()
    n = len(x)
    history = [f(x)]
    
    for iteration in range(max_iters):
        x_old = x.copy()
        
        for i in range(n):
            # Compute partial derivative numerically
            h = 1e-7
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            
            partial_grad = (f(x_plus) - f(x_minus)) / (2 * h)
            
            # Update coordinate
            x[i] -= learning_rate * partial_grad
        
        history.append(f(x))
        
        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            break
    
    return x, history


# ============================================================
# CHALLENGE 6: Proximal Gradient Descent (for L1)
# ============================================================

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft thresholding operator (proximal operator for L1).
    
    prox_{λ||·||₁}(x) = sign(x) * max(|x| - λ, 0)
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def proximal_gradient_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lambda_reg: float = 0.1,
    learning_rate: float = 0.01,
    max_iters: int = 1000
) -> Tuple[np.ndarray, List[float]]:
    """
    Problem: Implement proximal gradient descent for LASSO.
    
    Minimizes: f(x) + λ||x||₁
    
    Update:
    1. Gradient step: z = x - lr * ∇f(x)
    2. Proximal step: x = soft_threshold(z, lr * λ)
    
    Used for sparse optimization (L1 regularization).
    """
    x = x0.copy()
    history = [f(x) + lambda_reg * np.sum(np.abs(x))]
    
    for i in range(max_iters):
        # Gradient step
        grad = grad_f(x)
        z = x - learning_rate * grad
        
        # Proximal step (soft thresholding)
        x = soft_threshold(z, learning_rate * lambda_reg)
        
        # Total objective
        obj = f(x) + lambda_reg * np.sum(np.abs(x))
        history.append(obj)
    
    return x, history


# ============================================================
# CHALLENGE 7: L-BFGS (Limited Memory BFGS)
# ============================================================

class LBFGS:
    """
    L-BFGS optimizer - approximates Hessian inverse using limited memory.
    
    Stores m most recent (s, y) pairs where:
    - s_k = x_{k+1} - x_k
    - y_k = ∇f_{k+1} - ∇f_k
    """
    
    def __init__(self, m: int = 10):
        self.m = m
        self.s_history: List[np.ndarray] = []
        self.y_history: List[np.ndarray] = []
    
    def compute_direction(self, grad: np.ndarray) -> np.ndarray:
        """
        Two-loop recursion to compute search direction.
        
        Returns: H_k * grad (approximate Newton direction)
        """
        q = grad.copy()
        n = len(self.s_history)
        
        if n == 0:
            return -grad
        
        alphas = []
        
        # First loop (backward)
        for i in range(n - 1, -1, -1):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
            alpha_i = rho_i * np.dot(s_i, q)
            alphas.append(alpha_i)
            q = q - alpha_i * y_i
        
        alphas = alphas[::-1]
        
        # Initial Hessian approximation
        s_last = self.s_history[-1]
        y_last = self.y_history[-1]
        gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-10)
        r = gamma * q
        
        # Second loop (forward)
        for i in range(n):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
            beta_i = rho_i * np.dot(y_i, r)
            r = r + s_i * (alphas[i] - beta_i)
        
        return -r
    
    def update(self, s: np.ndarray, y: np.ndarray):
        """Store new (s, y) pair."""
        if len(self.s_history) >= self.m:
            self.s_history.pop(0)
            self.y_history.pop(0)
        
        self.s_history.append(s)
        self.y_history.append(y)


# ============================================================
# CHALLENGE 8: Stochastic Gradient Descent with Momentum
# ============================================================

def sgd_momentum(
    compute_grad: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 32,
    epochs: int = 100
) -> Tuple[np.ndarray, List[float]]:
    """
    Problem: Implement SGD with momentum.
    
    Momentum accelerates SGD in relevant directions:
    v_t = γ * v_{t-1} + lr * ∇f(x_t)
    x_{t+1} = x_t - v_t
    
    This dampens oscillations and accelerates convergence.
    """
    x = x0.copy()
    velocity = np.zeros_like(x)
    n_samples = len(X)
    history = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            # Compute gradient on mini-batch
            grad = compute_grad(x, X_batch, y_batch)
            
            # Update velocity
            velocity = momentum * velocity + learning_rate * grad
            
            # Update parameters
            x = x - velocity
        
        # Track loss (simplified)
        history.append(np.linalg.norm(velocity))
    
    return x, history


# ============================================================
# TESTS
# ============================================================

def run_tests():
    """Run optimization challenge tests."""
    print("Running Optimization Challenges Tests...")
    
    # Test 1: Gradient descent on quadratic
    def f(x): return x[0]**2 + x[1]**2
    def grad_f(x): return np.array([2*x[0], 2*x[1]])
    
    x_opt, history = gradient_descent(f, grad_f, np.array([5.0, 5.0]))
    assert np.allclose(x_opt, [0, 0], atol=0.1), "GD failed"
    print("✓ Gradient descent")
    
    # Test 2: Adam
    state = AdamState(m=np.zeros(2), v=np.zeros(2))
    update, state = adam_step(np.array([1.0, 2.0]), state)
    assert update is not None, "Adam failed"
    print("✓ Adam optimizer")
    
    # Test 3: Newton's method
    def hess_f(x): return np.array([[2.0, 0], [0, 2.0]])
    x_opt, history = newtons_method(f, grad_f, hess_f, np.array([5.0, 5.0]))
    assert np.allclose(x_opt, [0, 0], atol=0.01), "Newton failed"
    print("✓ Newton's method")
    
    # Test 4: Soft thresholding
    result = soft_threshold(np.array([0.5, -0.5, 2.0]), 1.0)
    expected = np.array([0, 0, 1.0])
    assert np.allclose(result, expected), "Soft threshold failed"
    print("✓ Soft thresholding")
    
    # Test 5: L-BFGS direction
    lbfgs = LBFGS(m=5)
    lbfgs.update(np.array([1.0, 0.0]), np.array([0.1, 0.0]))
    direction = lbfgs.compute_direction(np.array([1.0, 1.0]))
    assert direction is not None, "L-BFGS failed"
    print("✓ L-BFGS")
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    run_tests()
