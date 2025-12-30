"""
Optimization Module

This module implements various optimization algorithms from scratch, including
gradient descent variants, second-order methods, and constrained optimization.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional
from src.core.math_operations import gradient_descent, numerical_derivative


class Optimizer:
    """Base class for optimization algorithms"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters based on gradients"""
        raise NotImplementedError


class GradientDescent(Optimizer):
    """Standard gradient descent optimizer"""
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return params - self.learning_rate * gradients


class Momentum(Optimizer):
    """Momentum-based optimizer"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        return params + self.velocity


class Adam(Optimizer):
    """Adam optimizer implementation"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0     # Time step
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Bias correction
        m_corrected = self.m / (1 - self.beta1 ** self.t)
        v_corrected = self.v / (1 - self.beta2 ** self.t)
        
        return params - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)


def minimize(
    func: Callable, 
    initial_params: np.ndarray, 
    method: str = 'adam',
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    Minimize a function using specified optimization method.
    
    Args:
        func: Function to minimize
        initial_params: Initial parameters
        method: Optimization method ('gd', 'momentum', 'adam')
        learning_rate: Learning rate
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of optimized parameters and loss history
    """
    params = initial_params.astype(float)
    loss_history = []
    
    # Create optimizer
    if method == 'gd':
        optimizer = GradientDescent(learning_rate)
    elif method == 'momentum':
        optimizer = Momentum(learning_rate)
    elif method == 'adam':
        optimizer = Adam(learning_rate)
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    prev_loss = float('inf')
    
    for i in range(max_iterations):
        loss = func(params)
        loss_history.append(loss)
        
        # Compute gradients using numerical differentiation
        gradients = numerical_derivative(func, params)
        
        # Update parameters
        params = optimizer.step(params, gradients)
        
        # Check for convergence
        if abs(prev_loss - loss) < tolerance:
            break
            
        prev_loss = loss
    
    return params, loss_history


def lagrange_multipliers(
    objective_func: Callable,
    constraint_funcs: List[Callable],
    initial_params: np.ndarray,
    initial_lambdas: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve constrained optimization using Lagrange multipliers.
    
    Args:
        objective_func: Function to optimize
        constraint_funcs: List of constraint functions (should equal 0)
        initial_params: Initial parameters
        initial_lambdas: Initial lambda values (Lagrange multipliers)
        
    Returns:
        Tuple of optimized parameters and lambda values
    """
    n_params = len(initial_params)
    n_constraints = len(constraint_funcs)
    
    if initial_lambdas is None:
        initial_lambdas = np.zeros(n_constraints)
    
    # Combine parameters and lambdas
    combined_params = np.concatenate([initial_params, initial_lambdas])
    
    def lagrangian(params_lambdas):
        params = params_lambdas[:n_params]
        lambdas = params_lambdas[n_params:]
        
        # Objective function
        result = objective_func(params)
        
        # Add constraint terms
        for i, constraint_func in enumerate(constraint_funcs):
            result += lambdas[i] * constraint_func(params)
        
        return result
    
    # Minimize the Lagrangian
    result_params, _ = minimize(lagrangian, combined_params, method='adam')
    
    # Extract optimized parameters and lambdas
    optimized_params = result_params[:n_params]
    optimized_lambdas = result_params[n_params:]
    
    return optimized_params, optimized_lambdas


def newton_raphson(
    func: Callable,
    grad_func: Callable,
    hess_func: Callable,
    initial_params: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, List[float]]:
    """
    Newton-Raphson method for optimization (finding minima by setting gradient to 0).
    
    Args:
        func: Function to minimize
        grad_func: Gradient function
        hess_func: Hessian function
        initial_params: Initial parameters
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
        
        grad = grad_func(params)
        hess = hess_func(params)
        
        # Update params: params = params - H^(-1) * grad
        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            # If Hessian is singular, add regularization
            hess_inv = np.linalg.inv(hess + 1e-6 * np.eye(len(params)))
        
        update = hess_inv @ grad
        params = params - update
        
        # Check for convergence
        if np.linalg.norm(update) < tolerance:
            break
    
    return params, loss_history


def line_search(
    func: Callable,
    initial_params: np.ndarray,
    direction: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> float:
    """
    Perform line search to find optimal step size in given direction.
    
    Args:
        func: Function to minimize
        initial_params: Starting parameters
        direction: Search direction
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Optimal step size
    """
    # Use backtracking line search
    alpha = 1.0  # Initial step size
    c1 = 1e-4    # Armijo condition parameter
    rho = 0.5    # Reduction factor
    
    initial_value = func(initial_params)
    initial_grad = numerical_derivative(func, initial_params)
    
    for i in range(max_iterations):
        new_params = initial_params + alpha * direction
        new_value = func(new_params)
        
        # Check Armijo condition
        if new_value <= initial_value + c1 * alpha * (initial_grad @ direction):
            break
        
        alpha *= rho
    
    return alpha