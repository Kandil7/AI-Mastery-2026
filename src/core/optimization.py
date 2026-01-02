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
    """
    Adam optimizer: Adaptive Moment Estimation.
    
    Combines momentum (first moment) with RMSprop (second moment).
    
    Update rule:
        m_t = β₁ m_{t-1} + (1 - β₁) g_t          (first moment)
        v_t = β₂ v_{t-1} + (1 - β₂) g_t²         (second moment)
        m̂_t = m_t / (1 - β₁^t)                   (bias correction)
        v̂_t = v_t / (1 - β₂^t)
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
    
    Industrial Use Case:
        OpenAI uses Adam variants for training GPT models. The bias
        correction is crucial for stable training in the first epochs.
    
    Interview Question:
        Q: Why does Adam have bias correction?
        A: m and v are initialized at 0, so early estimates are biased
           toward 0. Dividing by (1 - β^t) corrects this, especially
           important when β is close to 1 (e.g., β₂ = 0.999).
    """
    
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


class RMSprop(Optimizer):
    """
    RMSprop optimizer: Root Mean Square Propagation.
    
    Divides the learning rate by a running average of recent gradient magnitudes.
    
    Update rule:
        v_t = ρ v_{t-1} + (1 - ρ) g_t²
        θ_t = θ_{t-1} - α * g_t / (√v_t + ε)
    
    Industrial Use Case:
        DeepMind used RMSprop for training the original DQN (Atari games).
        It handles the non-stationary nature of RL well.
    
    Interview Question:
        Q: When would you prefer RMSprop over Adam?
        A: RMSprop is simpler and sometimes faster in recurrent networks.
           It lacks momentum, which can be helpful when momentum causes
           instability (e.g., some RL tasks).
    """
    
    def __init__(self, learning_rate: float = 0.01, rho: float = 0.9, 
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.v = None  # Running average of squared gradients
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(params)
        
        self.v = self.rho * self.v + (1 - self.rho) * (gradients ** 2)
        return params - self.learning_rate * gradients / (np.sqrt(self.v) + self.epsilon)


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer: Adaptive Gradient.
    
    Accumulates all past squared gradients, giving smaller updates
    to frequently updated parameters.
    
    Update rule:
        v_t = v_{t-1} + g_t²
        θ_t = θ_{t-1} - α * g_t / (√v_t + ε)
    
    Industrial Use Case:
        Google uses AdaGrad for training large-scale sparse models
        like ad CTR prediction where different features have very
        different frequencies.
    
    Interview Question:
        Q: What's the main limitation of AdaGrad?
        A: The accumulated squared gradients grow monotonically, causing
           the learning rate to shrink to near-zero. This is why RMSprop
           and Adam use exponential moving averages instead.
    """
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.v = None  # Sum of squared gradients
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(params)
        
        self.v += gradients ** 2
        return params - self.learning_rate * gradients / (np.sqrt(self.v) + self.epsilon)


class NAdam(Optimizer):
    """
    NAdam optimizer: Nesterov-accelerated Adaptive Moment Estimation.
    
    Combines Adam with Nesterov momentum for potentially faster convergence.
    
    Industrial Use Case:
        Meta AI research uses NAdam variants for DLRM (Deep Learning
        Recommendation Models) training on massive-scale ad systems.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
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
        
        # Nesterov momentum: look ahead with current gradient
        m_nesterov = self.beta1 * m_corrected + (1 - self.beta1) * gradients / (1 - self.beta1 ** self.t)
        
        return params - self.learning_rate * m_nesterov / (np.sqrt(v_corrected) + self.epsilon)


# =============================================================================
# LEARNING RATE SCHEDULERS
# =============================================================================

class LearningRateScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.step_count = 0
    
    def step(self) -> float:
        """Advance one step and return new learning rate."""
        self.step_count += 1
        self.current_lr = self._compute_lr()
        return self.current_lr
    
    def _compute_lr(self) -> float:
        raise NotImplementedError


class StepDecay(LearningRateScheduler):
    """
    Step decay: reduce LR by factor every N steps.
    
    lr = initial_lr * factor^(floor(step / step_size))
    """
    
    def __init__(self, initial_lr: float, step_size: int = 10, factor: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.factor = factor
    
    def _compute_lr(self) -> float:
        return self.initial_lr * (self.factor ** (self.step_count // self.step_size))


class ExponentialDecay(LearningRateScheduler):
    """
    Exponential decay: smoothly reduce LR.
    
    lr = initial_lr * exp(-decay_rate * step)
    """
    
    def __init__(self, initial_lr: float, decay_rate: float = 0.01):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
    
    def _compute_lr(self) -> float:
        return self.initial_lr * np.exp(-self.decay_rate * self.step_count)


class CosineAnnealing(LearningRateScheduler):
    """
    Cosine annealing: smooth decay following cosine curve.
    
    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * step / T))
    
    Industrial Use Case:
        Google Brain's EfficientNet uses cosine annealing for
        stable training to high accuracy.
    """
    
    def __init__(self, initial_lr: float, T_max: int, min_lr: float = 0.0):
        super().__init__(initial_lr)
        self.T_max = T_max
        self.min_lr = min_lr
    
    def _compute_lr(self) -> float:
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1 + np.cos(np.pi * self.step_count / self.T_max)
        )


class WarmupScheduler(LearningRateScheduler):
    """
    Warmup + decay: linear warmup followed by decay.
    
    Industrial Use Case:
        BERT and GPT models use warmup to stabilize early training
        when gradients are noisy.
    
    Interview Question:
        Q: Why is warmup important for Transformers?
        A: Early gradients are noisy due to random initialization.
           Large learning rates can cause divergence. Warmup allows
           the model to find a stable region before increasing LR.
    """
    
    def __init__(self, initial_lr: float, warmup_steps: int, 
                 total_steps: int, min_lr: float = 0.0):
        super().__init__(initial_lr)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def _compute_lr(self) -> float:
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            return self.initial_lr * self.step_count / self.warmup_steps
        else:
            # Linear decay after warmup
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + (self.initial_lr - self.min_lr) * (1 - progress)


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