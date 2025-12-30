"""
Optimization Algorithms Module
==============================
Gradient-based optimization algorithms implemented from scratch.

Includes:
- Gradient Descent variants (Batch, SGD, Mini-batch)
- Adaptive optimizers (Adam, RMSprop, AdaGrad)
- Learning rate schedulers
- Regularization techniques

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

# Type aliases
GradientFunc = Callable[[np.ndarray], np.ndarray]
LossFunc = Callable[[np.ndarray], float]


# ============================================================
# BASE OPTIMIZER CLASS
# ============================================================

class Optimizer(ABC):
    """
    Abstract base class for optimizers.
    
    All optimizers implement the update rule:
        θ_new = θ_old - step
    
    Where 'step' depends on the specific algorithm.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.iterations = 0
    
    @abstractmethod
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Perform one optimization step.
        
        Args:
            params: Current parameters
            grads: Gradients of loss w.r.t. parameters
        
        Returns:
            Updated parameters
        """
        pass
    
    def reset(self):
        """Reset optimizer state."""
        self.iterations = 0


# ============================================================
# GRADIENT DESCENT VARIANTS
# ============================================================

class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.
    
    Update Rule (with momentum):
        v = β × v + (1 - β) × ∇L
        θ = θ - α × v
    
    Where:
        α: Learning rate
        β: Momentum coefficient (0 = no momentum)
        v: Velocity (exponential moving average of gradients)
    
    Args:
        learning_rate: Step size
        momentum: Momentum coefficient (0-1)
        nesterov: Use Nesterov accelerated gradient
    
    Example:
        >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
        >>> params = optimizer.step(params, gradients)
    """
    
    def __init__(self, learning_rate: float = 0.01, 
                 momentum: float = 0.0,
                 nesterov: bool = False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = None
    
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # Update velocity
        self.velocity = self.momentum * self.velocity + grads
        
        if self.nesterov:
            # Nesterov: look ahead before computing gradient
            update = self.momentum * self.velocity + grads
        else:
            update = self.velocity
        
        self.iterations += 1
        return params - self.learning_rate * update
    
    def reset(self):
        super().reset()
        self.velocity = None


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Combines momentum with adaptive learning rates per parameter.
    
    Update Rules:
        m = β₁ × m + (1 - β₁) × g           # First moment (mean)
        v = β₂ × v + (1 - β₂) × g²          # Second moment (variance)
        m̂ = m / (1 - β₁ᵗ)                   # Bias correction
        v̂ = v / (1 - β₂ᵗ)                   # Bias correction
        θ = θ - α × m̂ / (√v̂ + ε)
    
    Hyperparameters:
        α (learning_rate): Step size, typically 0.001
        β₁ (beta1): Decay rate for first moment, typically 0.9
        β₂ (beta2): Decay rate for second moment, typically 0.999
        ε (epsilon): Numerical stability, typically 1e-8
    
    Example:
        >>> optimizer = Adam(learning_rate=0.001)
        >>> params = optimizer.step(params, gradients)
    """
    
    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
    
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.iterations += 1
        t = self.iterations
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** t)
        v_hat = self.v / (1 - self.beta2 ** t)
        
        # Update parameters
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        super().reset()
        self.m = None
        self.v = None


class AdamW(Adam):
    """
    Adam with decoupled weight decay (AdamW).
    
    Difference from Adam:
        - L2 regularization in Adam: adds λθ to gradient
        - Weight decay in AdamW: directly subtracts λθ from weights
    
    This decoupling improves generalization in deep learning.
    
    Update Rule:
        (Same as Adam, but add: θ = θ - α × λ × θ)
    
    Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
    """
    
    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(learning_rate, beta1, beta2, epsilon)
        self.weight_decay = weight_decay
    
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        # Weight decay (decoupled from gradient)
        params = params - self.learning_rate * self.weight_decay * params
        
        # Standard Adam step
        return super().step(params, grads)


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Divides learning rate by running average of gradient magnitudes.
    Good for non-stationary objectives and RNNs.
    
    Update Rules:
        v = β × v + (1 - β) × g²
        θ = θ - α × g / (√v + ε)
    
    Args:
        learning_rate: Step size
        decay: Decay rate for moving average
        epsilon: Numerical stability
    """
    
    def __init__(self, learning_rate: float = 0.01,
                 decay: float = 0.99,
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.decay = decay
        self.epsilon = epsilon
        self.v = None
    
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(params)
        
        # Update moving average of squared gradients
        self.v = self.decay * self.v + (1 - self.decay) * (grads ** 2)
        
        self.iterations += 1
        return params - self.learning_rate * grads / (np.sqrt(self.v) + self.epsilon)
    
    def reset(self):
        super().reset()
        self.v = None


class AdaGrad(Optimizer):
    """
    Adaptive Gradient Algorithm.
    
    Adapts learning rate based on historical gradients.
    Good for sparse data (NLP, recommendations).
    
    Update Rules:
        G = G + g²
        θ = θ - α × g / (√G + ε)
    
    Note: Learning rate decreases monotonically (can be too aggressive).
    """
    
    def __init__(self, learning_rate: float = 0.01,
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.G = None  # Accumulated squared gradients
    
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.G is None:
            self.G = np.zeros_like(params)
        
        # Accumulate squared gradients
        self.G = self.G + grads ** 2
        
        self.iterations += 1
        return params - self.learning_rate * grads / (np.sqrt(self.G) + self.epsilon)
    
    def reset(self):
        super().reset()
        self.G = None


# ============================================================
# LEARNING RATE SCHEDULERS
# ============================================================

class LRScheduler(ABC):
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.learning_rate
        self.step_count = 0
    
    @abstractmethod
    def step(self):
        """Update learning rate."""
        pass
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.learning_rate


class StepLR(LRScheduler):
    """
    Decay learning rate by gamma every step_size epochs.
    
    lr = base_lr × γ^(epoch // step_size)
    """
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
    
    def step(self):
        self.step_count += 1
        factor = self.gamma ** (self.step_count // self.step_size)
        self.optimizer.learning_rate = self.base_lr * factor


class ExponentialLR(LRScheduler):
    """
    Exponential decay.
    
    lr = base_lr × γ^epoch
    """
    
    def __init__(self, optimizer: Optimizer, gamma: float = 0.95):
        super().__init__(optimizer)
        self.gamma = gamma
    
    def step(self):
        self.step_count += 1
        self.optimizer.learning_rate = self.base_lr * (self.gamma ** self.step_count)


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing schedule.
    
    lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))
    
    Smoothly decreases LR following cosine curve.
    Popular in Transformers and vision models.
    """
    
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def step(self):
        self.step_count += 1
        t = self.step_count % self.T_max
        self.optimizer.learning_rate = (
            self.eta_min + 0.5 * (self.base_lr - self.eta_min) * 
            (1 + np.cos(np.pi * t / self.T_max))
        )


class WarmupScheduler(LRScheduler):
    """
    Linear warmup followed by decay.
    
    Used in Transformers to prevent early instability.
    """
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int,
                 total_steps: int, min_lr: float = 0.0):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            # Linear decay
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 - progress)
        
        self.optimizer.learning_rate = lr


# ============================================================
# REGULARIZATION
# ============================================================

def l1_regularization(weights: np.ndarray, lambda_: float) -> Tuple[float, np.ndarray]:
    """
    L1 (Lasso) regularization.
    
    Penalty: λ × Σ|wᵢ|
    Gradient: λ × sign(w)
    
    Effect: Encourages sparsity (many weights become exactly 0)
    Use: Feature selection
    
    Args:
        weights: Model weights
        lambda_: Regularization strength
    
    Returns:
        Tuple of (penalty, gradient)
    """
    penalty = lambda_ * np.sum(np.abs(weights))
    gradient = lambda_ * np.sign(weights)
    return penalty, gradient


def l2_regularization(weights: np.ndarray, lambda_: float) -> Tuple[float, np.ndarray]:
    """
    L2 (Ridge) regularization.
    
    Penalty: λ × Σwᵢ²
    Gradient: 2λ × w
    
    Effect: Shrinks weights towards zero (but rarely exactly 0)
    Use: Prevent overfitting, weight decay
    
    Args:
        weights: Model weights
        lambda_: Regularization strength
    
    Returns:
        Tuple of (penalty, gradient)
    """
    penalty = lambda_ * np.sum(weights ** 2)
    gradient = 2 * lambda_ * weights
    return penalty, gradient


def elastic_net_regularization(weights: np.ndarray, lambda_: float,
                                l1_ratio: float = 0.5) -> Tuple[float, np.ndarray]:
    """
    Elastic Net regularization (L1 + L2).
    
    Penalty: λ × (ρ × |w| + (1-ρ) × w²)
    
    Combines benefits of L1 (sparsity) and L2 (stability).
    
    Args:
        weights: Model weights
        lambda_: Regularization strength
        l1_ratio: Mix of L1 vs L2 (0=L2 only, 1=L1 only)
    
    Returns:
        Tuple of (penalty, gradient)
    """
    l1_pen, l1_grad = l1_regularization(weights, lambda_ * l1_ratio)
    l2_pen, l2_grad = l2_regularization(weights, lambda_ * (1 - l1_ratio))
    
    return l1_pen + l2_pen, l1_grad + l2_grad


# ============================================================
# GRADIENT DESCENT TRAINING LOOP
# ============================================================

def gradient_descent_train(
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
    initial_params: np.ndarray,
    optimizer: Optimizer,
    epochs: int = 100,
    batch_size: Optional[int] = None,
    regularization: Optional[Callable] = None,
    reg_lambda: float = 0.01,
    verbose: bool = True
) -> Tuple[np.ndarray, List[float]]:
    """
    Generic gradient descent training loop.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        loss_fn: Function that computes loss and gradients
                 signature: (X, y, params) -> (loss, gradients)
        initial_params: Starting parameters
        optimizer: Optimizer instance
        epochs: Number of training epochs
        batch_size: Mini-batch size (None = full batch)
        regularization: Regularization function
        reg_lambda: Regularization strength
        verbose: Print progress
    
    Returns:
        Tuple of (final_params, loss_history)
    
    Example:
        >>> def mse_loss(X, y, w):
        ...     pred = X @ w
        ...     loss = np.mean((pred - y) ** 2)
        ...     grad = 2/len(y) * X.T @ (pred - y)
        ...     return loss, grad
        >>> 
        >>> optimizer = Adam(learning_rate=0.01)
        >>> params, history = gradient_descent_train(X, y, mse_loss, np.zeros(n), optimizer)
    """
    params = initial_params.copy()
    n_samples = X.shape[0]
    loss_history = []
    
    if batch_size is None:
        batch_size = n_samples
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_losses = []
        
        # Mini-batch iteration
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute loss and gradients
            loss, grads = loss_fn(X_batch, y_batch, params)
            
            # Add regularization
            if regularization is not None:
                reg_penalty, reg_grad = regularization(params, reg_lambda)
                loss += reg_penalty
                grads += reg_grad
            
            # Update parameters
            params = optimizer.step(params, grads)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return params, loss_history


# ============================================================
# NUMERICAL GRADIENT CHECKING
# ============================================================

def numerical_gradient(f: LossFunc, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Compute numerical gradient for gradient checking.
    
    Uses central difference: (f(x+ε) - f(x-ε)) / 2ε
    
    Args:
        f: Loss function
        x: Point to evaluate gradient
        epsilon: Perturbation size
    
    Returns:
        Numerical gradient approximation
    """
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    
    return grad


def gradient_check(analytical_grad: np.ndarray, numerical_grad: np.ndarray,
                   threshold: float = 1e-5) -> bool:
    """
    Verify analytical gradient against numerical gradient.
    
    Uses relative error: ||a - n|| / (||a|| + ||n||)
    
    Args:
        analytical_grad: Gradient from backpropagation
        numerical_grad: Gradient from numerical approximation
        threshold: Maximum allowed relative error
    
    Returns:
        True if gradients match within threshold
    """
    diff = np.linalg.norm(analytical_grad - numerical_grad)
    denom = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    
    if denom == 0:
        return diff == 0
    
    relative_error = diff / denom
    return relative_error < threshold


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Optimizers
    'Optimizer', 'SGD', 'Adam', 'AdamW', 'RMSprop', 'AdaGrad',
    # Schedulers
    'LRScheduler', 'StepLR', 'ExponentialLR', 'CosineAnnealingLR', 'WarmupScheduler',
    # Regularization
    'l1_regularization', 'l2_regularization', 'elastic_net_regularization',
    # Training
    'gradient_descent_train',
    # Utilities
    'numerical_gradient', 'gradient_check',
]
