"""
Neural Network Optimizers Module.

This module provides comprehensive optimization algorithms for training neural networks,
including SGD, Momentum, Adam, RMSprop, Adagrad, and more.

Each optimizer includes:
- Parameter update rules
- Learning rate scheduling
- Gradient clipping
- Weight decay

Example Usage:
    >>> import numpy as np
    >>> from optimizers import SGD, Adam, RMSprop
    >>> 
    >>> # Adam optimizer
    >>> optimizer = Adam(learning_rate=0.001)
    >>> params = {'weight': np.random.randn(10, 5), 'bias': np.zeros(5)}
    >>> grads = {'weight': np.random.randn(10, 5), 'bias': np.random.randn(5)}
    >>> params = optimizer.step(params, grads)
"""

from typing import Union, Optional, Dict, List, Tuple, Callable
import numpy as np
from numpy.typing import ArrayLike
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

Parameters = Dict[str, np.ndarray]
Gradients = Dict[str, np.ndarray]


class Optimizer(ABC):
    """Base class for optimization algorithms."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0
    ):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Base learning rate.
            weight_decay: L2 regularization strength.
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self._step = 0
    
    @abstractmethod
    def step(
        self,
        params: Parameters,
        grads: Gradients
    ) -> Parameters:
        """
        Perform one optimization step.
        
        Args:
            params: Model parameters.
            grads: Parameter gradients.
        
        Returns:
            Parameters: Updated parameters.
        """
        pass
    
    def zero_grad(self, grads: Gradients) -> Gradients:
        """
        Zero out all gradients.
        
        Args:
            grads: Current gradients.
        
        Returns:
            Gradients: Zeroed gradients.
        """
        return {k: np.zeros_like(v) for k, v in grads.items()}
    
    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate."""
        self.learning_rate = lr
        logger.debug(f"Learning rate set to {lr}")
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.learning_rate


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Update rule: θ = θ - lr * ∇L(θ)
    
    Properties:
    - Simple and effective
    - Can get stuck in local minima
    - May oscillate in narrow valleys
    
    Example:
        >>> optimizer = SGD(learning_rate=0.01)
        >>> params = {'w': np.array([1.0, 2.0])}
        >>> grads = {'w': np.array([0.1, 0.2])}
        >>> params = optimizer.step(params, grads)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False
    ):
        """
        Initialize SGD.
        
        Args:
            learning_rate: Learning rate. Default: 0.01.
            momentum: Momentum factor. Default: 0.0.
            weight_decay: L2 regularization. Default: 0.0.
            dampening: Momentum dampening. Default: 0.0.
            nesterov: Use Nesterov momentum. Default: False.
        """
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        
        self._velocity: Dict[str, np.ndarray] = {}
        
        logger.debug(f"SGD initialized: lr={learning_rate}, momentum={momentum}")
    
    def step(
        self,
        params: Parameters,
        grads: Gradients
    ) -> Parameters:
        """
        Perform SGD update step.
        
        Args:
            params: Model parameters.
            grads: Parameter gradients.
        
        Returns:
            Parameters: Updated parameters.
        """
        self._step += 1
        updated_params = {}
        
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated_params[name] = param
                continue
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Initialize velocity
            if name not in self._velocity:
                self._velocity[name] = np.zeros_like(param)
            
            # Update velocity
            if self.momentum > 0:
                if self.nesterov:
                    # Nesterov momentum
                    self._velocity[name] = self.momentum * self._velocity[name] + grad
                    grad = grad + self.momentum * self._velocity[name]
                else:
                    # Classic momentum
                    self._velocity[name] = self.momentum * self._velocity[name] + \
                                          (1 - self.dampening) * grad
                    grad = self._velocity[name]
            
            # Update parameter
            updated_params[name] = param - self.learning_rate * grad
        
        logger.debug(f"SGD step {self._step}: updated {len(updated_params)} parameters")
        return updated_params


class Momentum(Optimizer):
    """
    SGD with Momentum optimizer.
    
    Update rules:
    v_t = μ * v_{t-1} + ∇L(θ)
    θ = θ - lr * v_t
    
    Properties:
    - Accelerates convergence in relevant direction
    - Dampens oscillations
    - Can overshoot minima
    
    Example:
        >>> optimizer = Momentum(learning_rate=0.01, momentum=0.9)
        >>> params = {'w': np.array([1.0, 2.0])}
        >>> grads = {'w': np.array([0.1, 0.2])}
        >>> params = optimizer.step(params, grads)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0
    ):
        """
        Initialize Momentum optimizer.
        
        Args:
            learning_rate: Learning rate.
            momentum: Momentum coefficient. Default: 0.9.
            weight_decay: L2 regularization.
        """
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self._velocity: Dict[str, np.ndarray] = {}
        
        logger.debug(f"Momentum initialized: lr={learning_rate}, momentum={momentum}")
    
    def step(
        self,
        params: Parameters,
        grads: Gradients
    ) -> Parameters:
        """
        Perform Momentum update step.
        
        Args:
            params: Model parameters.
            grads: Parameter gradients.
        
        Returns:
            Parameters: Updated parameters.
        """
        self._step += 1
        updated_params = {}
        
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated_params[name] = param
                continue
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Initialize velocity
            if name not in self._velocity:
                self._velocity[name] = np.zeros_like(param)
            
            # Update velocity and parameter
            self._velocity[name] = self.momentum * self._velocity[name] + grad
            updated_params[name] = param - self.learning_rate * self._velocity[name]
        
        logger.debug(f"Momentum step {self._step}")
        return updated_params


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Update rules:
    m_t = β1 * m_{t-1} + (1 - β1) * ∇L(θ)
    v_t = β2 * v_{t-1} + (1 - β2) * ∇L(θ)²
    m̂_t = m_t / (1 - β1^t)  # Bias correction
    v̂_t = v_t / (1 - β2^t)  # Bias correction
    θ = θ - lr * m̂_t / (√v̂_t + ε)
    
    Properties:
    - Combines momentum and RMSprop
    - Adaptive learning rates per parameter
    - Works well out of the box
    
    Example:
        >>> optimizer = Adam(learning_rate=0.001)
        >>> params = {'w': np.array([1.0, 2.0])}
        >>> grads = {'w': np.array([0.1, 0.2])}
        >>> params = optimizer.step(params, grads)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate. Default: 0.001.
            beta1: Exponential decay for first moment. Default: 0.9.
            beta2: Exponential decay for second moment. Default: 0.999.
            epsilon: Numerical stability constant. Default: 1e-8.
            weight_decay: L2 regularization. Default: 0.0.
            amsgrad: Use AMSGrad variant. Default: False.
        """
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        
        self._m: Dict[str, np.ndarray] = {}  # First moment
        self._v: Dict[str, np.ndarray] = {}  # Second moment
        self._v_hat: Dict[str, np.ndarray] = {}  # For AMSGrad
        
        logger.debug(f"Adam initialized: lr={learning_rate}, beta1={beta1}, beta2={beta2}")
    
    def step(
        self,
        params: Parameters,
        grads: Gradients
    ) -> Parameters:
        """
        Perform Adam update step.
        
        Args:
            params: Model parameters.
            grads: Parameter gradients.
        
        Returns:
            Parameters: Updated parameters.
        """
        self._step += 1
        updated_params = {}
        
        # Bias correction factors
        bias_correction1 = 1 - self.beta1 ** self._step
        bias_correction2 = 1 - self.beta2 ** self._step
        
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated_params[name] = param
                continue
            
            # Add weight decay (decoupled weight decay)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Initialize moments
            if name not in self._m:
                self._m[name] = np.zeros_like(param)
                self._v[name] = np.zeros_like(param)
                if self.amsgrad:
                    self._v_hat[name] = np.zeros_like(param)
            
            # Update biased first moment estimate
            self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * (grad ** 2)
            
            if self.amsgrad:
                # AMSGrad: maintain max of all v_hat
                self._v_hat[name] = np.maximum(self._v_hat[name], self._v[name])
                v_corrected = self._v_hat[name] / bias_correction2
            else:
                v_corrected = self._v[name] / bias_correction2
            
            # Compute update
            m_corrected = self._m[name] / bias_correction1
            update = m_corrected / (np.sqrt(v_corrected) + self.epsilon)
            
            updated_params[name] = param - self.learning_rate * update
        
        logger.debug(f"Adam step {self._step}")
        return updated_params


class RMSprop(Optimizer):
    """
    RMSprop (Root Mean Square Propagation) optimizer.
    
    Update rules:
    v_t = α * v_{t-1} + (1 - α) * ∇L(θ)²
    θ = θ - lr * ∇L(θ) / (√v_t + ε)
    
    Properties:
    - Adaptive learning rates
    - Good for non-stationary objectives
    - Works well for RNNs
    
    Example:
        >>> optimizer = RMSprop(learning_rate=0.01)
        >>> params = {'w': np.array([1.0, 2.0])}
        >>> grads = {'w': np.array([0.1, 0.2])}
        >>> params = optimizer.step(params, grads)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        alpha: float = 0.99,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False
    ):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate. Default: 0.01.
            alpha: Smoothing constant. Default: 0.99.
            epsilon: Numerical stability constant. Default: 1e-8.
            weight_decay: L2 regularization. Default: 0.0.
            momentum: Momentum factor. Default: 0.0.
            centered: Use centered RMSprop. Default: False.
        """
        super().__init__(learning_rate, weight_decay)
        self.alpha = alpha
        self.epsilon = epsilon
        self.momentum = momentum
        self.centered = centered
        
        self._v: Dict[str, np.ndarray] = {}  # Running average of squared gradients
        self._g_avg: Dict[str, np.ndarray] = {}  # For centered version
        self._momentum_buffer: Dict[str, np.ndarray] = {}
        
        logger.debug(f"RMSprop initialized: lr={learning_rate}, alpha={alpha}")
    
    def step(
        self,
        params: Parameters,
        grads: Gradients
    ) -> Parameters:
        """
        Perform RMSprop update step.
        
        Args:
            params: Model parameters.
            grads: Parameter gradients.
        
        Returns:
            Parameters: Updated parameters.
        """
        self._step += 1
        updated_params = {}
        
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated_params[name] = param
                continue
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Initialize buffers
            if name not in self._v:
                self._v[name] = np.zeros_like(param)
                if self.centered:
                    self._g_avg[name] = np.zeros_like(param)
                if self.momentum > 0:
                    self._momentum_buffer[name] = np.zeros_like(param)
            
            # Update running average of squared gradients
            self._v[name] = self.alpha * self._v[name] + (1 - self.alpha) * (grad ** 2)
            
            if self.centered:
                # Centered RMSprop
                self._g_avg[name] = self.alpha * self._g_avg[name] + (1 - self.alpha) * grad
                denom = np.sqrt(self._v[name] - self._g_avg[name] ** 2 + self.epsilon)
            else:
                denom = np.sqrt(self._v[name] + self.epsilon)
            
            # Compute update
            update = grad / denom
            
            if self.momentum > 0:
                self._momentum_buffer[name] = self.momentum * self._momentum_buffer[name] + update
                update = self._momentum_buffer[name]
            
            updated_params[name] = param - self.learning_rate * update
        
        logger.debug(f"RMSprop step {self._step}")
        return updated_params


class Adagrad(Optimizer):
    """
    Adagrad (Adaptive Gradient) optimizer.
    
    Update rules:
    v_t = v_{t-1} + ∇L(θ)²
    θ = θ - lr * ∇L(θ) / (√v_t + ε)
    
    Properties:
    - Adapts learning rate to parameters
    - Good for sparse data
    - Learning rate monotonically decreases
    
    Example:
        >>> optimizer = Adagrad(learning_rate=0.01)
        >>> params = {'w': np.array([1.0, 2.0])}
        >>> grads = {'w': np.array([0.1, 0.2])}
        >>> params = optimizer.step(params, grads)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adagrad optimizer.
        
        Args:
            learning_rate: Learning rate. Default: 0.01.
            epsilon: Numerical stability constant. Default: 1e-8.
            weight_decay: L2 regularization. Default: 0.0.
        """
        super().__init__(learning_rate, weight_decay)
        self.epsilon = epsilon
        
        self._v: Dict[str, np.ndarray] = {}
        
        logger.debug(f"Adagrad initialized: lr={learning_rate}")
    
    def step(
        self,
        params: Parameters,
        grads: Gradients
    ) -> Parameters:
        """
        Perform Adagrad update step.
        
        Args:
            params: Model parameters.
            grads: Parameter gradients.
        
        Returns:
            Parameters: Updated parameters.
        """
        self._step += 1
        updated_params = {}
        
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated_params[name] = param
                continue
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Initialize accumulator
            if name not in self._v:
                self._v[name] = np.zeros_like(param)
            
            # Update accumulator
            self._v[name] += grad ** 2
            
            # Compute update
            update = grad / (np.sqrt(self._v[name]) + self.epsilon)
            
            updated_params[name] = param - self.learning_rate * update
        
        logger.debug(f"Adagrad step {self._step}")
        return updated_params


class AdamW(Optimizer):
    """
    AdamW (Adam with decoupled weight decay) optimizer.
    
    Same as Adam but with decoupled weight decay:
    θ = θ - lr * (m̂_t / (√v̂_t + ε) + λ * θ)
    
    Properties:
    - Proper weight decay implementation
    - Better generalization than Adam
    - Recommended for transformers
    
    Example:
        >>> optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01
    ):
        """
        Initialize AdamW optimizer.
        
        Args:
            learning_rate: Learning rate. Default: 0.001.
            beta1: First moment decay. Default: 0.9.
            beta2: Second moment decay. Default: 0.999.
            epsilon: Numerical stability. Default: 1e-8.
            weight_decay: Weight decay. Default: 0.01.
        """
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}
        
        logger.debug(f"AdamW initialized: lr={learning_rate}, wd={weight_decay}")
    
    def step(
        self,
        params: Parameters,
        grads: Gradients
    ) -> Parameters:
        """
        Perform AdamW update step with decoupled weight decay.
        
        Args:
            params: Model parameters.
            grads: Parameter gradients.
        
        Returns:
            Parameters: Updated parameters.
        """
        self._step += 1
        updated_params = {}
        
        bias_correction1 = 1 - self.beta1 ** self._step
        bias_correction2 = 1 - self.beta2 ** self._step
        
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated_params[name] = param
                continue
            
            # Initialize moments
            if name not in self._m:
                self._m[name] = np.zeros_like(param)
                self._v[name] = np.zeros_like(param)
            
            # Update moments
            self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * grad
            self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_corrected = self._m[name] / bias_correction1
            v_corrected = self._v[name] / bias_correction2
            
            # Adam update + decoupled weight decay
            adam_update = m_corrected / (np.sqrt(v_corrected) + self.epsilon)
            updated_params[name] = param - self.learning_rate * (adam_update + self.weight_decay * param)
        
        logger.debug(f"AdamW step {self._step}")
        return updated_params


class LearningRateScheduler:
    """
    Learning rate scheduling utilities.
    
    Example:
        >>> scheduler = LearningRateScheduler('step', initial_lr=0.1, milestones=[30, 60], gamma=0.1)
        >>> optimizer = SGD(learning_rate=scheduler.get_lr(0))
        >>> for epoch in range(100):
        ...     optimizer.set_learning_rate(scheduler.get_lr(epoch))
    """
    
    def __init__(
        self,
        schedule_type: str = 'constant',
        initial_lr: float = 0.1,
        **kwargs
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            schedule_type: Type of schedule ('constant', 'step', 'exponential', 'cosine', 'linear').
            initial_lr: Initial learning rate.
            **kwargs: Schedule-specific parameters.
        """
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.kwargs = kwargs
    
    def get_lr(self, epoch: int) -> float:
        """
        Get learning rate for given epoch.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            float: Learning rate.
        """
        if self.schedule_type == 'constant':
            return self.initial_lr
        
        elif self.schedule_type == 'step':
            milestones = self.kwargs.get('milestones', [])
            gamma = self.kwargs.get('gamma', 0.1)
            
            lr = self.initial_lr
            for milestone in milestones:
                if epoch >= milestone:
                    lr *= gamma
            return lr
        
        elif self.schedule_type == 'exponential':
            gamma = self.kwargs.get('gamma', 0.95)
            return self.initial_lr * (gamma ** epoch)
        
        elif self.schedule_type == 'cosine':
            max_epochs = self.kwargs.get('max_epochs', 100)
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
        
        elif self.schedule_type == 'linear':
            max_epochs = self.kwargs.get('max_epochs', 100)
            min_lr = self.kwargs.get('min_lr', 0.0)
            return self.initial_lr - (self.initial_lr - min_lr) * (epoch / max_epochs)
        
        elif self.schedule_type == 'warmup_cosine':
            warmup_epochs = self.kwargs.get('warmup_epochs', 5)
            max_epochs = self.kwargs.get('max_epochs', 100)
            
            if epoch < warmup_epochs:
                return self.initial_lr * (epoch / warmup_epochs)
            else:
                return self.initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


class GradientClipper:
    """
    Gradient clipping utilities.
    
    Example:
        >>> clipper = GradientClipper(max_norm=1.0)
        >>> grads = {'w': np.random.randn(10, 5)}
        >>> grads = clipper.clip(grads)
    """
    
    def __init__(
        self,
        max_norm: Optional[float] = None,
        clip_value: Optional[float] = None
    ):
        """
        Initialize gradient clipper.
        
        Args:
            max_norm: Maximum norm for clipping.
            clip_value: Maximum absolute value for clipping.
        """
        self.max_norm = max_norm
        self.clip_value = clip_value
        
        logger.debug(f"GradientClipper: max_norm={max_norm}, clip_value={clip_value}")
    
    def clip(self, grads: Gradients) -> Gradients:
        """
        Clip gradients.
        
        Args:
            grads: Original gradients.
        
        Returns:
            Gradients: Clipped gradients.
        """
        if self.max_norm is None and self.clip_value is None:
            return grads
        
        clipped_grads = {}
        
        if self.clip_value is not None:
            # Clip by value
            for name, grad in grads.items():
                clipped_grads[name] = np.clip(grad, -self.clip_value, self.clip_value)
        else:
            # Clip by norm
            total_norm = 0.0
            for grad in grads.values():
                total_norm += np.sum(grad ** 2)
            total_norm = np.sqrt(total_norm)
            
            clip_coef = self.max_norm / (total_norm + 1e-6)
            clip_coef = min(clip_coef, 1.0)
            
            for name, grad in grads.items():
                clipped_grads[name] = grad * clip_coef
        
        return clipped_grads


def get_optimizer(
    name: str,
    learning_rate: float = 0.001,
    **kwargs
) -> Optimizer:
    """
    Factory function to get optimizer by name.
    
    Args:
        name: Optimizer name ('sgd', 'adam', 'rmsprop', etc.).
        learning_rate: Learning rate.
        **kwargs: Optimizer-specific arguments.
    
    Returns:
        Optimizer: Optimizer instance.
    
    Raises:
        ValueError: If name is not recognized.
    
    Example:
        >>> optimizer = get_optimizer('adam', learning_rate=0.001)
        >>> optimizer = get_optimizer('sgd', learning_rate=0.01, momentum=0.9)
    """
    optimizers = {
        'sgd': SGD,
        'momentum': Momentum,
        'adam': Adam,
        'adamw': AdamW,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
    }
    
    name_lower = name.lower()
    if name_lower not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name_lower](learning_rate=learning_rate, **kwargs)


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Optimizers Module - Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simple quadratic function: f(x) = x²
    # Minimum at x = 0
    def loss_and_grad(x):
        return x ** 2, 2 * x
    
    x0 = np.array([5.0])
    optimizers_to_test = [
        ('SGD', SGD(learning_rate=0.1)),
        ('Momentum', Momentum(learning_rate=0.1, momentum=0.9)),
        ('Adam', Adam(learning_rate=0.1)),
        ('RMSprop', RMSprop(learning_rate=0.1)),
        ('Adagrad', Adagrad(learning_rate=0.5)),
    ]
    
    print("\nOptimizing f(x) = x², starting at x = 5.0")
    print("-" * 60)
    
    for name, optimizer in optimizers_to_test:
        x = x0.copy()
        params = {'x': x}
        
        for step in range(50):
            _, grad = loss_and_grad(params['x'])
            grads = {'x': grad}
            params = optimizer.step(params, grads)
        
        final_loss, _ = loss_and_grad(params['x'])
        print(f"{name:12s}: x = {params['x'][0]:.6f}, loss = {final_loss:.6e}")
    
    # Learning rate scheduler
    print("\n\nLearning Rate Schedules (initial_lr = 0.1):")
    print("-" * 60)
    
    schedulers = [
        ('Constant', LearningRateScheduler('constant', initial_lr=0.1)),
        ('Step', LearningRateScheduler('step', initial_lr=0.1, milestones=[30, 60], gamma=0.1)),
        ('Exponential', LearningRateScheduler('exponential', initial_lr=0.1, gamma=0.95)),
        ('Cosine', LearningRateScheduler('cosine', initial_lr=0.1, max_epochs=100)),
        ('Linear', LearningRateScheduler('linear', initial_lr=0.1, max_epochs=100)),
    ]
    
    for name, scheduler in schedulers:
        lrs = [scheduler.get_lr(i) for i in [0, 25, 50, 75, 100]]
        print(f"{name:12s}: {lrs}")
    
    # Gradient clipping
    print("\n\nGradient Clipping:")
    print("-" * 60)
    
    grads = {'w': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
    norm = np.sqrt(np.sum(np.array([1, 4, 9, 16, 25])))
    print(f"Original gradient norm: {norm:.4f}")
    
    clipper_norm = GradientClipper(max_norm=1.0)
    clipped = clipper_norm.clip(grads)
    clipped_norm = np.sqrt(np.sum(clipped['w'] ** 2))
    print(f"After norm clipping (max=1.0): {clipped_norm:.4f}")
    
    clipper_value = GradientClipper(clip_value=2.0)
    clipped = clipper_value.clip(grads)
    print(f"After value clipping (max=2.0): {clipped['w']}")
    
    print("\n" + "=" * 60)
