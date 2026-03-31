"""
Neural Network Activation Functions Module.

This module provides comprehensive activation functions for neural networks,
including sigmoid, ReLU, Tanh, Softmax, LeakyReLU, ELU, Swish, and more.

Each activation function includes:
- Forward pass computation
- Backward pass (derivative) for backpropagation
- Numerical stability optimizations

Example Usage:
    >>> import numpy as np
    >>> from activations import ReLU, Sigmoid, Softmax
    >>> 
    >>> # ReLU activation
    >>> relu = ReLU()
    >>> x = np.array([-1, 0, 1, 2])
    >>> output = relu.forward(x)
    >>> grad = relu.backward(np.ones_like(x))
    >>> 
    >>> # Softmax for classification
    >>> softmax = Softmax()
    >>> logits = np.array([[1, 2, 3], [4, 5, 6]])
    >>> probs = softmax.forward(logits)
"""

from typing import Union, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
import logging

logger = logging.getLogger(__name__)

ArrayLike2D = Union[np.ndarray, List]


class ActivationFunction:
    """Base class for activation functions."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute forward pass.
        
        Args:
            x: Input array.
        
        Returns:
            np.ndarray: Activated output.
        """
        raise NotImplementedError
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute backward pass (gradient).
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        raise NotImplementedError
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow calling instance as function."""
        return self.forward(x)


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    
    σ(x) = 1 / (1 + exp(-x))
    
    Properties:
    - Output range: (0, 1)
    - Zero-centered: No
    - Use case: Binary classification output, gating mechanisms
    
    Example:
        >>> sigmoid = Sigmoid()
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> output = sigmoid.forward(x)
        >>> np.allclose(output, [0.119, 0.269, 0.5, 0.731, 0.881], atol=0.001)
        True
    """
    
    def __init__(self):
        self._cache: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid forward pass.
        
        Args:
            x: Input array.
        
        Returns:
            np.ndarray: Sigmoid output.
        """
        x = np.asarray(x, dtype=np.float64)
        
        # Clip for numerical stability
        x_clipped = np.clip(x, -500, 500)
        output = 1 / (1 + np.exp(-x_clipped))
        
        # Cache for backward pass
        self._cache = output
        
        logger.debug(f"Sigmoid forward: input shape {x.shape}, output range [{output.min():.4f}, {output.max():.4f}]")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid backward pass.
        
        d/dx σ(x) = σ(x) * (1 - σ(x))
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        sigmoid_output = self._cache
        grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)
        
        logger.debug(f"Sigmoid backward: gradient shape {grad_input.shape}")
        return grad_input


class Tanh(ActivationFunction):
    """
    Hyperbolic Tangent activation function.
    
    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Properties:
    - Output range: (-1, 1)
    - Zero-centered: Yes
    - Use case: Hidden layers, RNNs
    
    Example:
        >>> tanh = Tanh()
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> output = tanh.forward(x)
        >>> np.allclose(output, [-0.964, -0.762, 0, 0.762, 0.964], atol=0.001)
        True
    """
    
    def __init__(self):
        self._cache: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute tanh forward pass.
        
        Args:
            x: Input array.
        
        Returns:
            np.ndarray: Tanh output.
        """
        x = np.asarray(x, dtype=np.float64)
        output = np.tanh(x)
        
        self._cache = output
        
        logger.debug(f"Tanh forward: input shape {x.shape}, output range [{output.min():.4f}, {output.max():.4f}]")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute tanh backward pass.
        
        d/dx tanh(x) = 1 - tanh²(x)
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        tanh_output = self._cache
        grad_input = grad_output * (1 - tanh_output ** 2)
        
        logger.debug(f"Tanh backward: gradient shape {grad_input.shape}")
        return grad_input


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit activation function.
    
    ReLU(x) = max(0, x)
    
    Properties:
    - Output range: [0, ∞)
    - Zero-centered: No
    - Use case: Default choice for hidden layers
    
    Example:
        >>> relu = ReLU()
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> output = relu.forward(x)
        >>> np.array_equal(output, [0, 0, 0, 1, 2])
        True
    """
    
    def __init__(self):
        self._cache: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute ReLU forward pass.
        
        Args:
            x: Input array.
        
        Returns:
            np.ndarray: ReLU output.
        """
        x = np.asarray(x, dtype=np.float64)
        output = np.maximum(0, x)
        
        self._cache = (x > 0).astype(np.float64)  # Mask for backward
        
        logger.debug(f"ReLU forward: input shape {x.shape}, output range [{output.min():.4f}, {output.max():.4f}]")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute ReLU backward pass.
        
        d/dx ReLU(x) = 1 if x > 0, else 0
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        grad_input = grad_output * self._cache
        
        logger.debug(f"ReLU backward: gradient shape {grad_input.shape}")
        return grad_input


class LeakyReLU(ActivationFunction):
    """
    Leaky ReLU activation function.
    
    LeakyReLU(x) = x if x > 0, else α * x
    
    Properties:
    - Output range: (-∞, ∞)
    - Zero-centered: No (but allows negative values)
    - Use case: Alternative to ReLU, prevents dying ReLU problem
    
    Example:
        >>> leaky_relu = LeakyReLU(alpha=0.01)
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> output = leaky_relu.forward(x)
        >>> np.allclose(output, [-0.02, -0.01, 0, 1, 2])
        True
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Initialize LeakyReLU.
        
        Args:
            alpha: Slope for negative values. Default: 0.01.
        """
        self.alpha = alpha
        self._cache: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute LeakyReLU forward pass.
        
        Args:
            x: Input array.
        
        Returns:
            np.ndarray: LeakyReLU output.
        """
        x = np.asarray(x, dtype=np.float64)
        output = np.where(x > 0, x, self.alpha * x)
        
        self._cache = (x > 0).astype(np.float64)
        
        logger.debug(f"LeakyReLU forward: input shape {x.shape}, alpha={self.alpha}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute LeakyReLU backward pass.
        
        d/dx LeakyReLU(x) = 1 if x > 0, else α
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        grad_input = np.where(self._cache, grad_output, self.alpha * grad_output)
        
        logger.debug(f"LeakyReLU backward: gradient shape {grad_input.shape}")
        return grad_input


class ELU(ActivationFunction):
    """
    Exponential Linear Unit activation function.
    
    ELU(x) = x if x > 0, else α * (exp(x) - 1)
    
    Properties:
    - Output range: (-α, ∞)
    - Zero-centered: Approximately
    - Use case: Alternative to ReLU with smoother negative region
    
    Example:
        >>> elu = ELU(alpha=1.0)
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> output = elu.forward(x)
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize ELU.
        
        Args:
            alpha: Scale for negative values. Default: 1.0.
        """
        self.alpha = alpha
        self._cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute ELU forward pass.
        
        Args:
            x: Input array.
        
        Returns:
            np.ndarray: ELU output.
        """
        x = np.asarray(x, dtype=np.float64)
        
        # Clip for numerical stability
        x_clipped = np.clip(x, -500, 500)
        output = np.where(x > 0, x, self.alpha * (np.exp(x_clipped) - 1))
        
        self._cache = (x > 0, output)
        
        logger.debug(f"ELU forward: input shape {x.shape}, alpha={self.alpha}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute ELU backward pass.
        
        d/dx ELU(x) = 1 if x > 0, else ELU(x) + α
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        positive_mask, output = self._cache
        grad_input = np.where(positive_mask, grad_output, (output + self.alpha) * grad_output)
        
        logger.debug(f"ELU backward: gradient shape {grad_input.shape}")
        return grad_input


class Swish(ActivationFunction):
    """
    Swish activation function (SiLU).
    
    Swish(x) = x * σ(x) = x / (1 + exp(-x))
    
    Properties:
    - Output range: (-∞, ∞)
    - Zero-centered: No
    - Use case: Modern alternative to ReLU, used in EfficientNet
    
    Example:
        >>> swish = Swish()
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> output = swish.forward(x)
    """
    
    def __init__(self):
        self._cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Swish forward pass.
        
        Args:
            x: Input array.
        
        Returns:
            np.ndarray: Swish output.
        """
        x = np.asarray(x, dtype=np.float64)
        
        x_clipped = np.clip(x, -500, 500)
        sigmoid_x = 1 / (1 + np.exp(-x_clipped))
        output = x * sigmoid_x
        
        self._cache = (x, sigmoid_x)
        
        logger.debug(f"Swish forward: input shape {x.shape}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute Swish backward pass.
        
        d/dx Swish(x) = Swish(x) + σ(x) * (1 - Swish(x))
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        x, sigmoid_x = self._cache
        swish_x = x * sigmoid_x
        
        grad_input = (swish_x + sigmoid_x * (1 - swish_x)) * grad_output
        
        logger.debug(f"Swish backward: gradient shape {grad_input.shape}")
        return grad_input


class Softmax(ActivationFunction):
    """
    Softmax activation function for multi-class classification.
    
    Softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
    
    Properties:
    - Output range: (0, 1), sums to 1
    - Use case: Multi-class classification output layer
    
    Note: Typically used with CrossEntropyLoss for numerical stability.
    
    Example:
        >>> softmax = Softmax()
        >>> logits = np.array([[1, 2, 3], [4, 5, 6]])
        >>> probs = softmax.forward(logits)
        >>> np.allclose(probs.sum(axis=1), 1.0)
        True
    """
    
    def __init__(self, axis: int = -1):
        """
        Initialize Softmax.
        
        Args:
            axis: Axis along which to compute softmax. Default: -1.
        """
        self.axis = axis
        self._cache: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute softmax forward pass.
        
        Args:
            x: Input array (logits).
        
        Returns:
            np.ndarray: Probability distribution.
        """
        x = np.asarray(x, dtype=np.float64)
        
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        output = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        
        self._cache = output
        
        logger.debug(f"Softmax forward: input shape {x.shape}, output sums to {output.sum(axis=self.axis).mean():.6f}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute softmax backward pass.
        
        For softmax with cross-entropy, gradient simplifies to (probs - target).
        For standalone softmax, full Jacobian is needed.
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        softmax_output = self._cache
        
        # For typical use with cross-entropy loss
        # Full Jacobian: J_ij = s_i * (δ_ij - s_j)
        # Simplified when combined with cross-entropy: grad = probs - target
        
        grad_input = softmax_output * (grad_output - np.sum(grad_output * softmax_output, axis=self.axis, keepdims=True))
        
        logger.debug(f"Softmax backward: gradient shape {grad_input.shape}")
        return grad_input


class GELU(ActivationFunction):
    """
    Gaussian Error Linear Unit activation function.
    
    GELU(x) = x * Φ(x) where Φ is the standard normal CDF
    
    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    Properties:
    - Output range: (-∞, ∞)
    - Zero-centered: Approximately
    - Use case: Transformer models (BERT, GPT)
    
    Example:
        >>> gelu = GELU()
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> output = gelu.forward(x)
    """
    
    def __init__(self, approximate: bool = True):
        """
        Initialize GELU.
        
        Args:
            approximate: Use approximate formula. Default: True.
        """
        self.approximate = approximate
        self._cache: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute GELU forward pass.
        
        Args:
            x: Input array.
        
        Returns:
            np.ndarray: GELU output.
        """
        x = np.asarray(x, dtype=np.float64)
        
        if self.approximate:
            # Approximate GELU
            coeff = np.sqrt(2 / np.pi)
            output = 0.5 * x * (1 + np.tanh(coeff * (x + 0.044715 * x ** 3)))
        else:
            # Exact GELU using error function
            from scipy.special import erf
            output = 0.5 * x * (1 + erf(x / np.sqrt(2)))
        
        self._cache = x
        
        logger.debug(f"GELU forward: input shape {x.shape}, approximate={self.approximate}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute GELU backward pass.
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        x = self._cache
        
        if self.approximate:
            coeff = np.sqrt(2 / np.pi)
            tanh_arg = coeff * (x + 0.044715 * x ** 3)
            sech_squared = 1 - np.tanh(tanh_arg) ** 2
            
            grad_input = 0.5 * (1 + np.tanh(tanh_arg)) + \
                        0.5 * x * coeff * sech_squared * (1 + 3 * 0.044715 * x ** 2)
        else:
            from scipy.special import erf
            from scipy.stats import norm
            grad_input = 0.5 * (1 + erf(x / np.sqrt(2))) + x * norm.pdf(x)
        
        grad_input = grad_input * grad_output
        
        logger.debug(f"GELU backward: gradient shape {grad_input.shape}")
        return grad_input


class SiLU(ActivationFunction):
    """
    Sigmoid Linear Unit (same as Swish with β=1).
    
    SiLU(x) = x * σ(x)
    
    Example:
        >>> silu = SiLU()
        >>> x = np.array([-1, 0, 1])
        >>> output = silu.forward(x)
    """
    
    def __init__(self):
        self._swish = Swish()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute SiLU forward pass."""
        return self._swish.forward(x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute SiLU backward pass."""
        return self._swish.backward(grad_output)


class Mish(ActivationFunction):
    """
    Mish activation function.
    
    Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    
    Properties:
    - Output range: (-∞, ∞)
    - Smooth, non-monotonic
    - Use case: Alternative to ReLU/Swish
    
    Example:
        >>> mish = Mish()
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> output = mish.forward(x)
    """
    
    def __init__(self):
        self._cache: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Mish forward pass.
        
        Args:
            x: Input array.
        
        Returns:
            np.ndarray: Mish output.
        """
        x = np.asarray(x, dtype=np.float64)
        
        # Clip for numerical stability
        x_clipped = np.clip(x, -20, 20)
        softplus = np.log(1 + np.exp(x_clipped))
        output = x * np.tanh(softplus)
        
        self._cache = x
        
        logger.debug(f"Mish forward: input shape {x.shape}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute Mish backward pass.
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        x = self._cache
        x_clipped = np.clip(x, -20, 20)
        
        omega = np.exp(x_clipped)
        softplus = np.log(1 + omega)
        tanh_softplus = np.tanh(softplus)
        
        # Derivative
        grad_input = tanh_softplus + x * omega / (1 + omega) * (1 - tanh_softplus ** 2)
        grad_input = grad_input * grad_output
        
        logger.debug(f"Mish backward: gradient shape {grad_input.shape}")
        return grad_input


class Identity(ActivationFunction):
    """
    Identity (linear) activation function.
    
    Identity(x) = x
    
    Use case: Regression output layer, linear layers
    
    Example:
        >>> identity = Identity()
        >>> x = np.array([-1, 0, 1])
        >>> output = identity.forward(x)
        >>> np.array_equal(output, x)
        True
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return input unchanged."""
        return np.asarray(x, dtype=np.float64)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Gradient passes through unchanged."""
        return grad_output


class BinaryStep(ActivationFunction):
    """
    Binary step (Heaviside) activation function.
    
    Step(x) = 1 if x >= 0, else 0
    
    Note: Not differentiable at x=0, gradient is 0 everywhere else.
    Use case: Binary classification (rarely used in practice due to gradient issues)
    
    Example:
        >>> step = BinaryStep()
        >>> x = np.array([-1, 0, 1])
        >>> output = step.forward(x)
        >>> np.array_equal(output, [0, 1, 1])
        True
    """
    
    def __init__(self, threshold: float = 0.0):
        """
        Initialize BinaryStep.
        
        Args:
            threshold: Threshold value. Default: 0.0.
        """
        self.threshold = threshold
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute step function."""
        x = np.asarray(x, dtype=np.float64)
        return (x >= self.threshold).astype(np.float64)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Return zero gradient (step function is not differentiable).
        
        In practice, use a surrogate gradient like sigmoid for training.
        """
        return np.zeros_like(grad_output)


def get_activation(name: str, **kwargs) -> ActivationFunction:
    """
    Factory function to get activation by name.
    
    Args:
        name: Activation name ('relu', 'sigmoid', 'tanh', etc.).
        **kwargs: Additional arguments for activation.
    
    Returns:
        ActivationFunction: Activation instance.
    
    Raises:
        ValueError: If name is not recognized.
    
    Example:
        >>> relu = get_activation('relu')
        >>> leaky_relu = get_activation('leaky_relu', alpha=0.1)
        >>> softmax = get_activation('softmax', axis=-1)
    """
    activations = {
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'elu': ELU,
        'swish': Swish,
        'silu': SiLU,
        'gelu': GELU,
        'mish': Mish,
        'softmax': Softmax,
        'identity': Identity,
        'linear': Identity,
        'binary_step': BinaryStep,
        'step': BinaryStep,
    }
    
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    
    return activations[name_lower](**kwargs)


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Activation Functions Module - Demonstration")
    print("=" * 60)
    
    x = np.array([-2, -1, 0, 1, 2])
    print(f"\nInput: {x}")
    
    activations = [
        ("Sigmoid", Sigmoid()),
        ("Tanh", Tanh()),
        ("ReLU", ReLU()),
        ("LeakyReLU", LeakyReLU(alpha=0.1)),
        ("ELU", ELU(alpha=1.0)),
        ("Swish", Swish()),
        ("GELU", GELU()),
        ("Mish", Mish()),
        ("Identity", Identity()),
    ]
    
    print("\nActivation Outputs:")
    print("-" * 60)
    
    for name, activation in activations:
        output = activation.forward(x)
        print(f"{name:12s}: {output}")
    
    # Softmax example
    print("\nSoftmax (multi-class):")
    logits = np.array([[1, 2, 3], [4, 5, 6]])
    softmax = Softmax()
    probs = softmax.forward(logits)
    print(f"Logits:\n{logits}")
    print(f"Probabilities:\n{probs}")
    print(f"Sum of probabilities: {probs.sum(axis=1)}")
    
    # Gradient demonstration
    print("\nGradient Flow (ReLU):")
    relu = ReLU()
    x_grad = np.array([-1, 0, 1, 2])
    output = relu.forward(x_grad)
    grad_upstream = np.ones_like(x_grad)
    grad_input = relu.backward(grad_upstream)
    print(f"Input: {x_grad}")
    print(f"Output: {output}")
    print(f"Gradient: {grad_input}")
    
    print("\n" + "=" * 60)
