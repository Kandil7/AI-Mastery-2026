"""
Neural Network Layers Module.

This module provides comprehensive neural network layer implementations,
including Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, and more.

Each layer includes:
- Forward pass computation
- Backward pass (gradient computation) for backpropagation
- Parameter initialization
- Weight updates

Example Usage:
    >>> import numpy as np
    >>> from layers import Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization
    >>> 
    >>> # Dense layer
    >>> dense = Dense(input_size=10, output_size=5)
    >>> x = np.random.randn(32, 10)
    >>> output = dense.forward(x)
    >>> grad = dense.backward(np.random.randn(32, 5))
    >>> 
    >>> # Conv2D layer
    >>> conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
    >>> x = np.random.randn(32, 3, 28, 28)
    >>> output = conv.forward(x)
"""

from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
from numpy.typing import ArrayLike
import logging

logger = logging.getLogger(__name__)

ArrayLike2D = Union[np.ndarray, List]
ArrayLike4D = Union[np.ndarray, List]


class Layer:
    """Base class for neural network layers."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize layer.
        
        Args:
            name: Optional layer name.
        """
        self.name = name or self.__class__.__name__
        self._trainable = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
        
        Returns:
            np.ndarray: Output tensor.
        """
        raise NotImplementedError
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        raise NotImplementedError
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        return {}
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get parameter gradients."""
        return {}
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set trainable parameters."""
        pass
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        pass
    
    @property
    def trainable(self) -> bool:
        """Whether layer has trainable parameters."""
        return self._trainable
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow calling instance as function."""
        return self.forward(x)


class Dense(Layer):
    """
    Fully connected (Dense) layer.
    
    y = x @ W.T + b
    
    Properties:
    - Input: (batch_size, input_size)
    - Output: (batch_size, output_size)
    - Parameters: weight (output_size, input_size), bias (output_size,)
    
    Example:
        >>> dense = Dense(input_size=10, output_size=5)
        >>> x = np.random.randn(32, 10)
        >>> output = dense.forward(x)
        >>> output.shape
        (32, 5)
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_bias: bool = True,
        weight_init: str = 'he',
        name: Optional[str] = None
    ):
        """
        Initialize Dense layer.
        
        Args:
            input_size: Number of input features.
            output_size: Number of output features.
            use_bias: Whether to use bias. Default: True.
            weight_init: Weight initialization ('he', 'xavier', 'normal').
            name: Layer name.
        """
        super().__init__(name)
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        # Initialize weights
        self.weight = self._init_weights(input_size, output_size, weight_init)
        if use_bias:
            self.bias = np.zeros(output_size)
        
        # Gradients
        self.grad_weight = np.zeros_like(self.weight)
        if use_bias:
            self.grad_bias = np.zeros_like(self.bias)
        
        # Cache for backward
        self._input: Optional[np.ndarray] = None
        
        logger.debug(f"Dense layer: {input_size} -> {output_size}, init={weight_init}")
    
    def _init_weights(
        self,
        fan_in: int,
        fan_out: int,
        method: str
    ) -> np.ndarray:
        """Initialize weights using specified method."""
        if method == 'he':
            # He initialization for ReLU
            std = np.sqrt(2.0 / fan_in)
        elif method == 'xavier':
            # Xavier/Glorot for tanh/sigmoid
            std = np.sqrt(2.0 / (fan_in + fan_out))
        elif method == 'normal':
            std = 0.01
        else:
            raise ValueError(f"Unknown initialization: {method}")
        
        return np.random.randn(fan_out, fan_in) * std
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = x @ W.T + b
        
        Args:
            x: Input (batch_size, input_size).
        
        Returns:
            np.ndarray: Output (batch_size, output_size).
        """
        x = np.asarray(x, dtype=np.float64)
        self._input = x
        
        output = x @ self.weight.T
        if self.use_bias:
            output += self.bias
        
        logger.debug(f"Dense forward: {x.shape} -> {output.shape}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        
        dL/dx = dL/dy @ W
        dL/dW = dL/dy.T @ x
        dL/db = sum(dL/dy, axis=0)
        
        Args:
            grad_output: Gradient (batch_size, output_size).
        
        Returns:
            np.ndarray: Gradient (batch_size, input_size).
        """
        if self._input is None:
            raise ValueError("Must call forward() before backward()")
        
        batch_size = self._input.shape[0]
        
        # Gradient w.r.t. input
        grad_input = grad_output @ self.weight
        
        # Gradient w.r.t. weights
        self.grad_weight = grad_output.T @ self._input
        
        # Gradient w.r.t. bias
        if self.use_bias:
            self.grad_bias = np.sum(grad_output, axis=0)
        
        logger.debug(f"Dense backward: grad_input shape {grad_input.shape}")
        return grad_input
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        params = {'weight': self.weight}
        if self.use_bias:
            params['bias'] = self.bias
        return params
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get parameter gradients."""
        grads = {'weight': self.grad_weight}
        if self.use_bias:
            grads['bias'] = self.grad_bias
        return grads
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set trainable parameters."""
        self.weight = params['weight']
        if self.use_bias and 'bias' in params:
            self.bias = params['bias']
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.grad_weight = np.zeros_like(self.weight)
        if self.use_bias:
            self.grad_bias = np.zeros_like(self.bias)


class Conv2D(Layer):
    """
    2D Convolutional layer.
    
    Properties:
    - Input: (batch_size, in_channels, height, width)
    - Output: (batch_size, out_channels, out_height, out_width)
    - Parameters: weight (out_channels, in_channels, kernel_size, kernel_size)
    
    Example:
        >>> conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        >>> x = np.random.randn(32, 3, 28, 28)
        >>> output = conv.forward(x)
        >>> output.shape
        (32, 16, 28, 28)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        use_bias: bool = True,
        weight_init: str = 'he',
        name: Optional[str] = None
    ):
        """
        Initialize Conv2D layer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of convolution kernel.
            stride: Convolution stride. Default: 1.
            padding: Zero padding. Default: 0.
            use_bias: Whether to use bias. Default: True.
            weight_init: Weight initialization method.
            name: Layer name.
        """
        super().__init__(name)
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        
        # Initialize weights
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = out_channels * self.kernel_size[0] * self.kernel_size[1]
        
        if weight_init == 'he':
            std = np.sqrt(2.0 / fan_in)
        elif weight_init == 'xavier':
            std = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            std = 0.01
        
        self.weight = np.random.randn(out_channels, in_channels, 
                                      self.kernel_size[0], self.kernel_size[1]) * std
        
        if use_bias:
            self.bias = np.zeros(out_channels)
        
        # Gradients
        self.grad_weight = np.zeros_like(self.weight)
        if use_bias:
            self.grad_bias = np.zeros_like(self.bias)
        
        # Cache
        self._input: Optional[np.ndarray] = None
        self._padded_input: Optional[np.ndarray] = None
        
        logger.debug(f"Conv2D: {in_channels} -> {out_channels}, "
                    f"kernel={self.kernel_size}, stride={stride}, padding={padding}")
    
    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        """Apply zero padding to input."""
        if self.padding == 0:
            return x
        
        return np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                         (self.padding, self.padding)), mode='constant')
    
    def _im2col(self, x: np.ndarray) -> np.ndarray:
        """
        Convert input to column matrix for efficient convolution.
        
        Transforms sliding windows into columns.
        """
        batch_size, channels, height, width = x.shape
        kh, kw = self.kernel_size
        
        out_height = (height + 2 * self.padding - kh) // self.stride + 1
        out_width = (width + 2 * self.padding - kw) // self.stride + 1
        
        # Pad input
        x_padded = self._pad_input(x)
        self._padded_input = x_padded
        
        # Create column matrix
        col = np.zeros((batch_size, channels * kh * kw, out_height * out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + kh
                w_start = j * self.stride
                w_end = w_start + kw
                
                col[:, :, i * out_width + j] = x_padded[:, :, h_start:h_end, w_start:w_end].reshape(
                    batch_size, -1
                )
        
        return col
    
    def _col2im(self, col: np.ndarray, output_shape: Tuple[int, int, int, int]) -> np.ndarray:
        """Convert column matrix back to image format."""
        batch_size, channels, height, width = output_shape
        kh, kw = self.kernel_size
        
        out_height = (height + 2 * self.padding - kh) // self.stride + 1
        out_width = (width + 2 * self.padding - kw) // self.stride + 1
        
        x_padded = np.zeros((batch_size, channels, height + 2 * self.padding, 
                            width + 2 * self.padding))
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + kh
                w_start = j * self.stride
                w_end = w_start + kw
                
                x_padded[:, :, h_start:h_end, w_start:w_end] += col[
                    :, :, i * out_width + j
                ].reshape(batch_size, channels, kh, kw)
        
        if self.padding > 0:
            return x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return x_padded
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input (batch_size, in_channels, height, width).
        
        Returns:
            np.ndarray: Output (batch_size, out_channels, out_height, out_width).
        """
        x = np.asarray(x, dtype=np.float64)
        self._input = x
        
        batch_size, channels, height, width = x.shape
        
        out_height = (height + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        # Use im2col for efficient convolution
        col = self._im2col(x)
        self._col = col
        
        # Reshape weight for matrix multiplication
        weight_col = self.weight.reshape(self.out_channels, -1)
        
        # Convolution as matrix multiplication
        output = np.zeros((batch_size, self.out_channels, out_height * out_width))
        for b in range(batch_size):
            output[b] = weight_col @ col[b]
        
        if self.use_bias:
            output += self.bias.reshape(1, -1, 1)
        
        output = output.reshape(batch_size, self.out_channels, out_height, out_width)
        
        logger.debug(f"Conv2D forward: {x.shape} -> {output.shape}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            grad_output: Gradient (batch_size, out_channels, out_height, out_width).
        
        Returns:
            np.ndarray: Gradient (batch_size, in_channels, height, width).
        """
        if self._input is None:
            raise ValueError("Must call forward() before backward()")
        
        batch_size = self._input.shape[0]
        
        # Reshape gradient
        grad_output_flat = grad_output.reshape(batch_size, self.out_channels, -1)
        
        # Gradient w.r.t. weights
        for b in range(batch_size):
            self.grad_weight += (grad_output_flat[b] @ self._col[b].T).reshape(
                self.weight.shape
            )
        
        # Gradient w.r.t. bias
        if self.use_bias:
            self.grad_bias = np.sum(grad_output, axis=(0, 2, 3))
        
        # Gradient w.r.t. input
        weight_col = self.weight.reshape(self.out_channels, -1)
        grad_col = np.zeros_like(self._col)
        for b in range(batch_size):
            grad_col[b] = weight_col.T @ grad_output_flat[b]
        
        grad_input = self._col2im(grad_col, self._input.shape)
        
        logger.debug(f"Conv2D backward: grad_input shape {grad_input.shape}")
        return grad_input
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        params = {'weight': self.weight}
        if self.use_bias:
            params['bias'] = self.bias
        return params
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get parameter gradients."""
        grads = {'weight': self.grad_weight}
        if self.use_bias:
            grads['bias'] = self.grad_bias
        return grads
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.grad_weight = np.zeros_like(self.weight)
        if self.use_bias:
            self.grad_bias = np.zeros_like(self.bias)


class MaxPool2D(Layer):
    """
    2D Max Pooling layer.
    
    Properties:
    - Input: (batch_size, channels, height, width)
    - Output: (batch_size, channels, out_height, out_width)
    - No trainable parameters
    
    Example:
        >>> pool = MaxPool2D(kernel_size=2, stride=2)
        >>> x = np.random.randn(32, 16, 28, 28)
        >>> output = pool.forward(x)
        >>> output.shape
        (32, 16, 14, 14)
    """
    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[int] = None,
        padding: int = 0,
        name: Optional[str] = None
    ):
        """
        Initialize MaxPool2D.
        
        Args:
            kernel_size: Size of pooling window.
            stride: Pooling stride. If None, equals kernel_size.
            padding: Zero padding. Default: 0.
            name: Layer name.
        """
        super().__init__(name)
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        self.stride = stride if stride is not None else self.kernel_size
        self.padding = padding
        
        self._input: Optional[np.ndarray] = None
        self._max_indices: Optional[np.ndarray] = None
        
        logger.debug(f"MaxPool2D: kernel={self.kernel_size}, stride={self.stride}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: max pooling.
        
        Args:
            x: Input (batch_size, channels, height, width).
        
        Returns:
            np.ndarray: Output (batch_size, channels, out_height, out_width).
        """
        x = np.asarray(x, dtype=np.float64)
        self._input = x
        
        batch_size, channels, height, width = x.shape
        
        out_height = (height + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        # Pad input
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                          (self.padding, self.padding)), mode='constant')
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        self._max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride
                w_end = w_start + self.kernel_size[1]
                
                window = x[:, :, h_start:h_end, w_start:w_end]
                window_flat = window.reshape(batch_size, channels, -1)
                
                max_idx_flat = np.argmax(window_flat, axis=2)
                max_idx_h = max_idx_flat // self.kernel_size[1]
                max_idx_w = max_idx_flat % self.kernel_size[1]
                
                output[:, :, i, j] = np.max(window, axis=(3, 4))
                self._max_indices[:, :, i, j, 0] = h_start + max_idx_h
                self._max_indices[:, :, i, j, 1] = w_start + max_idx_w
        
        logger.debug(f"MaxPool2D forward: {x.shape} -> {output.shape}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: route gradient only through max elements.
        
        Args:
            grad_output: Gradient.
        
        Returns:
            np.ndarray: Gradient w.r.t. input.
        """
        if self._input is None:
            raise ValueError("Must call forward() before backward()")
        
        batch_size, channels, out_height, out_width = grad_output.shape
        _, _, height, width = self._input.shape
        
        grad_input = np.zeros_like(self._input)
        
        for i in range(out_height):
            for j in range(out_width):
                b_idx = np.arange(batch_size)[:, None, None]
                c_idx = np.arange(channels)[None, :, None]
                
                h_idx = self._max_indices[:, :, i, j, 0]
                w_idx = self._max_indices[:, :, i, j, 1]
                
                grad_input[b_idx, c_idx, h_idx, w_idx] += grad_output[:, :, i, j][:, :, None]
        
        logger.debug(f"MaxPool2D backward: grad_input shape {grad_input.shape}")
        return grad_input


class Dropout(Layer):
    """
    Dropout regularization layer.
    
    Randomly sets input elements to zero during training.
    
    Properties:
    - Scales output by 1/(1-p) during training (inverted dropout)
    - No operation during inference
    
    Example:
        >>> dropout = Dropout(p=0.5)
        >>> x = np.random.randn(32, 100)
        >>> output_train = dropout.forward(x, training=True)
        >>> output_eval = dropout.forward(x, training=False)
    """
    
    def __init__(self, p: float = 0.5, name: Optional[str] = None):
        """
        Initialize Dropout.
        
        Args:
            p: Dropout probability. Default: 0.5.
            name: Layer name.
        """
        super().__init__(name)
        self.p = p  # Probability of dropping
        self._mask: Optional[np.ndarray] = None
        self._training = True
        
        logger.debug(f"Dropout: p={p}")
    
    def forward(
        self,
        x: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            training: Whether in training mode.
        
        Returns:
            np.ndarray: Output tensor.
        """
        x = np.asarray(x, dtype=np.float64)
        self._training = training
        
        if not training or self.p == 0:
            return x
        
        # Inverted dropout
        self._mask = (np.random.rand(*x.shape) > self.p).astype(np.float64)
        output = x * self._mask / (1 - self.p)
        
        logger.debug(f"Dropout forward: dropped {100 * self.p}% of neurons")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: route gradient only through non-dropped neurons.
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient w.r.t. input.
        """
        if not self._training or self.p == 0:
            return grad_output
        
        if self._mask is None:
            raise ValueError("Must call forward() before backward()")
        
        grad_input = grad_output * self._mask / (1 - self.p)
        
        logger.debug(f"Dropout backward: gradient shape {grad_input.shape}")
        return grad_input


class BatchNormalization(Layer):
    """
    Batch Normalization layer.
    
    Normalizes activations to have zero mean and unit variance,
    then applies learnable scale and shift.
    
    y = γ * (x - μ) / √(σ² + ε) + β
    
    Properties:
    - Stabilizes training
    - Allows higher learning rates
    - Has running statistics for inference
    
    Example:
        >>> bn = BatchNormalization(num_features=100)
        >>> x = np.random.randn(32, 100)
        >>> output = bn.forward(x, training=True)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize BatchNormalization.
        
        Args:
            num_features: Number of features/channels.
            eps: Small value for numerical stability.
            momentum: Momentum for running statistics.
            affine: Whether to use learnable affine transform.
            name: Layer name.
        """
        super().__init__(name)
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # Learnable parameters
        if affine:
            self.weight = np.ones(num_features)
            self.bias = np.zeros(num_features)
            self.grad_weight = np.zeros(num_features)
            self.grad_bias = np.zeros(num_features)
        
        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache for backward
        self._input: Optional[np.ndarray] = None
        self._normalized: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._var: Optional[np.ndarray] = None
        
        logger.debug(f"BatchNormalization: {num_features} features, eps={eps}")
    
    def forward(
        self,
        x: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            training: Whether in training mode.
        
        Returns:
            np.ndarray: Normalized output.
        """
        x = np.asarray(x, dtype=np.float64)
        
        if training:
            # Compute batch statistics
            self._mean = np.mean(x, axis=0)
            self._var = np.var(x, axis=0)
            self._std = np.sqrt(self._var + self.eps)
            
            # Normalize
            self._normalized = (x - self._mean) / self._std
            self._input = x
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * self._mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * self._var
        else:
            # Use running statistics
            self._std = np.sqrt(self.running_var + self.eps)
            self._normalized = (x - self.running_mean) / self._std
        
        # Apply affine transform
        if self.affine:
            output = self.weight * self._normalized + self.bias
        else:
            output = self._normalized
        
        logger.debug(f"BatchNorm forward: mean={self._normalized.mean():.6f}, "
                    f"std={self._normalized.std():.6f}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient w.r.t. input.
        """
        if self._input is None:
            raise ValueError("Must call forward() before backward()")
        
        batch_size = self._input.shape[0]
        
        # Gradient w.r.t. affine parameters
        if self.affine:
            self.grad_weight = np.sum(grad_output * self._normalized, axis=0)
            self.grad_bias = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t. normalized input
        if self.affine:
            grad_normalized = grad_output * self.weight
        else:
            grad_normalized = grad_output
        
        # Gradient w.r.t. input (batch norm backward)
        grad_input = (1 / (batch_size * self._std)) * (
            batch_size * grad_normalized -
            np.sum(grad_normalized, axis=0) -
            self._normalized * np.sum(grad_normalized * self._normalized, axis=0)
        )
        
        logger.debug(f"BatchNorm backward: gradient shape {grad_input.shape}")
        return grad_input
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        if self.affine:
            return {'weight': self.weight, 'bias': self.bias}
        return {}
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get parameter gradients."""
        if self.affine:
            return {'weight': self.grad_weight, 'bias': self.grad_bias}
        return {}
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        if self.affine:
            self.grad_weight = np.zeros_like(self.weight)
            self.grad_bias = np.zeros_like(self.bias)


class Flatten(Layer):
    """
    Flatten layer to convert multi-dimensional input to 2D.
    
    Example:
        >>> flatten = Flatten()
        >>> x = np.random.randn(32, 16, 7, 7)
        >>> output = flatten.forward(x)
        >>> output.shape
        (32, 784)
    """
    
    def __init__(self, start_dim: int = 1, name: Optional[str] = None):
        """
        Initialize Flatten.
        
        Args:
            start_dim: First dimension to flatten. Default: 1.
            name: Layer name.
        """
        super().__init__(name)
        self.start_dim = start_dim
        self._input_shape: Optional[Tuple] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: flatten input.
        
        Args:
            x: Input tensor.
        
        Returns:
            np.ndarray: Flattened output.
        """
        x = np.asarray(x, dtype=np.float64)
        self._input_shape = x.shape
        
        batch_size = x.shape[0]
        output = x.reshape(batch_size, -1)
        
        logger.debug(f"Flatten forward: {x.shape} -> {output.shape}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: reshape gradient to original shape.
        
        Args:
            grad_output: Gradient.
        
        Returns:
            np.ndarray: Reshaped gradient.
        """
        if self._input_shape is None:
            raise ValueError("Must call forward() before backward()")
        
        grad_input = grad_output.reshape(self._input_shape)
        
        logger.debug(f"Flatten backward: {grad_output.shape} -> {grad_input.shape}")
        return grad_input


class Reshape(Layer):
    """
    Reshape layer to change tensor shape.
    
    Example:
        >>> reshape = Reshape(target_shape=(32, 3, 32, 32))
        >>> x = np.random.randn(32, 3072)
        >>> output = reshape.forward(x)
        >>> output.shape
        (32, 3, 32, 32)
    """
    
    def __init__(
        self,
        target_shape: Tuple[int, ...],
        name: Optional[str] = None
    ):
        """
        Initialize Reshape.
        
        Args:
            target_shape: Target shape.
            name: Layer name.
        """
        super().__init__(name)
        self.target_shape = target_shape
        self._input_shape: Optional[Tuple] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: reshape input.
        
        Args:
            x: Input tensor.
        
        Returns:
            np.ndarray: Reshaped output.
        """
        x = np.asarray(x, dtype=np.float64)
        self._input_shape = x.shape
        
        output = x.reshape(self.target_shape)
        
        logger.debug(f"Reshape forward: {x.shape} -> {output.shape}")
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: reshape gradient to original shape.
        
        Args:
            grad_output: Gradient.
        
        Returns:
            np.ndarray: Reshaped gradient.
        """
        if self._input_shape is None:
            raise ValueError("Must call forward() before backward()")
        
        grad_input = grad_output.reshape(self._input_shape)
        
        logger.debug(f"Reshape backward: {grad_output.shape} -> {grad_input.shape}")
        return grad_input


class Sequential:
    """
    Sequential container for stacking layers.
    
    Example:
        >>> model = Sequential([
        ...     Dense(784, 128),
        ...     ReLU(),
        ...     Dense(128, 10),
        ... ])
        >>> x = np.random.randn(32, 784)
        >>> output = model.forward(x)
    """
    
    def __init__(self, layers: List[Layer]):
        """
        Initialize Sequential model.
        
        Args:
            layers: List of layers in order.
        """
        self.layers = layers
    
    def forward(
        self,
        x: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass through all layers.
        
        Args:
            x: Input tensor.
            training: Whether in training mode.
        
        Returns:
            np.ndarray: Output tensor.
        """
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization)):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through all layers (in reverse order).
        
        Args:
            grad_output: Gradient from loss.
        
        Returns:
            np.ndarray: Gradient w.r.t. input.
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def get_parameters(self) -> List[Dict[str, np.ndarray]]:
        """Get all trainable parameters."""
        return [layer.get_parameters() for layer in self.layers if layer.trainable]
    
    def get_gradients(self) -> List[Dict[str, np.ndarray]]:
        """Get all parameter gradients."""
        return [layer.get_gradients() for layer in self.layers if layer.trainable]
    
    def zero_grad(self) -> None:
        """Zero out all gradients."""
        for layer in self.layers:
            layer.zero_grad()
    
    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Allow calling instance as function."""
        return self.forward(x, training=training)


# Import activations for convenience
from .activations import ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish, GELU, Softmax, Identity

if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Neural Network Layers Module - Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Dense layer
    print("\n1. Dense Layer:")
    dense = Dense(input_size=10, output_size=5)
    x = np.random.randn(32, 10)
    output = dense.forward(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Weight shape: {dense.weight.shape}")
    
    grad = dense.backward(np.random.randn(32, 5))
    print(f"   Gradient shape: {grad.shape}")
    
    # Conv2D layer
    print("\n2. Conv2D Layer:")
    conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    x = np.random.randn(32, 3, 28, 28)
    output = conv.forward(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # MaxPool2D layer
    print("\n3. MaxPool2D Layer:")
    pool = MaxPool2D(kernel_size=2, stride=2)
    x = np.random.randn(32, 16, 28, 28)
    output = pool.forward(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Dropout
    print("\n4. Dropout Layer:")
    dropout = Dropout(p=0.5)
    x = np.ones((32, 100))
    output_train = dropout.forward(x, training=True)
    output_eval = dropout.forward(x, training=False)
    print(f"   Training mode: {output_train.mean():.4f} (should be ~1.0)")
    print(f"   Eval mode: {output_eval.mean():.4f} (should be 1.0)")
    
    # BatchNormalization
    print("\n5. BatchNormalization Layer:")
    bn = BatchNormalization(num_features=100)
    x = np.random.randn(32, 100) * 5 + 10  # Non-standardized
    output = bn.forward(x, training=True)
    print(f"   Input: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"   Output: mean={output.mean():.6f}, std={output.std():.6f}")
    
    # Sequential model
    print("\n6. Sequential Model:")
    model = Sequential([
        Dense(784, 128, name='fc1'),
        ReLU(),
        Dropout(0.5),
        Dense(128, 10, name='fc2'),
    ])
    x = np.random.randn(32, 784)
    output = model.forward(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n" + "=" * 60)
