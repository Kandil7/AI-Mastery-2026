"""
Convolutional Neural Networks Implementation

This module implements CNNs from scratch using NumPy,
including convolution, pooling, and common CNN architectures.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import matplotlib.pyplot as plt


def conv2d_forward(
    input_tensor: np.ndarray, 
    filters: np.ndarray, 
    bias: np.ndarray, 
    stride: int = 1, 
    padding: int = 0
) -> np.ndarray:
    """
    Forward pass for 2D convolution.
    
    Args:
        input_tensor: Input tensor of shape (batch_size, height, width, channels)
        filters: Filter tensor of shape (filter_height, filter_width, input_channels, num_filters)
        bias: Bias vector of shape (num_filters,)
        stride: Stride for convolution
        padding: Padding for input
        
    Returns:
        Output tensor after convolution
    """
    batch_size, h_in, w_in, c_in = input_tensor.shape
    h_f, w_f, c_f, n_f = filters.shape
    
    # Verify dimensions
    assert c_in == c_f, "Input channels must match filter channels"
    
    # Calculate output dimensions
    h_out = int((h_in + 2 * padding - h_f) / stride + 1)
    w_out = int((w_in + 2 * padding - w_f) / stride + 1)
    
    # Initialize output tensor
    output = np.zeros((batch_size, h_out, w_out, n_f))
    
    # Add padding to input
    if padding > 0:
        input_padded = np.pad(input_tensor, 
                             ((0, 0), (padding, padding), (padding, padding), (0, 0)), 
                             mode='constant')
    else:
        input_padded = input_tensor
    
    # Perform convolution
    for i in range(batch_size):
        for f in range(n_f):
            for h in range(h_out):
                for w in range(w_out):
                    h_start = h * stride
                    h_end = h_start + h_f
                    w_start = w * stride
                    w_end = w_start + w_f
                    
                    # Extract region of interest
                    roi = input_padded[i, h_start:h_end, w_start:w_end, :]
                    
                    # Perform element-wise multiplication and sum
                    output[i, h, w, f] = np.sum(roi * filters[:, :, :, f]) + bias[f]
    
    return output


def conv2d_backward(
    dout: np.ndarray,
    input_tensor: np.ndarray,
    filters: np.ndarray,
    stride: int = 1,
    padding: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for 2D convolution.
    
    Args:
        dout: Gradient of loss w.r.t. output
        input_tensor: Input tensor to forward pass
        filters: Filters used in forward pass
        stride: Stride used in forward pass
        padding: Padding used in forward pass
        
    Returns:
        Tuple of (grad_input, grad_filters, grad_bias)
    """
    batch_size, h_in, w_in, c_in = input_tensor.shape
    h_f, w_f, c_f, n_f = filters.shape
    batch_size_out, h_out, w_out, n_out = dout.shape
    
    # Initialize gradients
    grad_input = np.zeros_like(input_tensor)
    grad_filters = np.zeros_like(filters)
    grad_bias = np.zeros(n_f)
    
    # Add padding to grad_input
    if padding > 0:
        grad_input_padded = np.pad(grad_input, 
                                  ((0, 0), (padding, padding), (padding, padding), (0, 0)), 
                                  mode='constant')
        input_padded = np.pad(input_tensor, 
                             ((0, 0), (padding, padding), (padding, padding), (0, 0)), 
                             mode='constant')
    else:
        grad_input_padded = grad_input
        input_padded = input_tensor
    
    # Calculate gradients
    for i in range(batch_size):
        for f in range(n_f):
            for h in range(h_out):
                for w in range(w_out):
                    h_start = h * stride
                    h_end = h_start + h_f
                    w_start = w * stride
                    w_end = w_start + w_f
                    
                    # Gradient w.r.t. filters
                    roi = input_padded[i, h_start:h_end, w_start:w_end, :]
                    grad_filters[:, :, :, f] += roi * dout[i, h, w, f]
                    
                    # Gradient w.r.t. input
                    grad_input_padded[i, h_start:h_end, w_start:w_end, :] += (
                        filters[:, :, :, f] * dout[i, h, w, f]
                    )
    
    # Remove padding from grad_input
    if padding > 0:
        grad_input = grad_input_padded[:, padding:-padding, padding:-padding, :]
    else:
        grad_input = grad_input_padded
    
    # Gradient w.r.t. bias
    grad_bias = np.sum(dout, axis=(0, 1, 2))
    
    return grad_input, grad_filters, grad_bias


def max_pool2d_forward(
    input_tensor: np.ndarray, 
    pool_size: int = 2, 
    stride: int = 2
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Forward pass for 2D max pooling.
    
    Args:
        input_tensor: Input tensor of shape (batch_size, height, width, channels)
        pool_size: Size of pooling window
        stride: Stride for pooling
        
    Returns:
        Tuple of (output tensor, mask for backward pass)
    """
    batch_size, h_in, w_in, c_in = input_tensor.shape
    
    # Calculate output dimensions
    h_out = int((h_in - pool_size) / stride + 1)
    w_out = int((w_in - pool_size) / stride + 1)
    
    # Initialize output and mask
    output = np.zeros((batch_size, h_out, w_out, c_in))
    mask = np.zeros_like(input_tensor)
    
    # Perform max pooling
    for i in range(batch_size):
        for c in range(c_in):
            for h in range(h_out):
                for w in range(w_out):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size
                    
                    # Extract region of interest
                    roi = input_tensor[i, h_start:h_end, w_start:w_end, c]
                    
                    # Find max value and its position
                    max_val = np.max(roi)
                    output[i, h, w, c] = max_val
                    
                    # Create mask for backward pass
                    mask[i, h_start:h_end, w_start:w_end, c] += (
                        roi == max_val
                    ).astype(float)
    
    return output, mask


def max_pool2d_backward(
    dout: np.ndarray, 
    mask: np.ndarray, 
    pool_size: int = 2, 
    stride: int = 2
) -> np.ndarray:
    """
    Backward pass for 2D max pooling.
    
    Args:
        dout: Gradient of loss w.r.t. output
        mask: Mask from forward pass
        pool_size: Size of pooling window
        stride: Stride for pooling
        
    Returns:
        Gradient w.r.t. input
    """
    batch_size, h_out, w_out, c_out = dout.shape
    h_in, w_in, c_in = mask.shape[1], mask.shape[2], mask.shape[3]
    
    # Initialize gradient
    grad_input = np.zeros((batch_size, h_in, w_in, c_in))
    
    # Distribute gradients using mask
    for i in range(batch_size):
        for c in range(c_out):
            for h in range(h_out):
                for w in range(w_out):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size
                    
                    # Distribute gradient to positions that were max
                    grad_input[i, h_start:h_end, w_start:w_end, c] += (
                        mask[i, h_start:h_end, w_start:w_end, c] * dout[i, h, w, c]
                    )
    
    return grad_input


class Conv2D:
    """
    2D Convolutional Layer implementation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        activation: str = 'relu'
    ):
        """
        Initialize Conv2D layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride for convolution
            padding: Padding for input
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'linear')
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        self.stride = stride
        self.padding = padding
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = np.random.randn(
            self.kernel_size[0], 
            self.kernel_size[1], 
            in_channels, 
            out_channels
        ) * np.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        
        self.bias = np.zeros(out_channels)
        
        # Store intermediate values for backward pass
        self.input = None
        self.z = None  # Pre-activation values
        self.a = None  # Post-activation values
    
    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _apply_activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """Apply derivative of activation function."""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sigmoid_z = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return sigmoid_z * (1 - sigmoid_z)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation == 'linear':
            return np.ones_like(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            input_tensor: Input tensor of shape (batch_size, height, width, channels)
            
        Returns:
            Output tensor after convolution and activation
        """
        self.input = input_tensor
        
        # Convolution
        self.z = conv2d_forward(
            input_tensor, 
            self.weights, 
            self.bias, 
            self.stride, 
            self.padding
        )
        
        # Apply activation
        self.a = self._apply_activation(self.z)
        
        return self.a
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.
        
        Args:
            dout: Gradient of loss w.r.t. output of this layer
            
        Returns:
            Gradient of loss w.r.t. input of this layer
        """
        # Apply derivative of activation
        dout = dout * self._apply_activation_derivative(self.z)
        
        # Compute gradients
        grad_input, grad_weights, grad_bias = conv2d_backward(
            dout, 
            self.input, 
            self.weights, 
            self.stride, 
            self.padding
        )
        
        # Update parameters (in a real implementation, this would be done by an optimizer)
        self.weights -= 0.01 * grad_weights
        self.bias -= 0.01 * grad_bias
        
        return grad_input


class MaxPool2D:
    """
    2D Max Pooling Layer implementation.
    """
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        """
        Initialize MaxPool2D layer.
        
        Args:
            pool_size: Size of pooling window
            stride: Stride for pooling
        """
        self.pool_size = pool_size
        self.stride = stride
        self.mask = None
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            input_tensor: Input tensor of shape (batch_size, height, width, channels)
            
        Returns:
            Output tensor after max pooling
        """
        output, self.mask = max_pool2d_forward(
            input_tensor, 
            self.pool_size, 
            self.stride
        )
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.
        
        Args:
            dout: Gradient of loss w.r.t. output of this layer
            
        Returns:
            Gradient of loss w.r.t. input of this layer
        """
        return max_pool2d_backward(
            dout, 
            self.mask, 
            self.pool_size, 
            self.stride
        )


class Flatten:
    """
    Flatten layer to convert multi-dimensional tensors to 1D.
    """
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            input_tensor: Input tensor of any shape except batch dimension
            
        Returns:
            Flattened tensor of shape (batch_size, -1)
        """
        self.input_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        flattened = input_tensor.reshape(batch_size, -1)
        return flattened
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.
        
        Args:
            dout: Gradient of loss w.r.t. output of this layer
            
        Returns:
            Gradient of loss w.r.t. input of this layer
        """
        return dout.reshape(self.input_shape)


class CNN:
    """
    Convolutional Neural Network implementation.
    """
    
    def __init__(self, layers: List):
        """
        Initialize CNN with a list of layers.
        
        Args:
            layers: List of layer objects (Conv2D, MaxPool2D, Flatten, Dense, etc.)
        """
        self.layers = layers
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            X: Input tensor of shape (batch_size, height, width, channels)
            
        Returns:
            Output tensor after passing through all layers
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, dout: np.ndarray):
        """
        Backward pass through the network.
        
        Args:
            dout: Gradient of loss w.r.t. output of the network
        """
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input tensor of shape (batch_size, height, width, channels)
            
        Returns:
            Predictions
        """
        return self.forward(X)


def create_cnn_classifier(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    conv_layers: List[Tuple[int, int, int]] = [(32, 3, 1), (64, 3, 1)],
    dense_units: List[int] = [128]
) -> CNN:
    """
    Create a CNN classifier with specified architecture.
    
    Args:
        input_shape: Shape of input (height, width, channels)
        num_classes: Number of output classes
        conv_layers: List of (filters, kernel_size, stride) tuples for conv layers
        dense_units: List of units for dense layers after flattening
        
    Returns:
        CNN object
    """
    layers = []
    
    # Add input shape information
    current_shape = (None,) + input_shape  # (batch_size, height, width, channels)
    
    # Add convolutional layers
    for filters, kernel_size, stride in conv_layers:
        layers.append(
            Conv2D(
                in_channels=current_shape[3] if len(layers) == 0 else layers[-1].out_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                activation='relu'
            )
        )
        layers.append(MaxPool2D(pool_size=2, stride=2))
    
    # Add flatten layer
    layers.append(Flatten())
    
    # Add dense layers
    # For the first dense layer, we need to calculate the flattened size
    # This is a simplified calculation - in practice, you'd need to compute the exact size
    # based on the conv and pooling operations
    final_conv_size = 1  # Placeholder - would need to compute based on architecture
    for i, units in enumerate(dense_units):
        if i == 0:
            # Calculate the size after flattening
            # This is a simplified approach - in practice, you'd need to track the dimensions
            pass
        # In a real implementation, we would add Dense layers here
        # For now, we'll just add the final classification layer
    
    return CNN(layers)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate cross-entropy loss.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        
    Returns:
        Cross-entropy loss
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))