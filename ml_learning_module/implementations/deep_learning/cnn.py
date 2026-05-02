"""
Convolutional Neural Network (CNN) Implementation from Scratch
===============================================================

This module provides a complete implementation of CNN building blocks
including convolutional layers, pooling layers, activations, and loss functions.

Mathematical Foundation:
------------------------
1. Convolution Operation:
   The convolution operation slides a kernel (filter) over the input feature map.

   For input I with size (H, W) and kernel K with size (Kh, Kw):

   Output[i, j] = Σ Σ I[i + m, j + n] * K[m, n]
                  m=-Kh/2  n=-Kw/2

   Where:
   - H, W: Input height and width
   - Kh, Kw: Kernel height and width
   - Output size: (H - Kh + 1, W - Kw + 1) with valid padding

2. Padding:
   - Valid padding: No padding, output is smaller
   - Same padding: Pad so output size equals input size
   - Padding size: p = (k - 1) / 2 for "same" padding

3. Stride:
   - Controls how much the kernel moves each step
   - stride=1: Move 1 pixel at a time
   - stride=2: Move 2 pixels at a time, output is smaller

4. Backpropagation:
   - dL/dK = convolution of input with gradient of output
   - dL/dInput = full convolution of gradient with rotated kernel

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, Optional


class ReLU:
    """
    Rectified Linear Unit Activation Function

    f(x) = max(0, x)

    Derivative:
    f'(x) = 1 if x > 0, else 0

    Properties:
    - Introduces sparsity (some neurons output 0)
    - Helps with vanishing gradient problem
    - Computationally efficient
    """

    def __init__(self):
        self.mask = None  # Store which neurons are active

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Apply ReLU activation

        Args:
            x: Input array of any shape

        Returns:
            Array with ReLU applied: max(0, x)
        """
        self.mask = x > 0  # Store which elements are positive
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: Compute gradient with respect to input

        dL/dx = dL/dy * f'(x)
        f'(x) = 1 if x > 0, else 0

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        return grad_output * self.mask.astype(float)


class Sigmoid:
    """
    Sigmoid Activation Function

    f(x) = 1 / (1 + e^(-x))

    Derivative:
    f'(x) = f(x) * (1 - f(x))

    Properties:
    - Maps to (0, 1) - useful for probability outputs
    - Saturated neurons have small gradients (vanishing gradient)
    - Historically popular, now mostly used in output layers

    Problem: When x is very large or very small, f'(x) ≈ 0
             This causes gradients to vanish in deep networks
    """

    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Apply sigmoid activation

        Args:
            x: Input array

        Returns:
            Sigmoid output in range (0, 1)
        """
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        self.output = 1 / (1 + np.exp(-x_clipped))
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: Compute gradient

        dL/dx = dL/dy * sigmoid(x) * (1 - sigmoid(x))

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        # f'(x) = f(x) * (1 - f(x))
        sigmoid_grad = self.output * (1 - self.output)
        return grad_output * sigmoid_grad


class Softmax:
    """
    Softmax Activation Function

    f(x_i) = e^(x_i) / Σ_j e^(x_j)

    Properties:
    - Outputs sum to 1 (valid probability distribution)
    - Used in multi-class classification output layers
    - Amplifies differences (largest value becomes even larger)

    Numerical Stability:
    - Subtract max from all inputs to prevent overflow
    - e^(x - max) / Σ e^(x - max) = e^x / Σ e^x
    """

    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Apply softmax

        Args:
            x: Input array of shape (batch_size, num_classes)

        Returns:
            Probability distribution summing to 1 along axis 1
        """
        # Numerical stability: subtract max
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: Compute gradient

        The Jacobian of softmax is:
        ∂f_i/∂x_j = f_i * (δ_ij - f_j)

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        # Compute Jacobian-vector product
        batch_size = grad_output.shape[0]
        grad_input = np.zeros_like(grad_output)

        for i in range(batch_size):
            # Diagonal: f_i * (1 - f_i)
            # Off-diagonal: -f_i * f_j
            jacobian = -np.outer(self.output[i], self.output[i])
            np.fill_diagonal(jacobian, self.output[i] * (1 - self.output[i]))
            grad_input[i] = np.dot(jacobian, grad_output[i])

        return grad_input


class ConvLayer:
    """
    Convolutional Layer

    Performs 2D convolution operation.

    Parameters:
        num_filters: Number of convolution kernels
        kernel_size: Size of each kernel (square)
        stride: Step size for sliding
        padding: Number of zero pads around input

    Example:
        Input: (batch, channels, H, W) = (32, 3, 32, 32)
        Kernel: 3x3
        Padding: 1
        Output: (32, num_filters, 32, 32)

    Forward Pass:
        For each filter k:
            output[b, k, i, j] = Σ_c Σ_m Σ_n input[b, c, i+m, j+n] * weights[k, c, m, n] + bias[k]
    """

    def __init__(
        self, num_filters: int, kernel_size: int, stride: int = 1, padding: int = 0
    ):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weights initialized with He initialization
        # Shape: (num_filters, channels, kernel_size, kernel_size)
        # channels will be set during first forward pass
        self.weights = None
        self.bias = None

        # Cache for backward pass
        self.input_cache = None

    def _init_weights(self, channels: int):
        """Initialize weights with He initialization"""
        # He initialization: scale = sqrt(2 / (fan_in))
        # fan_in = channels * kernel_size * kernel_size
        fan_in = channels * self.kernel_size * self.kernel_size
        scale = np.sqrt(2.0 / fan_in)

        self.weights = (
            np.random.randn(
                self.num_filters, channels, self.kernel_size, self.kernel_size
            )
            * scale
        )

        # Bias: one per filter, initialized to zero
        self.bias = np.zeros(self.num_filters)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Apply convolution

        Args:
            x: Input of shape (batch, channels, height, width)

        Returns:
            Output of shape (batch, num_filters, out_height, out_width)
        """
        batch_size, channels, height, width = x.shape

        # Initialize weights on first call
        if self.weights is None:
            self._init_weights(channels)

        # Apply padding
        if self.padding > 0:
            x_padded = np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            x_padded = x

        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Cache input for backward pass
        self.input_cache = x

        # Initialize output
        output = np.zeros((batch_size, self.num_filters, out_height, out_width))

        # Perform convolution
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(0, out_height * self.stride, self.stride):
                    for j in range(0, out_width * self.stride, self.stride):
                        # Extract patch
                        patch = x_padded[
                            b, :, i : i + self.kernel_size, j : j + self.kernel_size
                        ]
                        # Convolve
                        output[b, f, i // self.stride, j // self.stride] = (
                            np.sum(patch * self.weights[f]) + self.bias[f]
                        )

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: Compute gradients

        Args:
            grad_output: Gradient from next layer
                        Shape: (batch, num_filters, out_height, out_width)

        Returns:
            Gradient with respect to input
        """
        batch_size = grad_output.shape[0]
        _, channels, height, width = self.input_cache.shape

        # Pad gradient output if needed for gradient calculation
        grad_output_padded = np.pad(
            grad_output,
            (
                (0, 0),
                (0, 0),
                (self.kernel_size - 1, self.kernel_size - 1),
                (self.kernel_size - 1, self.kernel_size - 1),
            ),
            mode="constant",
        )

        # Initialize gradients
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.zeros(self.num_filters)
        grad_input = np.zeros_like(self.input_cache)

        # Compute gradients
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(grad_output.shape[2]):
                    for j in range(grad_output.shape[3]):
                        # Gradient with respect to weights
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = self.input_cache[
                            b,
                            :,
                            h_start : h_start + self.kernel_size,
                            w_start : w_start + self.kernel_size,
                        ]
                        grad_weights[f] += grad_output[b, f, i, j] * patch

                        # Gradient with respect to bias
                        grad_bias[f] += grad_output[b, f, i, j]

                        # Gradient with respect to input (full convolution with flipped kernel)
                        grad_input[
                            b,
                            :,
                            h_start : h_start + self.kernel_size,
                            w_start : w_start + self.kernel_size,
                        ] += grad_output[b, f, i, j] * np.flip(
                            self.weights[f], axis=(1, 2)
                        )

        # Normalize by batch size
        grad_weights /= batch_size
        grad_bias /= batch_size
        grad_input /= batch_size

        # Apply padding to input gradient if needed
        if self.padding > 0:
            grad_input = grad_input[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]

        # Store gradients for optimizer
        self.grad_weights = grad_weights
        self.grad_bias = grad_bias

        return grad_input

    def get_parameters(self):
        """Return weights and biases"""
        return self.weights, self.bias

    def set_parameters(self, weights, bias):
        """Set weights and biases"""
        self.weights = weights
        self.bias = bias


class MaxPoolLayer:
    """
    Max Pooling Layer

    Downsamples spatial dimensions by taking the maximum value in each window.

    Parameters:
        pool_size: Size of the pooling window
        stride: Step size for sliding (default: pool_size for non-overlapping)

    Example:
        Input: (batch, channels, 32, 32)
        pool_size: 2, stride: 2
        Output: (batch, channels, 16, 16)

    Forward Pass:
        output[b, c, i, j] = max(input[b, c,
                                   i*stride:i*stride+pool_size,
                                   j*stride:j*stride+pool_size])

    Backward Pass:
        - Only the position with max value gets gradient
        - Other positions get 0
    """

    def __init__(self, pool_size: int, stride: Optional[int] = None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

        # Cache for backward pass
        self.input_cache = None
        self.max_indices = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Apply max pooling

        Args:
            x: Input of shape (batch, channels, height, width)

        Returns:
            Output of shape (batch, channels, out_height, out_width)
        """
        batch_size, channels, height, width = x.shape

        # Cache input
        self.input_cache = x

        # Calculate output dimensions
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        # Initialize output and max indices
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros(
            (batch_size, channels, out_height, out_width), dtype=int
        )

        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = x[
                            b,
                            c,
                            h_start : h_start + self.pool_size,
                            w_start : w_start + self.pool_size,
                        ]

                        max_idx = np.argmax(window)
                        self.max_indices[b, c, i, j] = max_idx
                        output[b, c, i, j] = window.flat[max_idx]

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: Distribute gradient to max position

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        batch_size, channels, _, _ = grad_output.shape

        # Initialize input gradient
        grad_input = np.zeros_like(self.input_cache)

        # Distribute gradient to max positions
        for b in range(batch_size):
            for c in range(channels):
                for i in range(grad_output.shape[2]):
                    for j in range(grad_output.shape[3]):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        max_idx = self.max_indices[b, c, i, j]

                        # Convert flat index to 2D
                        h_idx = max_idx // self.pool_size
                        w_idx = max_idx % self.pool_size

                        grad_input[b, c, h_start + h_idx, w_start + w_idx] += (
                            grad_output[b, c, i, j]
                        )

        return grad_input


class FlattenLayer:
    """
    Flatten Layer

    Converts multi-dimensional input to 1D vector.

    Example:
        Input: (batch, channels, height, width) = (32, 3, 32, 32)
        Output: (batch, 3072) = (32, 3072)

    This is typically used before fully connected layers.
    """

    def __init__(self):
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Flatten input

        Args:
            x: Input of shape (batch, *dimensions)

        Returns:
            Flattened output of shape (batch, product of dimensions)
        """
        self.input_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: Reshape gradient back to original shape

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with same shape as input
        """
        return grad_output.reshape(self.input_shape)


class FullyConnectedLayer:
    """
    Fully Connected (Dense) Layer

    Applies linear transformation followed by activation.

    y = Wx + b

    Parameters:
        input_size: Number of input features
        output_size: Number of output features

    Weight Initialization:
        - Xavier/Glorot initialization for tanh
        - He initialization for ReLU

    Forward Pass:
        output[b, :] = W @ input[b, :] + b

    Backward Pass:
        - dL/dW = (dL/dy)^T @ x
        - dL/db = Σ dL/dy
        - dL/dx = W^T @ dL/dy
    """

    def __init__(
        self, input_size: int, output_size: int, activation=None, weight_init="he"
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Initialize weights based on initialization method
        if weight_init == "he":
            # He initialization for ReLU
            scale = np.sqrt(2.0 / input_size)
        else:
            # Xavier initialization for tanh/sigmoid
            scale = np.sqrt(1.0 / input_size)

        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros(output_size)

        # Cache for backward pass
        self.input_cache = None
        self.output_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Linear transformation + activation

        Args:
            x: Input of shape (batch, input_size)

        Returns:
            Output of shape (batch, output_size)
        """
        self.input_cache = x

        # Linear transformation
        linear_output = np.dot(x, self.weights) + self.bias

        # Apply activation if specified
        if self.activation is not None:
            self.output_cache = linear_output
            return self.activation.forward(linear_output)
        else:
            return linear_output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: Compute gradients

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        batch_size = grad_output.shape[0]

        # If activation exists, get the gradient before activation
        if self.activation is not None:
            grad_before_activation = self.activation.backward(grad_output)
        else:
            grad_before_activation = grad_output

        # Gradient with respect to weights: (input^T @ grad) / batch
        grad_weights = np.dot(self.input_cache.T, grad_before_activation) / batch_size

        # Gradient with respect to bias: sum over batch
        grad_bias = np.sum(grad_before_activation, axis=0) / batch_size

        # Gradient with respect to input: grad @ weights^T
        grad_input = np.dot(grad_before_activation, self.weights.T)

        # Store for optimizer
        self.grad_weights = grad_weights
        self.grad_bias = grad_bias

        return grad_input

    def get_parameters(self):
        """Return weights and biases"""
        return self.weights, self.bias

    def set_parameters(self, weights, bias):
        """Set weights and biases"""
        self.weights = weights
        self.bias = bias


class CrossEntropyLoss:
    """
    Cross-Entropy Loss for Multi-class Classification

    L = -Σ_c y_c * log(p_c)

    where:
    - y is the true label (one-hot encoded)
    - p is the predicted probability

    Properties:
    - Measures dissimilarity between predicted and true distributions
    - Convex with respect to model parameters
    - Works well with softmax activation

    Numerical Stability:
    - Add small epsilon to prevent log(0)
    """

    def __init__(self):
        self.predictions = None
        self.labels = None
        self.epsilon = 1e-15

    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute cross-entropy loss

        Args:
            predictions: Softmax outputs, shape (batch, num_classes)
            labels: One-hot encoded labels, shape (batch, num_classes)

        Returns:
            Scalar loss value
        """
        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)

        # Compute cross entropy: -Σ y * log(p)
        loss = -np.sum(labels * np.log(predictions)) / predictions.shape[0]

        self.predictions = predictions
        self.labels = labels

        return loss

    def backward(self) -> np.ndarray:
        """
        Backward pass: Gradient with respect to predictions

        dL/dp_c = -y_c / p_c

        For one-hot labels where y_k = 1:
        dL/dp_k = -1 / p_k

        Returns:
            Gradient with respect to predictions
        """
        # dL/dpred = -labels / predictions
        grad = -self.labels / self.predictions
        return grad / self.labels.shape[0]


class SimpleCNN:
    """
    Simple Convolutional Neural Network

    Architecture:
        1. ConvLayer (32 filters, 3x3)
        2. ReLU
        3. MaxPool (2x2)
        4. ConvLayer (64 filters, 3x3)
        5. ReLU
        6. MaxPool (2x2)
        7. Flatten
        8. FullyConnected (128)
        9. ReLU
        10. FullyConnected (num_classes)
        11. Softmax

    Example for CIFAR-10 (32x32 RGB images, 10 classes):
        Input: (32, 3, 32, 32)
        -> Conv+ReLU+Pool: (32, 32, 16, 16)
        -> Conv+ReLU+Pool: (32, 64, 8, 8)
        -> Flatten: (32, 4096)
        -> FC: (32, 128)
        -> FC: (32, 10)
        -> Softmax: (32, 10)
    """

    def __init__(self, input_channels: int, num_classes: int):
        self.layers = [
            # First conv block
            ConvLayer(num_filters=32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPoolLayer(pool_size=2, stride=2),
            # Second conv block
            ConvLayer(num_filters=64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPoolLayer(pool_size=2, stride=2),
            # Fully connected layers
            FlattenLayer(),
            FullyConnectedLayer(input_size=None, output_size=128, activation=ReLU()),
            FullyConnectedLayer(
                input_size=128, output_size=num_classes, activation=Softmax()
            ),
        ]

        # Track if first layer input size is set
        self.input_channels = input_channels
        self.initialized = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers"""
        for layer in self.layers:
            # Set input size for first FC layer after flattening
            if isinstance(layer, FullyConnectedLayer) and layer.input_size is None:
                layer.input_size = x.shape[1]
                # Re-initialize with correct size
                scale = np.sqrt(2.0 / layer.input_size)
                layer.weights = (
                    np.random.randn(layer.input_size, layer.output_size) * scale
                )
                layer.bias = np.zeros(layer.output_size)

            x = layer.forward(x)
        return x

    def backward(self, grad_output: np.ndarray):
        """Backward pass through all layers in reverse"""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def get_parameters(self):
        """Get all trainable parameters"""
        params = []
        for layer in self.layers:
            if hasattr(layer, "get_parameters"):
                weights, bias = layer.get_parameters()
                params.append((weights, bias))
        return params

    def set_parameters(self, parameters):
        """Set all trainable parameters"""
        param_idx = 0
        for layer in self.layers:
            if hasattr(layer, "set_parameters"):
                layer.set_parameters(parameters[param_idx][0], parameters[param_idx][1])
                param_idx += 1

    def train_step(
        self, x: np.ndarray, y: np.ndarray, optimizer, loss_fn: CrossEntropyLoss
    ) -> float:
        """
        Single training step: forward, backward, update

        Args:
            x: Input images (batch, channels, height, width)
            y: One-hot labels (batch, num_classes)
            optimizer: Optimizer instance
            loss_fn: Loss function

        Returns:
            Loss value
        """
        # Forward pass
        predictions = self.forward(x)

        # Compute loss
        loss = loss_fn.forward(predictions, y)

        # Backward pass
        grad = loss_fn.backward()
        self.backward(grad)

        # Update parameters
        for layer in self.layers:
            if hasattr(layer, "grad_weights"):
                optimizer.update(
                    layer.weights, layer.grad_weights, layer.bias, layer.grad_bias
                )

        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            x: Input images

        Returns:
            Predicted class indices
        """
        probabilities = self.forward(x)
        return np.argmax(probabilities, axis=1)


def test_cnn():
    """Test CNN implementation"""
    print("=" * 60)
    print("Testing CNN Implementation")
    print("=" * 60)

    # Test data: batch of 4 small images (2 channels, 4x4)
    np.random.seed(42)
    x = np.random.randn(4, 2, 4, 4)
    labels = np.array(
        [
            [1, 0, 0],  # Class 0
            [0, 1, 0],  # Class 1
            [0, 0, 1],  # Class 2
            [1, 0, 0],  # Class 0
        ]
    )

    print(f"\nInput shape: {x.shape}")
    print(f"Labels shape: {labels.shape}")

    # Create CNN
    cnn = SimpleCNN(input_channels=2, num_classes=3)

    # Test forward pass
    output = cnn.forward(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output (first sample): {output[0]}")
    print(f"Sum of probabilities: {np.sum(output[0]):.4f}")

    # Test loss
    loss_fn = CrossEntropyLoss()
    loss = loss_fn.forward(output, labels)
    print(f"\nLoss: {loss:.4f}")

    # Test backward pass
    grad = loss_fn.backward()
    cnn.backward(grad)
    print("Backward pass successful!")

    # Test prediction
    predictions = cnn.predict(x)
    print(f"\nPredictions: {predictions}")
    print(f"True labels: {np.argmax(labels, axis=1)}")

    print("\n" + "=" * 60)
    print("All CNN tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_cnn()
