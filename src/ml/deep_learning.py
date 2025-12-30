"""
Deep Learning Module
====================
Neural network components built from scratch.

Includes:
- Layer abstractions (Dense, Activation, Dropout, BatchNorm)
- Sequential model with automatic backpropagation
- Loss functions
- Convolution operations basics

Mathematical Foundation:
- Forward: aˡ = σ(Wˡaˡ⁻¹ + bˡ)
- Backward: ∂L/∂W = ∂L/∂a × ∂a/∂z × ∂z/∂W

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod

# Try relative import, fall back to direct
try:
    from ..core.math_operations import sigmoid, relu, relu_derivative, softmax, tanh
except ImportError:
    from core.math_operations import sigmoid, relu, relu_derivative, softmax, tanh


# ============================================================
# LAYER ABSTRACTIONS
# ============================================================

class Layer(ABC):
    """
    Abstract base class for neural network layers.
    
    Each layer must implement:
        - forward: Compute output from input
        - backward: Compute gradients for backpropagation
    """
    
    def __init__(self):
        self.input = None
        self.output = None
        self.trainable = True
    
    @abstractmethod
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            input_data: Input to the layer
            training: Whether in training mode (affects dropout, batchnorm)
        
        Returns:
            Layer output
        """
        pass
    
    @abstractmethod
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            output_gradient: Gradient of loss w.r.t. layer output (∂L/∂y)
            learning_rate: Learning rate for weight updates
        
        Returns:
            Gradient of loss w.r.t. layer input (∂L/∂x)
        """
        pass
    
    def get_params(self) -> dict:
        """Get layer parameters (for saving/loading)."""
        return {}
    
    def set_params(self, params: dict):
        """Set layer parameters."""
        pass


class Dense(Layer):
    """
    Fully Connected (Dense) Layer.
    
    Forward: y = Wx + b
    Backward:
        ∂L/∂W = ∂L/∂y × xᵀ
        ∂L/∂b = ∂L/∂y
        ∂L/∂x = Wᵀ × ∂L/∂y
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        weight_init: 'xavier', 'he', or 'random'
    
    Example:
        >>> dense = Dense(784, 128, weight_init='he')
        >>> output = dense.forward(input_data)
    """
    
    def __init__(self, input_size: int, output_size: int,
                 weight_init: str = 'xavier'):
        super().__init__()
        
        # Weight initialization
        if weight_init == 'xavier':
            # Good for tanh/sigmoid: Var(W) = 1/n_in
            scale = np.sqrt(1.0 / input_size)
        elif weight_init == 'he':
            # Good for ReLU: Var(W) = 2/n_in
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = 0.01
        
        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros((1, output_size))
        
        # For momentum/Adam
        self.dW = None
        self.db = None
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        self.output = input_data @ self.weights + self.bias
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Gradients
        self.dW = self.input.T @ output_gradient
        self.db = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Gradient w.r.t. input (for previous layer)
        input_gradient = output_gradient @ self.weights.T
        
        # Update weights
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db
        
        return input_gradient
    
    def get_params(self) -> dict:
        return {'weights': self.weights.copy(), 'bias': self.bias.copy()}
    
    def set_params(self, params: dict):
        self.weights = params['weights']
        self.bias = params['bias']


class Activation(Layer):
    """
    Activation Layer.
    
    Applies element-wise activation function.
    
    Supported:
        - 'relu': max(0, x)
        - 'sigmoid': 1 / (1 + e^{-x})
        - 'tanh': (e^x - e^{-x}) / (e^x + e^{-x})
        - 'softmax': e^{xᵢ} / Σe^{xⱼ}
        - 'leaky_relu': max(αx, x)
    
    Example:
        >>> activation = Activation('relu')
        >>> output = activation.forward(input_data)
    """
    
    def __init__(self, activation: str = 'relu', alpha: float = 0.01):
        super().__init__()
        self.activation = activation
        self.alpha = alpha  # For leaky ReLU
        self.trainable = False
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        
        if self.activation == 'relu':
            self.output = relu(input_data)
        elif self.activation == 'sigmoid':
            self.output = sigmoid(input_data)
        elif self.activation == 'tanh':
            self.output = tanh(input_data)
        elif self.activation == 'softmax':
            self.output = softmax(input_data)
        elif self.activation == 'leaky_relu':
            self.output = np.where(input_data > 0, input_data, self.alpha * input_data)
        else:
            self.output = input_data  # Linear
        
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.activation == 'relu':
            return output_gradient * relu_derivative(self.input)
        elif self.activation == 'sigmoid':
            s = sigmoid(self.input)
            return output_gradient * s * (1 - s)
        elif self.activation == 'tanh':
            t = tanh(self.input)
            return output_gradient * (1 - t ** 2)
        elif self.activation == 'softmax':
            # For cross-entropy loss, combined gradient is simpler
            return output_gradient
        elif self.activation == 'leaky_relu':
            return output_gradient * np.where(self.input > 0, 1, self.alpha)
        else:
            return output_gradient


class Dropout(Layer):
    """
    Dropout regularization layer.
    
    Randomly zeros out units during training to prevent overfitting.
    During inference, all units are active (scaled by keep probability).
    
    Args:
        rate: Probability of dropping a unit (0-1)
    
    Example:
        >>> dropout = Dropout(0.5)
        >>> output = dropout.forward(input_data, training=True)
    """
    
    def __init__(self, rate: float = 0.5):
        super().__init__()
        self.rate = rate
        self.mask = None
        self.trainable = False
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        
        if training:
            # Create random mask
            self.mask = np.random.binomial(1, 1 - self.rate, input_data.shape)
            # Scale by 1/(1-rate) to maintain expected values
            self.output = input_data * self.mask / (1 - self.rate)
        else:
            self.output = input_data
        
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return output_gradient * self.mask / (1 - self.rate)


class BatchNormalization(Layer):
    """
    Batch Normalization layer.
    
    Normalizes activations to have zero mean and unit variance.
    Reduces internal covariate shift and allows higher learning rates.
    
    Forward (training):
        μ = (1/m) Σxᵢ
        σ² = (1/m) Σ(xᵢ - μ)²
        x̂ = (x - μ) / √(σ² + ε)
        y = γx̂ + β
    
    Args:
        n_features: Number of features
        momentum: For running statistics
        epsilon: Numerical stability
    
    Example:
        >>> bn = BatchNormalization(128)
        >>> output = bn.forward(input_data, training=True)
    """
    
    def __init__(self, n_features: int, momentum: float = 0.9, epsilon: float = 1e-5):
        super().__init__()
        
        # Learnable parameters
        self.gamma = np.ones((1, n_features))  # Scale
        self.beta = np.zeros((1, n_features))   # Shift
        
        # Running statistics for inference
        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))
        
        self.momentum = momentum
        self.epsilon = epsilon
        
        # For backward pass
        self.x_normalized = None
        self.std = None
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        
        if training:
            # Batch statistics
            mean = np.mean(input_data, axis=0, keepdims=True)
            var = np.var(input_data, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        self.std = np.sqrt(var + self.epsilon)
        self.x_normalized = (input_data - mean) / self.std
        
        # Scale and shift
        self.output = self.gamma * self.x_normalized + self.beta
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        m = output_gradient.shape[0]
        
        # Gradients for learnable parameters
        dgamma = np.sum(output_gradient * self.x_normalized, axis=0, keepdims=True)
        dbeta = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Gradient w.r.t. normalized input
        dx_normalized = output_gradient * self.gamma
        
        # Gradient w.r.t. input
        input_gradient = (1 / m) / self.std * (
            m * dx_normalized 
            - np.sum(dx_normalized, axis=0) 
            - self.x_normalized * np.sum(dx_normalized * self.x_normalized, axis=0)
        )
        
        # Update parameters
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return input_gradient


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class Loss(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute loss."""
        pass
    
    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of loss w.r.t. predictions."""
        pass


class MSELoss(Loss):
    """
    Mean Squared Error Loss.
    
    L = (1/n) Σ(ŷ - y)²
    ∂L/∂ŷ = (2/n)(ŷ - y)
    """
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = y_pred.shape[0]
        return (2 / n) * (y_pred - y_true)


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy Loss (for classification).
    
    Binary: L = -[y log(p) + (1-y) log(1-p)]
    Multi-class: L = -Σ yᵢ log(pᵢ)
    
    With softmax output, gradient simplifies to: ∂L/∂z = p - y
    """
    
    def __init__(self, from_logits: bool = False):
        self.from_logits = from_logits
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.from_logits:
            y_pred = softmax(y_pred)
        
        # Clip for numerical stability
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # If y_true is not one-hot, convert
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_true_onehot = np.zeros_like(y_pred)
            y_true_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1
            y_true = y_true_onehot
        
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        if self.from_logits:
            y_pred = softmax(y_pred)
        
        # If y_true is not one-hot, convert
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_true_onehot = np.zeros_like(y_pred)
            y_true_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1
            y_true = y_true_onehot
        
        n = y_pred.shape[0]
        return (y_pred - y_true) / n


class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross-Entropy Loss.
    
    L = -[y log(σ(z)) + (1-y) log(1-σ(z))]
    ∂L/∂z = σ(z) - y
    """
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        n = y_pred.shape[0]
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * n)


# ============================================================
# NEURAL NETWORK (Sequential Model)
# ============================================================

class NeuralNetwork:
    """
    Sequential Neural Network model.
    
    Stacks layers in sequence, handles forward/backward propagation.
    
    Example:
        >>> model = NeuralNetwork()
        >>> model.add(Dense(784, 256, weight_init='he'))
        >>> model.add(Activation('relu'))
        >>> model.add(Dropout(0.3))
        >>> model.add(Dense(256, 10))
        >>> model.add(Activation('softmax'))
        >>> 
        >>> model.compile(loss=CrossEntropyLoss(), learning_rate=0.001)
        >>> history = model.fit(X_train, y_train, epochs=10, batch_size=32)
    """
    
    def __init__(self):
        self.layers: List[Layer] = []
        self.loss: Optional[Loss] = None
        self.learning_rate: float = 0.001
    
    def add(self, layer: Layer):
        """Add a layer to the model."""
        self.layers.append(layer)
    
    def compile(self, loss: Loss, learning_rate: float = 0.001):
        """Configure the model for training."""
        self.loss = loss
        self.learning_rate = learning_rate
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through all layers."""
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output
    
    def backward(self, gradient: np.ndarray):
        """Backward pass through all layers (reverse order)."""
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, self.learning_rate)
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Single training step."""
        # Forward
        y_pred = self.forward(X, training=True)
        
        # Compute loss
        loss_value = self.loss.forward(y_pred, y)
        
        # Backward
        gradient = self.loss.backward(y_pred, y)
        self.backward(gradient)
        
        return loss_value
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: bool = True) -> dict:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
            epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_data: Optional (X_val, y_val) tuple
            verbose: Print training progress
        
        Returns:
            Dictionary with training history
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        
        history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss = self.train_step(X_batch, y_batch)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            
            # Compute accuracy
            y_pred = self.predict(X)
            accuracy = np.mean(y_pred == y) if y.ndim == 1 else np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
            history['accuracy'].append(accuracy)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.forward(X_val, training=False)
                val_loss = self.loss.forward(val_pred, y_val)
                val_accuracy = np.mean(self.predict(X_val) == y_val) if y_val.ndim == 1 else np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - acc: {accuracy:.4f}"
                if validation_data is not None:
                    msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f}"
                print(msg)
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        output = self.forward(X, training=False)
        
        # For classification, return class indices
        if output.ndim > 1 and output.shape[1] > 1:
            return np.argmax(output, axis=1)
        return output
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.forward(X, training=False)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate model on test data."""
        y_pred = self.forward(X, training=False)
        loss = self.loss.forward(y_pred, y)
        
        if y.ndim == 1:
            accuracy = np.mean(np.argmax(y_pred, axis=1) == y)
        else:
            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        
        return loss, accuracy
    
    def summary(self):
        """Print model summary."""
        print("=" * 60)
        print("Model Summary")
        print("=" * 60)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_type = layer.__class__.__name__
            params = layer.get_params()
            n_params = sum(p.size for p in params.values()) if params else 0
            total_params += n_params
            print(f"Layer {i+1}: {layer_type:20s} | Params: {n_params:,}")
        
        print("=" * 60)
        print(f"Total Parameters: {total_params:,}")
        print("=" * 60)


# ============================================================
# CONVOLUTION BASICS (For understanding)
# ============================================================

def conv2d_single(input_image: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    2D convolution (single channel, single kernel).
    
    For educational purposes. Production uses im2col or cuDNN.
    
    Args:
        input_image: 2D array (H, W)
        kernel: 2D array (Kh, Kw)
        stride: Step size
    
    Returns:
        Convolved output
    """
    h, w = input_image.shape
    kh, kw = kernel.shape
    
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = input_image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output


def max_pool2d(input_image: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    """
    Max pooling operation.
    
    Reduces spatial dimensions while keeping important features.
    """
    h, w = input_image.shape
    
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = input_image[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)
    
    return output


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Layers
    'Layer', 'Dense', 'Activation', 'Dropout', 'BatchNormalization',
    # Loss
    'Loss', 'MSELoss', 'CrossEntropyLoss', 'BinaryCrossEntropyLoss',
    # Model
    'NeuralNetwork',
    # Convolution
    'conv2d_single', 'max_pool2d',
]
