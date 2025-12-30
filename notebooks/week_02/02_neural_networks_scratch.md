# Week 02: Neural Networks from Scratch

Building neural networks from first principles using only NumPy.

## Learning Objectives
1. Understand forward propagation mathematically
2. Implement backpropagation using chain rule
3. Build a complete neural network class

---

## 1. The Neuron Model

### 1.1 Single Neuron (Perceptron)

```python
import numpy as np
from typing import List, Tuple, Callable

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation: σ(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative: σ'(x) = σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)


class Neuron:
    """
    Single neuron implementation.
    
    Computation: z = w·x + b, a = σ(z)
    """
    def __init__(self, n_inputs: int):
        # Xavier initialization
        self.weights = np.random.randn(n_inputs) * np.sqrt(2.0 / n_inputs)
        self.bias = 0.0
    
    def forward(self, x: np.ndarray) -> float:
        """Compute neuron output."""
        self.z = np.dot(self.weights, x) + self.bias
        self.a = sigmoid(self.z)
        return self.a


# Test
neuron = Neuron(3)
x = np.array([1.0, 2.0, 3.0])
output = neuron.forward(x)
print(f"Neuron output: {output:.4f}")
```

---

## 2. Neural Network Layer

### 2.1 Dense Layer

```python
class DenseLayer:
    """
    Fully connected layer.
    
    Shape:
    - Input: (batch_size, n_inputs)
    - Output: (batch_size, n_outputs)
    - Weights: (n_inputs, n_outputs)
    - Bias: (n_outputs,)
    """
    def __init__(self, n_inputs: int, n_outputs: int, activation: str = 'relu'):
        # He initialization for ReLU
        self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros(n_outputs)
        self.activation = activation
    
    def _activate(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            return z  # linear
    
    def _activate_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of activation function."""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activate(z)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        else:
            return np.ones_like(z)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        z = X @ W + b
        a = activation(z)
        """
        self.X = X  # Cache for backward
        self.z = X @ self.W + self.b
        self.a = self._activate(self.z)
        return self.a
    
    def backward(self, da: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass.
        
        Given da (gradient w.r.t. activation output):
        1. dz = da * activation'(z)
        2. dW = X.T @ dz
        3. db = sum(dz, axis=0)
        4. dX = dz @ W.T (pass to previous layer)
        
        Returns dX for previous layer.
        """
        batch_size = self.X.shape[0]
        
        # Gradient through activation
        dz = da * self._activate_derivative(self.z)
        
        # Gradients for weights and bias
        dW = (self.X.T @ dz) / batch_size
        db = np.sum(dz, axis=0) / batch_size
        
        # Gradient for input (to pass backward)
        dX = dz @ self.W.T
        
        # Update parameters
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dX


# Test
layer = DenseLayer(3, 2, activation='relu')
X = np.random.randn(5, 3)  # batch of 5
output = layer.forward(X)
print(f"Layer output shape: {output.shape}")
```

---

## 3. Complete Neural Network

### 3.1 Multi-Layer Perceptron

```python
class NeuralNetwork:
    """
    Multi-layer neural network for classification.
    
    Example:
        >>> nn = NeuralNetwork([784, 128, 64, 10])
        >>> nn.fit(X_train, y_train, epochs=100)
        >>> predictions = nn.predict(X_test)
    """
    def __init__(self, layer_sizes: List[int], activation: str = 'relu'):
        """
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activation: Activation for hidden layers
        """
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            # Last layer uses sigmoid for classification
            act = 'sigmoid' if i == len(layer_sizes) - 2 else activation
            layer = DenseLayer(layer_sizes[i], layer_sizes[i+1], act)
            self.layers.append(layer)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Binary cross-entropy loss.
        
        L = -1/n * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
        """
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        loss = -np.mean(
            y_true * np.log(y_pred) + 
            (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, learning_rate: float):
        """
        Backward pass through all layers.
        
        Starts with gradient of loss w.r.t. output.
        """
        # Gradient of BCE loss w.r.t. predictions
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        da = (y_pred - y_true) / (y_pred * (1 - y_pred) + eps)
        
        # Backpropagate through layers in reverse
        for layer in reversed(self.layers):
            da = layer.backward(da, learning_rate)
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """Train the network."""
        n_samples = X.shape[0]
        history = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0
            n_batches = 0
            
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward
                y_pred = self.forward(X_batch)
                
                # Loss
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss
                n_batches += 1
                
                # Backward
                self.backward(y_pred, y_batch, learning_rate)
            
            avg_loss = epoch_loss / n_batches
            history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict(X)
        return (probs > threshold).astype(int)


# Test on XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 4, 1])
history = nn.fit(X, y, epochs=1000, learning_rate=1.0, verbose=False)

print("\nXOR Problem Results:")
print("Predictions:", nn.predict(X).flatten())
print("Expected:   ", y.flatten())
print(f"Final loss: {history[-1]:.4f}")
```

---

## 4. Backpropagation Deep Dive

### 4.1 Computational Graph

```python
def backprop_visualization():
    """
    Visualize backpropagation on a simple 2-layer network.
    
    Forward:
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2) = y_pred
    L = BCE(y_pred, y_true)
    
    Backward:
    ∂L/∂a2 = (a2 - y) / (a2 * (1-a2))
    ∂L/∂z2 = ∂L/∂a2 * sigmoid'(z2) = a2 - y
    ∂L/∂W2 = a1.T @ ∂L/∂z2
    ∂L/∂a1 = ∂L/∂z2 @ W2.T
    ∂L/∂z1 = ∂L/∂a1 * relu'(z1)
    ∂L/∂W1 = X.T @ ∂L/∂z1
    """
    # Simple example
    X = np.array([[1.0, 2.0]])
    y = np.array([[1.0]])
    
    # Initialize weights
    W1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 2x3
    b1 = np.array([0.0, 0.0, 0.0])
    W2 = np.array([[0.7], [0.8], [0.9]])  # 3x1
    b2 = np.array([0.0])
    
    # ===== FORWARD PASS =====
    print("FORWARD PASS:")
    z1 = X @ W1 + b1
    print(f"z1 = X @ W1 + b1 = {z1}")
    
    a1 = np.maximum(0, z1)  # ReLU
    print(f"a1 = relu(z1) = {a1}")
    
    z2 = a1 @ W2 + b2
    print(f"z2 = a1 @ W2 + b2 = {z2}")
    
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid
    print(f"a2 = sigmoid(z2) = {a2}")
    
    loss = -np.mean(y * np.log(a2) + (1-y) * np.log(1-a2))
    print(f"Loss = {loss:.4f}")
    
    # ===== BACKWARD PASS =====
    print("\nBACKWARD PASS:")
    
    # Gradient of loss w.r.t. z2 (simplified for sigmoid + BCE)
    dz2 = a2 - y
    print(f"∂L/∂z2 = a2 - y = {dz2}")
    
    dW2 = a1.T @ dz2
    print(f"∂L/∂W2 = a1.T @ ∂L/∂z2 = \n{dW2}")
    
    da1 = dz2 @ W2.T
    print(f"∂L/∂a1 = ∂L/∂z2 @ W2.T = {da1}")
    
    dz1 = da1 * (z1 > 0)  # ReLU derivative
    print(f"∂L/∂z1 = ∂L/∂a1 * relu'(z1) = {dz1}")
    
    dW1 = X.T @ dz1
    print(f"∂L/∂W1 = X.T @ ∂L/∂z1 = \n{dW1}")


backprop_visualization()
```

---

## 5. Activation Functions Comparison

```python
def compare_activations():
    """Compare different activation functions."""
    x = np.linspace(-5, 5, 100)
    
    activations = {
        'Sigmoid': (lambda x: 1 / (1 + np.exp(-x)), 
                    lambda x: 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))),
        'Tanh': (np.tanh, 
                 lambda x: 1 - np.tanh(x)**2),
        'ReLU': (lambda x: np.maximum(0, x), 
                 lambda x: (x > 0).astype(float)),
        'Leaky ReLU': (lambda x: np.where(x > 0, x, 0.01 * x),
                       lambda x: np.where(x > 0, 1, 0.01)),
    }
    
    print("Activation Functions Comparison:")
    print("-" * 60)
    print(f"{'Activation':<15} {'Output Range':<20} {'Gradient Range':<20}")
    print("-" * 60)
    
    for name, (func, deriv) in activations.items():
        y = func(x)
        dy = deriv(x)
        print(f"{name:<15} [{y.min():.2f}, {y.max():.2f}]       [{dy.min():.2f}, {dy.max():.2f}]")


compare_activations()
```

---

## 6. Exercises

### Exercise 1: Add Momentum to SGD
```python
def sgd_momentum(params, grads, velocities, lr=0.01, momentum=0.9):
    """
    Implement SGD with momentum.
    
    v_t = γ * v_{t-1} + lr * grad
    θ_t = θ_{t-1} - v_t
    """
    # YOUR CODE HERE
    for i in range(len(params)):
        velocities[i] = momentum * velocities[i] + lr * grads[i]
        params[i] -= velocities[i]
    return params, velocities
```

### Exercise 2: Add Dropout Layer
```python
class Dropout:
    """
    Dropout regularization.
    
    During training: randomly zero out neurons with probability p
    During inference: scale by (1-p)
    """
    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
            return X * self.mask
        else:
            return X
    
    def backward(self, da: np.ndarray) -> np.ndarray:
        return da * self.mask
```

### Exercise 3: Implement Batch Normalization
```python
class BatchNorm:
    """
    Batch normalization layer.
    
    x_norm = (x - μ) / √(σ² + ε)
    y = γ * x_norm + β
    """
    def __init__(self, n_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        self.eps = eps
        self.momentum = momentum
        
        # Running statistics
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mean = X.mean(axis=0)
            var = X.var(axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        self.x_norm = (X - mean) / np.sqrt(var + self.eps)
        return self.gamma * self.x_norm + self.beta
```

---

## Summary

| Component | Purpose | Key Formula |
|-----------|---------|-------------|
| Neuron | Basic computational unit | z = w·x + b, a = σ(z) |
| Layer | Transform representations | Z = X @ W + b |
| Forward Pass | Compute predictions | Layer by layer activation |
| Backward Pass | Compute gradients | Chain rule through layers |
| Loss Function | Measure error | BCE, MSE |

---

## Next Week Preview
Week 03 will cover:
- Classical Machine Learning Algorithms
- Linear Regression, Logistic Regression
- Decision Trees and Ensembles
