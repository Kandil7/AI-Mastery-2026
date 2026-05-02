# Chapter 6: Neural Networks - From Perceptrons to Deep Learning

> **Learning Duration:** 5 Days  
> **Difficulty:** Intermediate  
> **Prerequisites:** Linear algebra, calculus, Python

---

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Understand the biological inspiration and mathematical model of neurons
- Build a perceptron from scratch
- Implement multi-layer perceptrons (MLPs)
- Master the backpropagation algorithm
- Train neural networks with various optimizers
- Apply neural networks to real classification problems

---

## 6.1 The Biological Inspiration

### How Our Brains Work

Your brain contains ~86 billion **neurons**. Each neuron:
1. Receives signals from other neurons through **dendrites**
2. **Processes** the signals in the cell body
3. Sends output through the **axon** to other neurons

```
     Dendrites          Cell Body           Axon
    (inputs)          (process)          (output)
       
        ○──○──○    ┌──────────┐    ○──○──○──○
         ╲ │ ╱     │  Process  │     ╱ │ ╲
          ╲│╱      │   +      │    ╱  │  ╲
           ●        │ Threshold│   ●   ●   ●
          ╱│╲      └──────────┘    ╲  │  ╱
         ╱ │ ╲         ↓            ╲ │ ╱
        ○──○──○      Signal         ○──○──○

Signals accumulate, fire if exceeds threshold
```

### The Artificial Neuron

**Warren McCulloch & Walter Pitts (1943)** created the first mathematical neuron model:

```
    Inputs          Weights         Sum + Activation    Output
    x₁ ──────► w₁ ─┐
                   │
    x₂ ──────► w₂ ─┼──► Σ + activation ──► ŷ
                   │
    x₃ ──────► w₃ ─┘
         +        bias ──►
```

Mathematical model:
$$a = \sigma(w_1 x_1 + w_2 x_2 + ... + w_n x_n + b)$$

Where:
- $x$ = input values
- $w$ = weights (learnable parameters)
- $b$ = bias (learnable parameter)
- $\sigma$ = activation function

---

## 6.2 The Perceptron

### Single-Layer Perceptron

The **perceptron** is the simplest neural network:

1. **Weighted sum**: $z = \sum w_i x_i + b$
2. **Activation**: $y = \sigma(z)$

For **binary classification** with step activation:
```
      z > 0: output = 1
      z ≤ 0: output = 0
```

### Implementation

```python
import numpy as np

class Perceptron:
    """
    Single-layer perceptron for binary classification.
    
    Uses step function as activation.
    """
    
    def __init__(self, n_inputs: int, learning_rate: float = 0.1):
        """
        Initialize perceptron.
        
        Args:
            n_inputs: Number of input features
            learning_rate: Step size for weight updates
        """
        # Initialize weights randomly (small values)
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.lr = learning_rate
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input matrix of shape (m, n_inputs)
            
        Returns:
            Binary predictions of shape (m,)
        """
        # Linear combination
        z = X @ self.weights + self.bias
        
        # Step activation
        return (z > 0).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw z values for analysis."""
        return X @ self.weights + self.bias
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            n_epochs: int = 100) -> 'Perceptron':
        """
        Train the perceptron using the perceptron learning algorithm.
        
        Algorithm:
            For each epoch:
                For each sample (x_i, y_i):
                    1. Compute prediction
                    2. Update if wrong:
                        w = w + lr * (y - y_pred) * x
                        b = b + lr * (y - y_pred)
                        
        Args:
            X: Training features (m, n)
            y: Training labels (m,) - 0 or 1
            n_epochs: Number of training epochs
            
        Returns:
            self: Trained model
        """
        m = len(y)
        
        for epoch in range(n_epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(m):
                xi = X_shuffled[i:i+1]  # Keep 2D
                yi = y_shuffled[i]
                
                # Forward pass
                prediction = self.predict(xi)[0]
                error = yi - prediction
                
                # Update weights (only if misclassified)
                if error != 0:
                    self.weights += self.lr * error * xi.flatten()
                    self.bias += self.lr * error
                    
        return self
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

### Perceptron Learning Algorithm

The perceptron can only solve **linearly separable** problems!

```
    Linearly Separable         NOT Linearly Separable
    
    ○ ○ ○                       ○ ● ●
    ○ ○ ○        ──────         ● ● ○
    ● ● ●                       ● ○ ○
    
    Can draw a line           No single line separates
    to separate classes        classes
```

---

## 6.3 Activation Functions

### Why Do We Need Activation Functions?

Without non-linear activations, stacking layers doesn't help:

$$W_2(W_1x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = Wx + b$$

Still just a **linear** model!

### Common Activation Functions

```python
class ActivationFunctions:
    """
    Collection of activation functions and their derivatives.
    """
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid: σ(z) = 1 / (1 + e^(-z))
        
        Properties:
        - Output between 0 and 1
        - Gradient is maximum at 0 (≈0.25)
        - Can cause vanishing gradients
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative: σ'(z) = σ(z) * (1 - σ(z))"""
        s = ActivationFunctions.sigmoid(z)
        return s * (1 - s)
    
    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """
        Hyperbolic Tangent: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
        
        Properties:
        - Output between -1 and 1
        - Zero-centered
        - Gradient stronger than sigmoid
        """
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative: tanh'(z) = 1 - tanh²(z)"""
        return 1 - np.tanh(z) ** 2
    
    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """
        Rectified Linear Unit: max(0, z)
        
        Properties:
        - Output ≥ 0
        - Simple and efficient
        - Sparse activation (some neurons output 0)
        - Can "die" during training (梯度为0)
        """
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative: 1 if z > 0, else 0"""
        return (z > 0).astype(float)
    
    @staticmethod
    def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Leaky ReLU: max(αz, z) where α is small (e.g., 0.01)
        
        Prevents "dying ReLU" problem.
        """
        return np.where(z > 0, z, alpha * z)
    
    @staticmethod
    def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative: 1 if z > 0, else α"""
        return np.where(z > 0, 1, alpha)
    
    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Softmax: exp(z_i) / Σ exp(z_j)
        
        Used for multi-class classification output layer.
        Converts logits to probabilities that sum to 1.
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def step(z: np.ndarray) -> np.ndarray:
        """Step function: 1 if z > 0, else 0"""
        return (z > 0).astype(int)
```

### Visual Comparison

```
Sigmoid:        Tanh:           ReLU:           Leaky ReLU:
  1 │ ╱           1 │          1 │         ╱      1 │
    │╱             │ ╱          │╱          │╱       │╱
  0 ├─────────   0 ├────────  0 ├────────  0 ├────────
    │            ──┼──         ──┼────      ──┼────
 -1 │            -1 │          │           α│
    └────────       └──────     └────────    └──────
    -3  0  3       -3  0  3    0   3        -3  0  3
```

---

## 6.4 Multi-Layer Perceptron (MLP)

### Architecture

An MLP has:
- **Input layer**: Receives features
- **Hidden layers**: Process information
- **Output layer**: Produces predictions

```
    Input      Hidden 1      Hidden 2      Output
    Layer       Layer         Layer        Layer
    
      ○────────○────────○────────○──► y₁
      │        │        │        │
      ○────────○────────○────────○──► y₂
      │        │        │        │
      ○────────○────────○────────○──► y₃
      │        │        │        │
      ○        ○        ○        ○
     x₁       h₁₁      h₂₁      o₁
     x₂       h₁₂      h₂₂      o₂
     x₃       h₁₃      h₂₃
     x₄
     
Each arrow = weight parameter to learn
```

### Complete MLP Implementation

```python
import numpy as np
from typing import List, Callable, Tuple
from collections import defaultdict

class Layer:
    """A single layer in the neural network."""
    
    def __init__(self, n_inputs: int, n_neurons: int, 
                 activation: str = 'relu'):
        """
        Initialize a layer.
        
        Args:
            n_inputs: Number of input features
            n_neurons: Number of neurons in this layer
            activation: Activation function name
        """
        # He initialization (good for ReLU)
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.bias = np.zeros((1, n_neurons))
        self.activation = activation
        
        # Store for backprop
        self.inputs = None
        self.output = None
        self.z = None  # Pre-activation values
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        self.inputs = X
        self.z = X @ self.weights + self.bias
        
        # Apply activation
        if self.activation == 'sigmoid':
            self.output = self._sigmoid(self.z)
        elif self.activation == 'tanh':
            self.output = np.tanh(self.z)
        elif self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'leaky_relu':
            self.output = np.where(self.z > 0, self.z, 0.01 * self.z)
        elif self.activation == 'linear':
            self.output = self.z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
            
        return self.output
    
    def backward(self, dY: np.ndarray, 
                 prev_weights: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass through the layer.
        
        Args:
            dY: Gradient of loss with respect to output
            prev_weights: Previous layer's weights (for computing weight gradients)
            
        Returns:
            dX: Gradient with respect to inputs
            dW: Gradient with respect to weights
            db: Gradient with respect to bias
        """
        m = self.inputs.shape[0]
        
        # Compute activation gradient
        if self.activation == 'sigmoid':
            dZ = dY * self._sigmoid_deriv(self.z)
        elif self.activation == 'tanh':
            dZ = dY * (1 - np.tanh(self.z) ** 2)
        elif self.activation == 'relu':
            dZ = dY * (self.z > 0).astype(float)
        elif self.activation == 'leaky_relu':
            dZ = dY * np.where(self.z > 0, 1, 0.01)
        elif self.activation == 'linear':
            dZ = dY
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # Gradients
        dW = (1/m) * self.inputs.T @ dZ
        db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.weights.T
        
        return dX, dW, db
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_deriv(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)


class MLP:
    """
    Multi-Layer Perceptron / Neural Network.
    
    Supports arbitrary architecture, various activations, and optimizers.
    """
    
    def __init__(self, layer_sizes: List[int], 
                 activations: List[str] = None,
                 learning_rate: float = 0.01,
                 regularization: float = 0.0):
        """
        Initialize MLP.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activations: List of activation functions for each layer (except input)
            learning_rate: Learning rate for gradient descent
            regularization: L2 regularization strength
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Default activations
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 1)
            activations[-1] = 'linear'  # Output layer
        
        # Create layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)
            
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict(X)
        return np.argmax(probs, axis=1)
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE loss with regularization."""
        predictions = self.predict(X)
        m = X.shape[0]
        
        # MSE
        loss = (1/(2*m)) * np.sum((predictions - y) ** 2)
        
        # L2 regularization
        if self.regularization > 0:
            reg_loss = 0
            for layer in self.layers:
                reg_loss += np.sum(layer.weights ** 2)
            loss += (self.regularization / (2*m)) * reg_loss
            
        return loss
    
    def backpropagation(self, X: np.ndarray, y: np.ndarray):
        """Perform backpropagation to compute gradients."""
        m = X.shape[0]
        
        # Forward pass
        self.forward(X)
        
        # Initialize gradients
        gradients = []
        
        # Output layer gradient (MSE loss for regression)
        # dL/dy = (y_pred - y)
        dY = (self.layers[-1].output - y)
        
        # Backward through each layer
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            prev_weights = self.layers[i-1].weights if i > 0 else None
            
            dX, dW, db = layer.backward(dY, prev_weights)
            gradients.append((dW, db))
            
            # Next layer's gradient (if not first layer)
            if i > 0:
                # Add regularization gradient to weights
                if self.regularization > 0:
                    dW_reg = (self.regularization / m) * layer.weights
                    dW = dW + dW_reg
                    
                dY = dX
                
        return gradients[::-1]  # Reverse to match layer order
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            n_epochs: int = 1000, batch_size: int = 32,
            verbose: bool = True) -> 'MLP':
        """
        Train the MLP using mini-batch gradient descent.
        
        Args:
            X: Training features (m, n_features)
            y: Training targets (m,) or (m, n_outputs)
            n_epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Print progress
            
        Returns:
            self: Trained model
        """
        m = X.shape[0]
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Compute gradients
                gradients = self.backpropagation(X_batch, y_batch)
                
                # Update weights
                for layer, (dW, db) in zip(self.layers, gradients):
                    layer.weights -= self.learning_rate * dW
                    layer.bias -= self.learning_rate * db
                    
            # Record metrics
            if verbose and epoch % 100 == 0:
                loss = self.compute_loss(X, y)
                self.loss_history.append(loss)
                
                # Accuracy for classification
                if len(np.unique(y)) < m:  # Classification task
                    preds = self.predict_classes(X)
                    acc = np.mean(preds == y)
                    print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
                else:
                    print(f"Epoch {epoch}: Loss = {loss:.4f}")
                    
        return self
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        predictions = self.predict_classes(X)
        return np.mean(predictions == y)
```

---

## 6.5 Loss Functions

```python
class LossFunctions:
    """
    Collection of loss functions and their gradients.
    """
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient of MSE with respect to predictions."""
        return y_pred - y_true
    
    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Cross-entropy loss for multi-class classification.
        
        L = -Σ y_true * log(y_pred)
        """
        # Clip to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + 
                       (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def cross_entropy_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient of cross-entropy."""
        return y_pred - y_true
    
    @staticmethod
    def categorical_cross_entropy(y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> float:
        """
        Categorical cross-entropy for one-hot encoded labels.
        """
        # Clip for numerical stability
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    @staticmethod
    def categorical_ce_gradient(y_true: np.ndarray, 
                                y_pred: np.ndarray) -> np.ndarray:
        """Gradient for categorical CE with softmax."""
        return y_pred - y_true
```

---

## 6.6 Complete Training Example

```python
# Example: Classification with MLP

# Generate synthetic data
np.random.seed(42)
m = 200

# Class 0: centered at (0, 0)
X0 = np.random.randn(m//2, 2) * 0.5 + np.array([0, 0])
y0 = np.zeros(m//2)

# Class 1: centered at (2, 2)
X1 = np.random.randn(m//2, 2) * 0.5 + np.array([2, 2])
y1 = np.ones(m//2)

X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

# One-hot encode
y_onehot = np.zeros((m, 2))
y_onehot[np.arange(m), y.astype(int)] = 1

# Create and train MLP
mlp = MLP(
    layer_sizes=[2, 16, 16, 2],  # 2 inputs, 2 hidden, 2 outputs
    activations=['relu', 'relu', 'linear'],
    learning_rate=0.1,
    regularization=0.001
)

mlp.fit(X, y_onehot, n_epochs=1000, batch_size=32, verbose=False)

# Evaluate
accuracy = mlp.accuracy(X, y)
print(f"Training Accuracy: {accuracy:.4f}")

# Visualize decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                      np.linspace(y_min, y_max, 100))

Z = mlp.predict_classes(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X0[:, 0], X0[:, 1], c='blue', label='Class 0')
plt.scatter(X1[:, 0], X1[:, 1], c='red', label='Class 1')
plt.legend()
plt.show()
```

---

## 📝 Summary

### Key Takeaways

1. **Neuron**: $y = \sigma(Wx + b)$ - weighted sum + activation
2. **Perceptron**: Single layer, step function - only linearly separable
3. **Activation functions**: Add non-linearity (ReLU, sigmoid, tanh, softmax)
4. **MLP**: Multiple layers - can learn complex patterns
5. **Backpropagation**: Chain rule to compute gradients efficiently
6. **Loss functions**: MSE (regression), Cross-Entropy (classification)

### Architecture Design

| Problem Type | Output Activation | Loss Function |
|--------------|-------------------|---------------|
| Binary Classification | Sigmoid | Binary Cross-Entropy |
| Multi-class | Softmax | Categorical Cross-Entropy |
| Regression | Linear | MSE |

---

## ❓ Quick Check

1. Why do we need activation functions?
2. What is the vanishing gradient problem?
3. Why is He initialization preferred for ReLU?
4. How does backpropagation work?

*Answers at end of chapter*