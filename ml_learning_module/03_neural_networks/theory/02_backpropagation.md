# Backpropagation: The Learning Algorithm of Neural Networks

## Introduction

**Backpropagation** (short for "backward propagation of errors") is the algorithm that enables neural networks to learn from data. It efficiently computes the gradient of the loss function with respect to each weight using the chain rule.

### Historical Context

- **1950s-1960s**: Early ideas from control theory (Bellman, Kalman)
- **1970s**: Seppo Linnainmaa discovered automatic differentiation, the computational basis
- **1986**: Rumelhart, Hinton, and Williams published the modern backpropagation algorithm
- **2012**: Backpropagation + deep networks → ImageNet breakthrough (AlexNet)

---

## 1. The Forward Pass

### 1.1 Network Computation

Given an input $x$ and network with $L$ layers:

```
Forward Pass:

Input Layer:        x → [x₁, x₂, ..., xₙ]

Layer 1:            a¹ = σ(W¹x + b¹)  → [a₁¹, a₂¹, ..., aₘ¹]
Layer 2:            a² = σ(W²a¹ + b²) → [a₁², a₂², ..., aₖ²]
...
Layer L:            aᴸ = σ(Wᴸaᴸ⁻¹ + bᴸ) → ŷ

Output:             ŷ (prediction)
```

### 1.2 Layer-by-Layer Computation

For layer $l$:
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$

Where:
- $z^{(l)}$: Pre-activation values
- $W^{(l)}$: Weight matrix for layer $l$
- $b^{(l)}$: Bias vector for layer $l$
- $a^{(l)}$: Activation values (output of layer $l$)
- $\sigma$: Activation function

### 1.3 Example: 2-Layer Network

```
        Input          Layer 1         Layer 2         Output
         
         x₁ ──► ╲                     ╲
                ╲ W¹, b¹ → a¹ ╲      ╲ W², b² → a² ╲      ŷ
         x₂ ──► ╱                     ╱                  ╱
         
Dimensions:
- x: (n⁰, 1)
- W¹: (n¹, n⁰)
- a¹: (n¹, 1)
- W²: (n², n¹)
- a² = ŷ: (n², 1)
```

---

## 2. The Loss Function

### 2.1 Common Loss Functions

**Mean Squared Error (Regression):**
$$L = \frac{1}{m}\sum_{i=1}^{m}(y^{(i)} - \hat{y}^{(i)})^2$$

**Cross-Entropy / Log Loss (Classification):**
$$L = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]$$

**Softmax Cross-Entropy (Multi-class):**
$$L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_k^{(i)} \log\left(\frac{e^{z_k^{(i)}}}{\sum_j e^{z_j^{(i)}}}\right)$$

### 2.2 Loss for Single Sample (Simplified Notation)

For the rest of this derivation, we'll work with a single sample:

$$L(\hat{y}, y)$$

---

## 3. The Backward Pass: Chain Rule

### 3.1 The Core Idea

Backpropagation applies the chain rule to compute gradients:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### 3.2 Chain Rule Visualization

```
Computation Graph:
                    w₁
                     ╲
                      ╲
              w₂ ════►╲
                       ╲
                        ╲ a → z → ŷ → L
                       ╱
              w₃ ════►╱
                     ╱
                    w₄
                    
Backward Pass (gradients):
                    ∂L/∂w₁
                     ╱
                    ╱
            ∂L/∂w₂ ╱
                   ╱
        ∂L/∂a ← ∂L/∂z ← ∂L/∂ŷ ← L
                   ╲
            ∂L/∂w₃ ╲
                    ╲
                     ╲
                    ∂L/∂w₄
```

---

## 4. Step-by-Step Gradient Computation

### 4.1 Output Layer Gradients

For the last layer (layer $L$) with cross-entropy loss and softmax:

**Step 1: Gradient of loss w.r.t. pre-activation $z^{(L)}$:**

$$\frac{\partial L}{\partial z^{(L)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}}$$

For cross-entropy with softmax:
$$\frac{\partial L}{\partial z^{(L)}} = a^{(L)} - y$$

*(This is beautifully simple!)*

**Step 2: Gradient w.r.t. weights $W^{(L)}$:**

$$\frac{\partial L}{\partial W^{(L)}} = \frac{\partial L}{\partial z^{(L)}} \cdot \frac{\partial z^{(L)}}{\partial W^{(L)}}$$

Since $z^{(L)} = W^{(L)} a^{(L-1)} + b^{(L)}$:
$$\frac{\partial z^{(L)}}{\partial W^{(L)}} = a^{(L-1)}$$

Therefore:
$$\frac{\partial L}{\partial W^{(L)}} = \delta^{(L)} \cdot (a^{(L-1)})^T$$

Where $\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}}$ (the "error" signal)

### 4.2 Hidden Layer Gradients

For hidden layer $l$ ($l < L$):

**Step 1: Gradient w.r.t. pre-activation:**

$$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

Where:
- $(W^{(l+1)})^T \delta^{(l+1)}$: Backpropagated error from next layer
- $\odot$: Element-wise multiplication
- $\sigma'(z^{(l)})$: Derivative of activation function

**Step 2: Gradient w.r.t. weights:**

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T$$

**Step 3: Gradient w.r.t. biases:**

$$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

### 4.3 Activation Function Derivatives

| Activation | Function | Derivative |
|------------|----------|------------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ |
| **Tanh** | $\tanh(z)$ | $1 - \tanh^2(z)$ |
| **ReLU** | $\max(0, z)$ | $1$ if $z>0$, $0$ otherwise |
| **Leaky ReLU** | $\max(0.01z, z)$ | $1$ if $z>0$, $0.01$ otherwise |
| **Softmax** | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | $a(\delta_{ij} - a_j)$ |

---

## 5. Complete Algorithm

### 5.1 Backpropagation Algorithm

```
Algorithm: Backpropagation

Input: Training data (x, y), network parameters, learning rate α

1. FORWARD PASS:
   For each layer l = 1 to L:
       z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
       a⁽ˡ⁾ = σ(z⁽ˡ⁾)

2. COMPUTE OUTPUT ERROR:
   δᴸ = (aᴸ - y) ⊙ σ'(zᴸ)    [for MSE]
   For cross-entropy + softmax:
       δᴸ = aᴸ - y           [simplified!]

3. BACKWARD PASS:
   For each layer l = L to 1:
       dW⁽ˡ⁾ = δ⁽ˡ⁾ ⊗ a⁽ˡ⁻¹⁾    (outer product)
       db⁽ˡ⁾ = δ⁽ˡ⁾
       
       If l > 1:
           δ⁽ˡ⁻¹⁾ = (W⁽ˡ⁾)ᵀ δ⁽ˡ⁾ ⊙ σ'(z⁽ˡ⁻¹⁾)

4. UPDATE PARAMETERS:
   For each layer l:
       W⁽ˡ⁾ = W⁽ˡ⁾ - α * dW⁽ˡ⁾
       b⁽ˡ⁾ = b⁽ˡ⁾ - α * db⁽ˡ⁾

5. Repeat for all training samples (or batch)
```

### 5.2 Vectorized Form

For a mini-batch of $m$ samples:

```
Forward:     Z⁽ˡ⁾ = W⁽ˡ⁾ A⁽ˡ⁻¹⁾ + b⁽ˡ⁾
             A⁽ˡ⁾ = σ(Z⁽ˡ⁾)

Backward:    δᴸ = Aᴸ - Y
             dW⁽ˡ⁾ = (1/m) δ⁽ˡ⁾ (A⁽ˡ⁻¹⁾)ᵀ
             db⁽ˡ⁾ = (1/m) sum(δ⁽ˡ⁾, axis=1)
             δ⁽ˡ⁻¹⁾ = (W⁽ˡ⁾)ᵀ δ⁽ˡ⁾ ⊙ σ'(Z⁽ˡ⁻¹⁾)
```

---

## 6. Implementation from Scratch

### 6.1 Complete MLP with Backpropagation

```python
import numpy as np

class MLP:
    """Multi-Layer Perceptron with Backpropagation."""
    
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_name = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Activation functions
        self.activations = {
            'sigmoid': (self._sigmoid, self._sigmoid_deriv),
            'tanh': (np.tanh, lambda x: 1 - np.tanh(x)**2),
            'relu': (self._relu, self._relu_deriv),
            'linear': (lambda x: x, lambda x: np.ones_like(x))
        }
        
        self.activation, self.activation_deriv = self.activations[activation]
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_deriv(self, x):
        return x * (1 - x)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _softmax_deriv(self, x):
        """Derivative of softmax (for backprop)."""
        return x  # Simplified: combined with cross-entropy gives x - y
    
    def forward(self, X):
        """Forward pass storing intermediate values."""
        self.activations_cache = [X]
        self.z_cache = []
        
        A = X
        for i in range(len(self.weights)):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_cache.append(Z)
            
            if i == len(self.weights) - 1:  # Output layer
                A = self._softmax(Z)
            else:
                A = self.activation(Z)
            
            self.activations_cache.append(A)
        
        return A
    
    def backward(self, y_onehot):
        """Backward pass computing gradients."""
        m = y_onehot.shape[0]
        
        # Output layer error (combined softmax + cross-entropy)
        delta = self.activations_cache[-1] - y_onehot
        
        # Gradients for output layer
        self.grad_weights = []
        self.grad_biases = []
        
        # Output layer gradient
        grad_w = self.activations_cache[-2].T @ delta / m
        grad_b = np.sum(delta, axis=0, keepdims=True) / m
        self.grad_weights.append(grad_w)
        self.grad_biases.append(grad_b)
        
        # Backprop through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i+1].T) * self.activation_deriv(self.activations_cache[i+1])
            
            grad_w = self.activations_cache[i].T @ delta / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            self.grad_weights.insert(0, grad_w)
            self.grad_biases.insert(0, grad_b)
    
    def fit(self, X, y, epochs=100, verbose=True):
        """Train the network."""
        # One-hot encode labels
        n_classes = len(np.unique(y))
        y_onehot = np.eye(n_classes)[y]
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(y_onehot)
            
            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * self.grad_weights[i]
                self.biases[i] -= self.learning_rate * self.grad_biases[i]
            
            if verbose and epoch % 10 == 0:
                predictions = np.argmax(output, axis=1)
                accuracy = np.mean(predictions == y)
                loss = -np.mean(y_onehot * np.log(output + 1e-10))
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    def predict(self, X):
        """Make predictions."""
        output = self.forward(X)
        return np.argmax(output, axis=1)
```

---

## 7. Optimizations and Best Practices

### 7.1 Gradient Checking

Verify implementation using numerical gradient:

```python
def gradient_check(model, X, y, epsilon=1e-7):
    """Check gradients numerically."""
    # Compute analytical gradient
    model.forward(X)
    y_onehot = np.eye(len(np.unique(y)))[y]
    model.backward(y_onehot)
    
    # Compute numerical gradient
    for i, (w, gw) in enumerate(zip(model.weights, model.grad_weights)):
        numerical_grad = np.zeros_like(w)
        
        for r in range(w.shape[0]):
            for c in range(w.shape[1]):
                w_plus = w.copy()
                w_plus[r, c] += epsilon
                model.weights[i] = w_plus
                loss_plus = -np.mean(y_onehot * np.log(model.forward(X) + 1e-10))
                
                w_minus = w.copy()
                w_minus[r, c] -= epsilon
                model.weights[i] = w_minus
                loss_minus = -np.mean(y_onehot * np.log(model.forward(X) + 1e-10))
                
                numerical_grad[r, c] = (loss_plus - loss_minus) / (2 * epsilon)
        
        model.weights[i] = w  # Restore
        
        # Compare
        diff = np.max(np.abs(gw - numerical_grad))
        print(f"Layer {i}: Max gradient difference = {diff}")
```

### 7.2 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Vanishing gradients** | Gradients very small, no learning | Use ReLU, residual connections, proper initialization |
| **Exploding gradients** | Gradients very large, NaN | Gradient clipping, proper scaling, lower LR |
| **Nan values** | Loss becomes NaN | Check for log(0), numerical stability |
| **Not learning** | Loss not decreasing | Check learning rate, initialization, data preprocessing |

### 7.3 Initialization

| Method | Formula | When to Use |
|--------|---------|-------------|
| **Xavier/Glorot** | $N(0, \sqrt{1/n_{in}})$ or $N(0, \sqrt{2/(n_{in}+n_{out})})$ | Sigmoid, tanh |
| **He** | $N(0, \sqrt{2/n_{in})})$ | ReLU |
| **Bias initialization** | 0 | Most cases |
| **Bias init for ReLU** | 0.01 | Can help with dead neurons |

---

## 8. Computational Complexity

### 8.1 Forward vs Backward

**Forward pass**: $O(L \cdot n^2)$ approximately
- Each layer: matrix multiplication

**Backward pass**: Also $O(L \cdot n^2)$
- Same computation as forward, just in reverse

**Total per epoch**: $O(2 \cdot L \cdot n^2)$ = similar to forward

### 8.2 Memory Usage

- **Activations**: Store for all layers (for backprop)
- **Gradients**: Computed on-the-fly or stored
- **Optimizer state**: Additional memory for momentum, Adam, etc.

---

## 9. Summary

| Step | What to Compute | Formula |
|------|-----------------|---------|
| **Forward** | Predictions | $a^{(l)} = \sigma(W^{(l)}a^{(l-1)} + b^{(l)})$ |
| **Output error** | $\delta^{(L)}$ | $a^{(L)} - y$ (for cross-entropy + softmax) |
| **Hidden error** | $\delta^{(l)}$ | $(W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$ |
| **Weight gradient** | $\frac{\partial L}{\partial W^{(l)}}$ | $\delta^{(l)} \otimes a^{(l-1)}$ |
| **Bias gradient** | $\frac{\partial L}{\partial b^{(l)}}$ | $\delta^{(l)}$ |
| **Update** | New weights | $W \leftarrow W - \alpha \cdot \frac{\partial L}{\partial W}$ |

**Key Insight**: Backpropagation is just the chain rule efficiently applied through the network. The error signal $\delta^{(l)}$ propagates backward, scaling at each layer by the weight matrix and activation derivative.