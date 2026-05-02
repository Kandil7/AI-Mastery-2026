# Chapter 3: Calculus and Optimization for Machine Learning

> **Learning Duration:** 4 Days  
> **Difficulty:** Intermediate  
> **Prerequisites:** Chapters 1-2 (Vectors, Matrices)

---

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Understand derivatives as rates of change and slopes
- Compute partial derivatives and gradients
- Understand the chain rule for composite functions
- Apply gradient descent to find minima
- Implement optimization algorithms from scratch

---

## 3.1 The Concept of Derivatives

### Intuitive Understanding

A **derivative** measures how a function changes as its input changes. It's the instantaneous rate of change.

**Key Interpretations:**
1. **Slope**: At any point, the derivative is the slope of the tangent line
2. **Rate of Change**: How fast output changes with respect to input
3. **Sensitivity**: How sensitive the output is to input changes

### Mathematical Definition

The derivative of $f(x)$ at point $x$ is:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

This is the slope of the tangent line as $h$ approaches zero.

### Visual Representation

```
        f(x)
          │
    f(x+h)│──────────●
          │        ╱│
          │      ╱  │
          │    ╱    │ ← h
          │  ╱      │
    f(x)  │●────────┘
          └────────────── x
              x    x+h
              
Derivative = slope of tangent = (f(x+h) - f(x)) / h as h → 0
```

### Common Derivatives

| Function | Derivative |
|----------|------------|
| $f(x) = x^n$ | $f'(x) = nx^{n-1}$ |
| $f(x) = e^x$ | $f'(x) = e^x$ |
| $f(x) = \ln(x)$ | $f'(x) = 1/x$ |
| $f(x) = \sin(x)$ | $f'(x) = \cos(x)$ |
| $f(x) = \cos(x)$ | $f'(x) = -\sin(x)$ |

---

## 3.2 Partial Derivatives

### Why Partial Derivatives?

In ML, we work with functions of **multiple variables**:

$$f(x_1, x_2, ..., x_n) = L(\theta_1, \theta_2, ..., \theta_m)$$

A **partial derivative** measures how $f$ changes as we vary **one** variable while keeping others constant.

### Definition

The partial derivative with respect to $x_1$:

$$\frac{\partial f}{\partial x_1} = \lim_{h \to 0} \frac{f(x_1+h, x_2, ..., x_n) - f(x_1, x_2, ..., x_n)}{h}$$

**Interpretation:** "If I change $x_1$ slightly while holding everything else constant, how does $f$ change?"

### Example: Gradient of a Function

```python
import numpy as np

def f(x1, x2):
    """f(x1, x2) = x1² + x2²"""
    return x1**2 + x2**2

def gradient(x1, x2):
    """Gradient = [∂f/∂x1, ∂f/∂x2]"""
    df_dx1 = 2 * x1
    df_dx2 = 2 * x2
    return np.array([df_dx1, df_dx2])

# Gradient at point (3, 4)
grad = gradient(3, 4)  # [6, 8]
```

### The Gradient Vector

For function $f(x_1, x_2, ..., x_n)$, the **gradient** is:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ ... \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Key Property:** The gradient points in the **direction of steepest ascent** (maximum increase).

---

## 3.3 The Chain Rule

### Why Do We Need It?

Neural networks are **composite functions** - functions within functions:

$$f(g(h(x)))$$

The **chain rule** tells us how to differentiate composite functions.

### Single Variable Chain Rule

If $y = f(g(x))$, then:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

### Multiple Variables Chain Rule

If $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$:

$$\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}$$

### Neural Network Example

```
Output = f(W2 · g(W1 · x + b1) + b2)

∂Output/∂W1 = ∂Output/∂g · ∂g/∂(W1·x) · ∂(W1·x)/∂W1
```

This is exactly how **backpropagation** works!

---

## 3.4 Gradient Descent

### The Core Idea

In ML, we want to **minimize** a loss function:

$$\theta^* = \arg\min_\theta L(\theta)$$

**Gradient descent** finds the minimum by:
1. Computing the gradient
2. Moving in the opposite direction (downhill)
3. Repeating until convergence

### The Algorithm

```
1. Initialize θ (random or zeros)
2. Repeat until convergence:
   a. Compute gradient: ∇L(θ)
   b. Update: θ = θ - α · ∇L(θ)
   
   Where α = learning rate (step size)
```

### Visual Representation

```
                    ● Minimum
                   ╱
                 ╱
               ╱
            ╱  Current position
         ╱ ←── Gradient points UP, so go DOWN (opposite)
      ╱
   ● Start
```

### Implementation

```python
def gradient_descent(gradient_fn, initial_point, learning_rate=0.1, 
                     max_iterations=1000, tolerance=1e-6):
    """
    Basic gradient descent implementation.
    
    Args:
        gradient_fn: Function that computes gradient
        initial_point: Starting point (numpy array)
        learning_rate: Step size (alpha)
        max_iterations: Maximum iterations
        tolerance: Convergence threshold
        
    Returns:
        optimal_point: Found minimum
        history: List of points visited
    """
    point = initial_point.copy()
    history = [point.copy()]
    
    for i in range(max_iterations):
        grad = gradient_fn(point)
        
        # Check convergence
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged at iteration {i}")
            break
            
        # Update (move opposite to gradient)
        point = point - learning_rate * grad
        history.append(point.copy())
    
    return point, history
```

### Choosing Learning Rate

| Learning Rate | Behavior |
|---------------|----------|
| Too small | Slow convergence |
| Just right | Good convergence |
| Too large | Oscillation/divergence |

```
Learning Rate = 0.01       Learning Rate = 0.5       Learning Rate = 1.1
      ↓                         ↘↙↘↙                      ↔↔↔
    ●●●●●                      ●↘  ↙ ●                 ● ↔ × (diverges)
```

---

## 3.5 Variants of Gradient Descent

### 3.5.1 Vanilla Gradient Descent

Uses the **entire dataset** for each gradient computation:

$$\theta = \theta - \alpha \nabla L(\theta)$$

**Pros:** Direct path to minimum
**Cons:** Slow for large datasets

```python
def vanilla_gd(X, y, learning_rate=0.01, epochs=100):
    """Vanilla gradient descent"""
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        # Compute gradient using ALL data
        gradient = (1/m) * X.T @ (X @ theta - y)
        theta = theta - learning_rate * gradient
        
    return theta
```

### 3.5.2 Stochastic Gradient Descent (SGD)

Uses **one sample** at a time:

$$\theta = \theta - \alpha \nabla L_i(\theta)$$

**Pros:** Fast, can escape local minima
**Cons:** Noisy, may oscillate

```python
def sgd(X, y, learning_rate=0.01, epochs=100):
    """Stochastic gradient descent"""
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        
        for i in indices:
            xi = X[i:i+1]  # Single sample
            yi = y[i]
            # Gradient for single sample
            gradient = xi.T @ (xi @ theta - yi)
            theta = theta - learning_rate * gradient
            
    return theta
```

### 3.5.3 Mini-batch Gradient Descent

Uses **batches** of samples:

```python
def mini_batch_gd(X, y, batch_size=32, learning_rate=0.01, epochs=100):
    """Mini-batch gradient descent"""
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            batch_idx = indices[start:end]
            
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            # Gradient for batch
            gradient = (1/batch_size) * X_batch.T @ (X_batch @ theta - y_batch)
            theta = theta - learning_rate * gradient
            
    return theta
```

### Comparison

| Method | Gradient Computation | Speed | Noise |
|--------|----------------------|-------|-------|
| Vanilla | Full dataset | Slow | Low |
| SGD | One sample | Fast | High |
| Mini-batch | Batches | Medium | Medium |

---

## 3.6 Advanced Optimizers

### 3.6.1 Momentum

Adds **momentum** to overcome local minima and reduce oscillation:

$$v_t = \beta v_{t-1} + \alpha \nabla L(\theta)$$
$$\theta = \theta - v_t$$

Where $\beta$ is momentum (typically 0.9)

```python
def momentum_gd(X, y, learning_rate=0.01, momentum=0.9, epochs=100):
    """Gradient descent with momentum"""
    m, n = X.shape
    theta = np.zeros(n)
    velocity = np.zeros(n)
    
    for epoch in range(epochs):
        gradient = (1/m) * X.T @ (X @ theta - y)
        
        # Update velocity with momentum
        velocity = momentum * velocity + learning_rate * gradient
        theta = theta - velocity
        
    return theta
```

### 3.6.2 RMSprop

Adaptive learning rate per parameter:
- Divides gradient by running average of magnitudes

```python
def rmsprop(X, y, learning_rate=0.01, decay=0.9, epsilon=1e-8, epochs=100):
    """RMSprop optimizer"""
    m, n = X.shape
    theta = np.zeros(n)
    cache = np.zeros(n)  # Running average of squared gradients
    
    for epoch in range(epochs):
        gradient = (1/m) * X.T @ (X @ theta - y)
        
        # Update cache
        cache = decay * cache + (1 - decay) * gradient**2
        
        # Update with adaptive learning rate
        theta = theta - learning_rate * gradient / (np.sqrt(cache) + epsilon)
        
    return theta
```

### 3.6.3 Adam (Adaptive Moment Estimation)

Combines momentum and RMSprop:
- Uses moving average of gradients (momentum)
- Uses moving average of squared gradients (RMSprop)

```python
def adam(X, y, learning_rate=0.001, beta1=0.9, beta2=0.999, 
         epsilon=1e-8, epochs=100):
    """Adam optimizer"""
    m, n = X.shape
    theta = np.zeros(n)
    
    m_t = np.zeros(n)  # First moment (momentum)
    v_t = np.zeros(n)  # Second moment (RMSprop)
    
    for t in range(1, epochs + 1):
        gradient = (1/m) * X.T @ (X @ theta - y)
        
        # Update biased first moment estimate
        m_t = beta1 * m_t + (1 - beta1) * gradient
        
        # Update biased second moment estimate  
        v_t = beta2 * v_t + (1 - beta2) * gradient**2
        
        # Bias correction
        m_hat = m_t / (1 - beta1**t)
        v_hat = v_t / (1 - beta2**t)
        
        # Update parameters
        theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
    return theta
```

### Visual Comparison

```
SGD:              Momentum:          Adam:
  ●                  ●                   ●
   ●                  ↘                   ↘●
    ●                   ↘ ●                 ↘ ●
     ●                  ↘   ●                ↘   ●
      ●                 ↘     ●               ↘     ●
       ●                ↘       ●             ↘       ●
        ●               ↘         ●            ↘         ●
```

---

## 3.7 Second-Order Methods (Overview)

### Newton's Method

Uses **Hessian** (second derivatives) for faster convergence:

$$\theta_{t+1} = \theta_t - H^{-1} \nabla L(\theta_t)$$

**Pros:** Quadratic convergence (fast!)
**Cons:** Expensive to compute Hessian

### Quasi-Newton Methods

Approximate Hessian without full computation:
- BFGS
- L-BFGS

---

## 3.8 Practical Considerations

### 3.8.1 Convergence Criteria

1. **Gradient magnitude**: Stop when $||\nabla L|| < \epsilon$
2. **Loss change**: Stop when $|L_{t} - L_{t-1}| < \epsilon$
3. **Maximum iterations**: Hard limit

### 3.8.2 Learning Rate Scheduling

```python
def learning_rate_schedule(learning_rate, epoch, schedule_type='step'):
    if schedule_type == 'step':
        return learning_rate * (0.1 ** (epoch // 30))
    elif schedule_type == 'exponential':
        return learning_rate * (0.95 ** epoch)
    elif schedule_type == 'cosine':
        return learning_rate * (1 + np.cos(np.pi * epoch / 100)) / 2
```

### 3.8.3 Feature Scaling

Gradient descent converges faster when features are normalized:

```python
def normalize(X):
    """Mean normalization"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std
```

---

## 3.9 Application to Machine Learning

### 3.9.1 Linear Regression

**Loss Function:** Mean Squared Error (MSE)

$$L(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

**Gradient:**

$$\frac{\partial L}{\partial \theta_j} = \frac{2}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

```python
def linear_regression_gd(X, y, learning_rate=0.01, epochs=1000):
    """Linear regression with gradient descent"""
    m, n = X.shape
    X = np.c_[np.ones(m), X]  # Add bias
    theta = np.zeros(n + 1)
    
    for epoch in range(epochs):
        predictions = X @ theta
        error = predictions - y
        gradient = (2/m) * X.T @ error
        theta = theta - learning_rate * gradient
        
    return theta
```

### 3.9.2 Logistic Regression

**Loss Function:** Cross-Entropy

$$L(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

### 3.9.3 Neural Networks (Backpropagation)

Uses chain rule to compute gradients through layers:

```
Forward:  x → h₁ → h₂ → ... → output
                    ↑
Backward:  ← gradient flows back through each layer
```

---

## 📝 Summary

### Key Takeaways

1. **Derivatives** measure instantaneous rate of change
2. **Gradients** point in direction of steepest ascent
3. **Gradient descent** finds minima by moving opposite to gradient
4. **Learning rate** controls step size (critical for convergence)
5. **Momentum, RMSprop, Adam** improve convergence speed and stability
6. **Chain rule** enables backpropagation in neural networks

### Algorithms Summary

| Optimizer | Update Rule | Best For |
|-----------|-------------|----------|
| Vanilla GD | $\theta - \alpha \nabla L$ | Small datasets |
| SGD | $\theta - \alpha \nabla L_i$ | Large datasets |
| Momentum | $\theta - \alpha(v + \nabla L)$ | Escaping local minima |
| RMSprop | $\theta - \alpha \nabla L / \sqrt{cache}$ | Adaptive rates |
| Adam | $\theta - \alpha \cdot m / \sqrt{v}$ | Default choice |

---

## ❓ Quick Check

1. What does the gradient vector point to?
2. What's the difference between SGD and mini-batch GD?
3. Why is feature scaling important for gradient descent?
4. How does momentum help optimization?

*Answers at end of chapter*