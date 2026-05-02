# Chapter 5: Linear Regression - From Theory to Implementation

> **Learning Duration:** 2 Days  
> **Difficulty:** Beginner  
> **Prerequisites:** Basic Python, linear algebra basics

---

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Understand the geometry and algebra of linear regression
- Implement linear regression from scratch using NumPy
- Understand gradient descent vs closed-form solutions
- Handle polynomial and multivariate regression
- Evaluate regression models with appropriate metrics

---

## 5.1 What Is Linear Regression?

### Intuitive Definition

**Linear regression** finds the best straight line (or plane) that fits through your data points.

```
      y
      │
  3.0 │        ●
      │      ●
  2.0 │    ●
      │  ●
  1.0 │●
      └────────────── x
         1   2   3
         
The line y = mx + b best represents these points
```

### Formal Definition

Given a set of input features $X$ and target variable $y$, linear regression finds parameters $\theta$ such that:

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$$

Or in vector form:

$$h_\theta(x) = \theta^T x$$

Where $\theta$ is the parameter vector and $x$ is the feature vector.

---

## 5.2 The Cost Function

### Mean Squared Error (MSE)

How do we measure "best fit"? We use the **Mean Squared Error**:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Where:
- $m$ = number of training examples
- $h_\theta(x^{(i)})$ = predicted value
- $y^{(i)}$ = actual value

### Why MSE?

1. **Squared** penalizes large errors more than small ones
2. **Mean** normalizes by sample size
3. **Smooth** - differentiable everywhere
4. **Always positive** - good for minimization

### Visual Interpretation

```
        Prediction error (residual)
        │
    y_i │        ● (actual)
        │         │
        │    y_hat│───●── (predicted)
        │         │
        │         │
        └─────────x───→ x_i
        
Residual = y_i - y_hat_i
MSE = (1/m) × Σ(residuals)²
```

---

## 5.3 Finding the Solution

### Method 1: Closed-Form (Normal Equation)

For small datasets, we can solve directly:

$$\theta = (X^T X)^{-1} X^T y$$

This is called the **Normal Equation**.

```python
import numpy as np

def linear_regression_closed_form(X, y):
    """
    Solve linear regression using normal equation.
    
    Args:
        X: Feature matrix (m x n), where m = samples, n = features
        y: Target vector (m,)
        
    Returns:
        theta: Parameter vector (n,)
    """
    # Add bias term (column of ones)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Solve: θ = (XᵀX)⁻¹Xᵀy
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    return theta
```

**When to use:**
- Small datasets ($n < 10,000$)
- Need exact solution
- No need for iterative optimization

**When NOT to use:**
- Large datasets (matrix inversion is $O(n^3)$)
- Singular matrices (non-invertible)

### Method 2: Gradient Descent

For larger datasets or when normal equation fails:

```python
def linear_regression_gradient_descent(X, y, learning_rate=0.01, 
                                        n_iterations=1000):
    """
    Solve linear regression using gradient descent.
    
    Args:
        X: Feature matrix (m x n)
        y: Target vector (m,)
        learning_rate: Step size for gradient descent
        n_iterations: Number of iterations
        
    Returns:
        theta: Parameter vector (n+1,) including bias
    """
    m, n = X.shape
    
    # Add bias term
    X_b = np.c_[np.ones((m, 1)), X]
    
    # Initialize parameters to zeros
    theta = np.zeros(n + 1)
    
    for i in range(n_iterations):
        # Compute predictions
        predictions = X_b @ theta
        
        # Compute error
        error = predictions - y
        
        # Compute gradient: (1/m) * Xᵀ * (Xθ - y)
        gradient = (1/m) * X_b.T @ error
        
        # Update parameters
        theta = theta - learning_rate * gradient
        
    return theta
```

### Comparison

| Aspect | Normal Equation | Gradient Descent |
|--------|-----------------|------------------|
| Speed | $O(n^3)$ per solve | $O(m \times n \times \text{iterations})$ |
| Accuracy | Exact | Approximate (depends on convergence) |
| Memory | High ($X^TX$) | Low |
| Large n | Slow | Better |
| Numerical issues | Can fail | More robust |

---

## 5.4 Implementation From Scratch

### Complete Implementation

```python
import numpy as np
from typing import Tuple, Optional

class LinearRegression:
    """
    Complete Linear Regression implementation.
    
    Supports both closed-form and gradient descent solutions.
    
    Attributes:
        theta: Learned parameters (including bias)
        n_iterations: Number of gradient descent iterations (if using GD)
        learning_rate: Learning rate for gradient descent
    """
    
    def __init__(self, method: str = 'closed_form', 
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 regularization: float = 0.0):
        """
        Initialize Linear Regression.
        
        Args:
            method: 'closed_form' or 'gradient_descent'
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations for GD
            regularization: L2 regularization strength (lambda)
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.theta = None
        self.loss_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model.
        
        Args:
            X: Feature matrix of shape (m, n)
            y: Target vector of shape (m,)
            
        Returns:
            self: Fitted model
        """
        m, n = X.shape
        
        # Add bias term
        X_b = np.c_[np.ones((m, 1)), X]
        
        if self.method == 'closed_form':
            self.theta = self._fit_closed_form(X_b, y)
        else:
            self.theta = self._fit_gradient_descent(X_b, y)
            
        return self
    
    def _fit_closed_form(self, X_b: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve using normal equation with regularization.
        
        θ = (XᵀX + λI)⁻¹Xᵀy
        
        Regularization helps with ill-conditioned matrices.
        """
        n_features = X_b.shape[1]
        
        # Add regularization to avoid singular matrix
        # (don't regularize bias term - index 0)
        regularization_matrix = self.regularization * np.eye(n_features)
        regularization_matrix[0, 0] = 0
        
        # Solve: (XᵀX + λI)⁻¹Xᵀy
        XtX = X_b.T @ X_b
        XtX_reg = XtX + regularization_matrix
        
        self.theta = np.linalg.inv(XtX_reg) @ X_b.T @ y
        
        return self.theta
    
    def _fit_gradient_descent(self, X_b: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve using gradient descent with regularization.
        
        Gradient of J(θ) = (1/m)Xᵀ(Xθ - y) + (λ/m)θ
        (regularization doesn't apply to bias)
        """
        m = len(y)
        n_features = X_b.shape[1]
        theta = np.zeros(n_features)
        
        self.loss_history = []
        
        for i in range(self.n_iterations):
            # Compute predictions
            predictions = X_b @ theta
            
            # Compute error
            error = predictions - y
            
            # Compute gradient
            gradient = (1/m) * X_b.T @ error
            
            # Add regularization (not to bias)
            gradient[1:] += (self.regularization / m) * theta[1:]
            
            # Update
            theta = theta - self.learning_rate * gradient
            
            # Record loss
            loss = self._compute_loss(X_b, y, theta)
            self.loss_history.append(loss)
            
        return theta
    
    def _compute_loss(self, X_b: np.ndarray, y: np.ndarray, 
                      theta: np.ndarray) -> float:
        """Compute MSE loss with regularization."""
        m = len(y)
        predictions = X_b @ theta
        mse = (1/(2*m)) * np.sum((predictions - y) ** 2)
        
        # Add regularization (not to bias)
        reg_term = (self.regularization / (2*m)) * np.sum(theta[1:] ** 2)
        
        return mse + reg_term
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix of shape (m, n)
            
        Returns:
            Predictions of shape (m,)
        """
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        return X_b @ self.theta
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² coefficient of determination.
        
        R² = 1 - SS_res / SS_tot
        
        Where:
        - SS_res = Σ(y - ŷ)² (residual sum of squares)
        - SS_tot = Σ(y - ȳ)² (total sum of squares)
        """
        predictions = self.predict(X)
        
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot)
    
    def coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bias and weights separately."""
        return self.theta[0], self.theta[1:]
```

---

## 5.5 Usage Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Create and train model
model = LinearRegression(method='gradient_descent', 
                         learning_rate=0.1,
                         n_iterations=1000)
model.fit(X, y)

# Make predictions
X_test = np.array([[0], [2]])
predictions = model.predict(X_test)

# Evaluate
r2 = model.score(X, y.flatten())
print(f"R² Score: {r2:.4f}")
print(f"Bias: {model.theta[0]:.4f}")
print(f"Slope: {model.theta[1]:.4f}")

# Plot
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X_test, predictions, 'r-', linewidth=2, label='Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

---

## 5.6 Multivariate Linear Regression

When we have multiple features:

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$$

```python
# Multiple features example
np.random.seed(42)
m = 100

# Three features: size, bedrooms, age
X = np.random.rand(m, 3) * np.array([3000, 5, 50])  # realistic ranges
X[:, 0] += 500  # offset for size

# Target: price = 100k + 150*size + 20k*bedrooms - 500*age + noise
y = 100000 + 150 * X[:, 0] + 20000 * X[:, 1] - 500 * X[:, 2] + np.random.randn(m) * 20000

# Train
model = LinearRegression(method='closed_form')
model.fit(X, y)

# Check R²
print(f"R² Score: {model.score(X, y):.4f}")
print(f"Bias (base price): ${model.theta[0]:,.0f}")
print(f"Coefficients: {model.theta[1:]}")
```

---

## 5.7 Polynomial Regression

When the relationship isn't linear, we can add polynomial terms:

```python
def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Create polynomial features."""
    m = X.shape[0]
    poly_X = X.copy()
    
    for d in range(2, degree + 1):
        poly_X = np.c_[poly_X, X ** d]
    
    return poly_X

# Example: Quadratic relationship
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = X**2 + np.random.randn(100, 1) * 0.5  # y = x² + noise

# Transform to polynomial
X_poly = polynomial_features(X, 2)

# Train linear model on polynomial features
model = LinearRegression(method='closed_form')
model.fit(X_poly, y)

# Plot
plt.scatter(X, y, alpha=0.5)
plt.plot(X, model.predict(X_poly), 'r-', linewidth=2)
plt.show()
```

---

## 5.8 Model Evaluation Metrics

### Metrics for Regression

```python
def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute multiple evaluation metrics.
    
    Returns:
        Dictionary with various metrics
    """
    m = len(y_true)
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }
```

| Metric | Formula | Best | Use Case |
|--------|---------|------|----------|
| MSE | $\frac{1}{m}\sum(y-\hat{y})^2$ | 0 | Standard |
| RMSE | $\sqrt{MSE}$ | 0 | Same scale as y |
| MAE | $\frac{1}{m}\sum|y-\hat{y}|$ | 0 | Robust to outliers |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | 1 | Goodness of fit |
| MAPE | $\frac{1}{m}\frac{|y-\hat{y}|}{|y|}$ × 100 | 0 | Percentage error |

---

## 5.9 Regularization

### Why Regularize?

When model is too complex or data is limited, it can **overfit** - memorize training data instead of learning general patterns.

Regularization adds a penalty to the cost function:

$$J_{reg}(\theta) = J(\theta) + \lambda \cdot R(\theta)$$

### L2 Regularization (Ridge)

$$R(\theta) = \sum_{i=1}^{n} \theta_i^2$$

Penalizes large weights, but doesn't make them exactly zero.

```python
# In our LinearRegression class:
# Already implemented with regularization parameter
model = LinearRegression(regularization=1.0)
```

### L1 Regularization (Lasso)

$$R(\theta) = \sum_{i=1}^{n} |\theta_i|$$

Can drive weights to exactly zero (feature selection).

### Comparison

```
Ridge (L2):          Lasso (L1):
    w                    w
     │                    │
     │  ╲                 │  ＋──→ zero
     │   ╲                │
     │    ●               │──●
     │                    │
     └────────            └────────
     
     Shrinks toward 0    Can reach exactly 0
```

---

## 📝 Summary

### Key Takeaways

1. **Linear regression** finds parameters that minimize MSE
2. **Normal equation** gives closed-form solution: $\theta = (X^T X)^{-1} X^T y$
3. **Gradient descent** is better for large datasets
4. **R²** measures how well model explains variance
5. **Regularization** prevents overfitting
6. **Polynomial features** can model non-linear relationships

### Implementation Checklist

- [x] Add bias term to feature matrix
- [x] Initialize parameters
- [x] Choose optimization method (closed-form vs GD)
- [x] Compute and minimize cost function
- [x] Evaluate with metrics
- [x] Consider regularization if needed

---

## ❓ Quick Check

1. What does MSE measure?
2. When would you prefer gradient descent over normal equation?
3. What is the purpose of regularization?
4. How do you interpret R² = 0.85?

*Answers at end of chapter*