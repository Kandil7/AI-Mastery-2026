# 📈 Module 2.1: Classical ML - Linear & Logistic Regression

**Master the foundations of machine learning**

---

## 📋 Overview

**Duration:** 12 hours  
**Difficulty:** ⭐⭐⭐☆☆  
**Prerequisites:** Tier 1 (Fundamentals), Python, NumPy, basic calculus

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- ✅ Derive linear regression from first principles
- ✅ Implement gradient descent for regression
- ✅ Understand the bias-variance tradeoff
- ✅ Derive logistic regression for classification
- ✅ Implement sigmoid function and cross-entropy loss
- ✅ Evaluate models with proper metrics
- ✅ Apply regularization to prevent overfitting
- ✅ Build production-ready ML pipelines

---

## 📚 Module Structure

```
Part 1: Linear Regression Theory (3 hours)
  ↓
Part 2: Linear Regression Implementation (3 hours)
  ↓
Part 3: Logistic Regression Theory (3 hours)
  ↓
Part 4: Logistic Regression Implementation (3 hours)
```

---

## Part 1: Linear Regression Theory (3 hours)

### 1.1 What is Linear Regression?

**Goal:** Predict a continuous value based on input features.

**Equation:**
```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Where:
- **y** = predicted value
- **xᵢ** = input features
- **wᵢ** = weights (learned from data)
- **b** = bias (intercept)

### 1.2 The Cost Function

**Mean Squared Error (MSE):**
```
J(w, b) = (1/2m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

Where:
- **m** = number of training examples
- **h(x)** = hypothesis (prediction)
- **y** = actual value

**Why MSE?**
- Penalizes large errors more
- Differentiable (needed for gradient descent)
- Convex (guaranteed global minimum)

### 1.3 Gradient Descent

**Update Rules:**
```
wⱼ := wⱼ - α × ∂J/∂wⱼ
b := b - α × ∂J/∂b
```

Where:
- **α** = learning rate
- **∂J/∂wⱼ** = partial derivative of cost w.r.t. wⱼ

**Derivatives:**
```
∂J/∂wⱼ = (1/m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) × xⱼ⁽ⁱ⁾
∂J/∂b = (1/m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)
```

### 1.4 Bias-Variance Tradeoff

**Bias:** Error from overly simple assumptions
- High bias → Underfitting
- Misses patterns in data

**Variance:** Error from sensitivity to training data
- High variance → Overfitting
- Memorizes noise instead of signal

**Sweet Spot:** Balance bias and variance for best generalization

---

## Part 2: Linear Regression Implementation (3 hours)

### 2.1 Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """Linear Regression implemented from scratch."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_lambda=0.0, reg_type=None):
        """
        Initialize linear regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of gradient descent iterations
            reg_lambda: Regularization strength
            reg_type: 'l1' (Lasso), 'l2' (Ridge), or None
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Args:
            X: Training features (m, n)
            y: Training targets (m,)
        """
        m, n = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self._predict(X)
            
            # Calculate cost
            cost = self._calculate_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)
            
            # Add regularization
            if self.reg_type == 'l2':  # Ridge
                dw += (self.reg_lambda / m) * self.weights
            elif self.reg_type == 'l1':  # Lasso
                dw += (self.reg_lambda / m) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def _predict(self, X):
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias
    
    def _calculate_cost(self, y, y_pred):
        """Calculate cost with optional regularization."""
        m = len(y)
        mse = (1 / (2 * m)) * np.sum((y - y_pred) ** 2)
        
        if self.reg_type == 'l2':
            reg_term = (self.reg_lambda / (2 * m)) * np.sum(self.weights ** 2)
            return mse + reg_term
        elif self.reg_type == 'l1':
            reg_term = (self.reg_lambda / m) * np.sum(np.abs(self.weights))
            return mse + reg_term
        
        return mse
    
    def predict(self, X):
        """Predict on new data."""
        return self._predict(X)
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
```

### 2.2 Test on Sample Data

```python
# Generate sample data
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# Train model
model = LinearRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X, y.flatten())

# Results
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias:.4f}")
print(f"R² Score: {model.score(X, y.flatten()):.4f}")

# Plot cost history
plt.figure(figsize=(10, 5))
plt.plot(model.cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Linear Regression - Cost Convergence')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Part 3: Logistic Regression Theory (3 hours)

### 3.1 What is Logistic Regression?

**Goal:** Predict binary classification (0 or 1).

**Key Difference from Linear Regression:**
- Uses sigmoid function to output probabilities
- Optimizes cross-entropy loss instead of MSE

### 3.2 The Sigmoid Function

```
σ(z) = 1 / (1 + e^(-z))
```

**Properties:**
- Output range: (0, 1)
- σ(0) = 0.5
- σ(∞) = 1
- σ(-∞) = 0

### 3.3 Hypothesis

```
h(x) = σ(w·x + b) = 1 / (1 + e^(-(w·x + b)))
```

**Interpretation:**
- h(x) ≥ 0.5 → Predict class 1
- h(x) < 0.5 → Predict class 0

### 3.4 Cross-Entropy Loss

```
J(w, b) = -(1/m) × Σ[y⁽ⁱ⁾log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h(x⁽ⁱ⁾))]
```

**Why Cross-Entropy?**
- Penalizes confident wrong predictions heavily
- Convex for logistic regression
- Better than MSE for classification

### 3.5 Gradient Descent

**Update Rules (same form as linear regression):**
```
wⱼ := wⱼ - α × ∂J/∂wⱼ
b := b - α × ∂J/∂b
```

**But gradients are different:**
```
∂J/∂wⱼ = (1/m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) × xⱼ⁽ⁱ⁾
∂J/∂b = (1/m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)
```

(Same form, but h(x) is now sigmoid!)

---

## Part 4: Logistic Regression Implementation (3 hours)

### 4.1 Implementation from Scratch

```python
class LogisticRegression:
    """Logistic Regression implemented from scratch."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_lambda=0.0):
        """
        Initialize logistic regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of iterations
            reg_lambda: L2 regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Args:
            X: Training features (m, n)
            y: Training labels (m,) binary
        """
        m, n = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_pred)
            
            # Calculate cost (cross-entropy)
            cost = self._calculate_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)
            
            # Add L2 regularization
            if self.reg_lambda > 0:
                dw += (self.reg_lambda / m) * self.weights
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def _calculate_cost(self, y, y_pred):
        """Calculate cross-entropy loss."""
        m = len(y)
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -(1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # Add regularization
        if self.reg_lambda > 0:
            reg_term = (self.reg_lambda / (2 * m)) * np.sum(self.weights ** 2)
            cost += reg_term
        
        return cost
    
    def predict_proba(self, X):
        """Predict probability of class 1."""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

### 4.2 Test on Classification Data

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Generate binary classification data
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Detailed metrics
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

---

## 📊 Evaluation Metrics

### Regression Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | (1/m)Σ(y-ŷ)² | Lower is better |
| **RMSE** | √MSE | Same units as y |
| **R²** | 1 - SS_res/SS_tot | % variance explained |
| **MAE** | (1/m)Σ|y-ŷ| | Robust to outliers |

### Classification Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP+TN)/Total | Balanced classes |
| **Precision** | TP/(TP+FP) | Minimize false positives |
| **Recall** | TP/(TP+FN) | Minimize false negatives |
| **F1-Score** | 2×(Prec×Rec)/(Prec+Rec) | Balance precision/recall |

---

## ✅ Module Checklist

- [ ] Derived linear regression cost function
- [ ] Implemented gradient descent from scratch
- [ ] Understood bias-variance tradeoff
- [ ] Derived logistic regression with sigmoid
- [ ] Implemented cross-entropy loss
- [ ] Evaluated models with proper metrics
- [ ] Applied L1/L2 regularization
- [ ] Tested on real datasets

---

## 📖 Additional Resources

### Papers
- [Original] "Linear Regression" - Standard statistical text
- [Regularization] "Regression Shrinkage and Selection via the Lasso" - Tibshirani (1996)

### Videos
- [Andrew Ng] Linear Regression Explained
- [StatQuest] Logistic Regression Clearly Explained

### Exercises
1. Implement stochastic gradient descent
2. Add early stopping to prevent overfitting
3. Compare L1 vs L2 regularization
4. Build polynomial regression extension

---

**Module Created:** April 2, 2026  
**Last Updated:** April 2, 2026  
**Estimated Time:** 12 hours  
**Difficulty:** Intermediate

---

[← Back to Tier 2 Catalog](../README.md) | [Next: Decision Trees](module-02-tree-models/README.md)
