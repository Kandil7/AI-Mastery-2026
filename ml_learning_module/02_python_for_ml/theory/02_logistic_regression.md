# Logistic Regression: Theory and Implementation

## Introduction

Logistic Regression is a fundamental classification algorithm that predicts discrete categorical outcomes. Despite its name containing "regression," it solves **classification problems**, not regression problems.

### Why Not Linear Regression for Classification?

Consider predicting whether an email is spam (1) or not (0):

```
Linear Regression Output:
┌─────────────────────────────────────────────────────┐
│                                                     │
│   y = θ₀ + θ₁*x₁ + θ₂*x₂ + ... + θₙ*xₙ            │
│                                                     │
│   Problem: Predictions can be negative or > 1!     │
│   Example: y = -0.5 (invalid probability)          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Issues with using linear regression for classification:**
1. Predictions can be outside [0, 1] range
2. No guarantee of proper probability estimation
3. Sensitive to outliers
4. Assumes linear decision boundary only for balanced classes

---

## 1. The Logistic (Sigmoid) Function

### 1.1 Definition

The logistic function (also called sigmoid) squashes any real number into the [0, 1] range:

$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{e^z + 1}$$

**Properties:**
- $\sigma(z) \to 0$ as $z \to -\infty$
- $\sigma(0) = 0.5$
- $\sigma(z) \to 1$ as $z \to +\infty$
- $\sigma(-z) = 1 - \sigma(z)$ (symmetry)
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

### 1.2 Visual Representation

```
σ(z)
 │
1┼──────────────────────────────────
 │                    ╱──────────
 │                  ╱
 │                ╱
 │              ╱
 │            ╱
 │          ╱
0┼─────────╱────────────────────────
  -4   -2   0   2   4   6   8   z
        ╱╲
       ╱  ╲    σ(z) = 1 / (1 + e^(-z))
      ╱    ╲
```

### 1.3 Mathematical Properties

**Derivative:**
$$\frac{d}{dz}\sigma(z) = \sigma(z)(1 - \sigma(z))$$

**Log-odds interpretation:**
If $p = \sigma(z)$, then:
$$z = \log\left(\frac{p}{1-p}\right)$$

This is the **log-odds** or **logit** function - the inverse of sigmoid.

---

## 2. Logistic Regression Model

### 2.1 Hypothesis Function

For binary classification with features $x = (x_1, x_2, ..., x_n)$:

$$h_\theta(x) = \sigma(\theta^T x) = \sigma(\theta_0 + \theta_1 x_1 + ... + \theta_n x_n)$$

Expanded:
$$h_\theta(x) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x_1 + ... + \theta_n x_n)}}$$

**Interpretation:** $h_\theta(x)$ = estimated probability that $y = 1$ given features $x$

### 2.2 Decision Boundary

The classifier predicts:
- $y = 1$ if $h_\theta(x) \geq 0.5$ (i.e., $\theta^T x \geq 0$)
- $y = 0$ if $h_\theta(x) < 0.5$ (i.e., $\theta^T x < 0$)

The **decision boundary** is where $\theta^T x = 0$.

**Linear Decision Boundary:**
$$\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0$$

This is a line in 2D, a plane in 3D, a hyperplane in higher dimensions.

### 2.3 Example: Binary Classification

```
Decision Boundary (θ₀ + θ₁x₁ + θ₂x₂ = 0)

x₂
 │
6┼    ● ● ●                    ○ ○ ○
 │  ● ● ● ●                  ○ ○ ○ ○
 │● ● ● ●                    ○ ○ ○
 │──────────────────────────────
 │  ○ ○ ○                    ● ● ●
 │○ ○ ○ ○                    ● ● ● ●
4┼  ○ ○ ○                    ● ● ●
 │
 │   Class 0 (y=0)         Class 1 (y=1)
 │
0┼────────────────────────────────────────
  0         2         4         6         8  x₁
```

---

## 3. Cost Function

### 3.1 Why Not Mean Squared Error?

If we use MSE loss for logistic regression:
$$J(\theta) = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$$

This cost function is **non-convex** with local minima!

```
Non-convex MSE for Logistic Regression

      Cost
        │
   high │    ╲    ╱╲    ╱
        │     ╲  ╱   ╲  ╱
        │      ╲╱     ╲╱
 low    │──────────────────  θ
        │    local      global
        │    minima     minimum
```

### 3.2 Cross-Entropy Loss (Log Loss)

The proper cost function for logistic regression:

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

**Intuition:**

| Case | y | h(x) | -y log(h) | -(1-y)log(1-h) |
|------|---|------|-----------|----------------|
| Correct 1 | 1 | 0.9 | -log(0.9) ≈ 0.105 | 0 |
| Correct 1 | 1 | 0.99 | -log(0.99) ≈ 0.01 | 0 |
| Wrong 1 | 1 | 0.1 | -log(0.1) ≈ 2.3 | 0 |
| Correct 0 | 0 | 0.1 | 0 | -log(0.9) ≈ 0.105 |
| Wrong 0 | 0 | 0.9 | 0 | -log(0.1) ≈ 2.3 |

**Key insight:** The more confident (and correct) the prediction, the lower the cost.

### 3.3 Convexity Proof

The Hessian matrix of the cross-entropy loss is **positive definite**, guaranteeing a convex optimization landscape with a single global minimum.

---

## 4. Gradient Descent for Logistic Regression

### 4.1 Gradient Derivation

We need to find $\frac{\partial J}{\partial \theta_j}$ for each parameter.

Starting with the cost function:
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(\sigma(z^{(i)})) + (1-y^{(i)})\log(1-\sigma(z^{(i)}))]$$

Where $z^{(i)} = \theta^T x^{(i)}$.

**Using the chain rule and sigmoid derivative:**

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^{m}(\sigma(z^{(i)}) - y^{(i)})x_j^{(i)}$$

**Vectorized form:**

$$\nabla_\theta J = \frac{1}{m} X^T \cdot (\sigma(X\theta) - y)$$

### 4.2 Update Rule

For each iteration:
$$\theta_j := \theta_j - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}(\sigma(z^{(i)}) - y^{(i)})x_j^{(i)}$$

Where $\alpha$ is the learning rate.

### 4.3 Algorithm

```
Algorithm: Gradient Descent for Logistic Regression

Input: Training data (X, y), learning rate α, iterations N
Output: Learned parameters θ

1. Initialize θ = 0 (or random)
2. For iter = 1 to N:
   a. Compute predictions: h = σ(Xθ)
   b. Compute gradient: g = (1/m) * X^T * (h - y)
   c. Update parameters: θ = θ - α * g
3. Return θ
```

### 4.4 Convergence Visualization

```
Cost vs Iterations (Gradient Descent)

Cost│
    │    ╲
    │     ╲
    │      ╲__
    │         ╲___
    │              ╲___
    │                   ╲______
    │                         ╲______
    │                               ╲___
    │                                     ╲___
    │                                           ╲___
    │                                                 ╲___
  ──┼────────────────────────────────────────────────────────────────
    0       100      200      300      400      500      600    Iterations
                                ▲
                              Converged
```

---

## 5. Multi-class Classification

### 5.1 One-vs-Rest (OvR) Strategy

Train $K$ binary classifiers, where $K$ = number of classes.

```
For 3 classes (0, 1, 2):

Classifier 0: "Is class 0?" vs "Not class 0"
Classifier 1: "Is class 1?" vs "Not class 1"  
Classifier 2: "Is class 2?" vs "Not class 2"

Prediction: Choose class with highest probability
```

**Algorithm:**
1. For each class $k$, train a logistic regression classifier $h_k(x)$ to compute $P(y=k|x)$
2. For new input $x$, predict class with highest probability

### 5.2 Softmax Regression (Multinomial Logistic Regression)

For $K$ classes, use the softmax function:

$$P(y=k|x) = \frac{e^{\theta_k^T x}}{\sum_{j=1}^{K} e^{\theta_j^T x}}$$

**Cost function (cross-entropy):**

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} \mathbb{1}\{y^{(i)}=k\} \log\left(\frac{e^{\theta_k^T x^{(i)}}}{\sum_{j=1}^{K} e^{\theta_j^T x^{(i)}}}\right)$$

**Note:** Only one set of parameters needed (vs K classifiers in OvR).

---

## 6. Regularization in Logistic Regression

### 6.1 Why Regularize?

- **Prevent overfitting**: Especially with many features
- **Handle multicollinearity**: Correlated features
- **Improve generalization**: Better performance on test data

### 6.2 L2 Regularization (Ridge)

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2$$

**Note:** Typically don't regularize $\theta_0$ (intercept).

### 6.3 L1 Regularization (Lasso)

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{m}\sum_{j=1}^{n}|\theta_j|$$

**Effect:** Creates sparse solutions (some coefficients become exactly 0) - feature selection.

### 6.4 Elastic Net (L1 + L2)

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))] + \lambda\left(\frac{1-r}{2}\sum_{j=1}^{n}\theta_j^2 + r\sum_{j=1}^{n}|\theta_j|\right)$$

Where $r$ is the mix ratio (0 = pure L2, 1 = pure L1).

---

## 7. Implementation from Scratch

### 7.1 NumPy Implementation

```python
import numpy as np

class LogisticRegression:
    """
    Logistic Regression implementation from scratch.
    
    Supports:
    - Binary classification
    - Multi-class classification (One-vs-Rest)
    - L1, L2, and Elastic Net regularization
    - Gradient descent optimization
    """
    
    def __init__(self, 
                 learning_rate=0.01, 
                 n_iterations=1000,
                 regularization='l2',
                 lambda_param=0.1,
                 elastic_mix=0.5,
                 fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.elastic_mix = elastic_mix
        self.fit_intercept = fit_intercept
        
        self.theta = None
        self.classes_ = None
        
    def _sigmoid(self, z):
        """Numerically stable sigmoid function."""
        # For positive z: sigma(z) = 1 / (1 + exp(-z))
        # For negative z: sigma(z) = exp(z) / (1 + exp(z))
        return np.where(z >= 0,
                      1 / (1 + np.exp(-z)),
                      np.exp(z) / (1 + np.exp(z)))
    
    def _add_intercept(self, X):
        """Add bias term to features."""
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def _regularization_term(self):
        """Compute regularization penalty gradient."""
        if self.regularization == 'none':
            return 0
        
        # Don't regularize intercept
        theta_without_intercept = self.theta[1:] if self.fit_intercept else self.theta
        
        if self.regularization == 'l2':
            return (self.lambda_param / len(theta_without_intercept)) * theta_without_intercept
        elif self.regularization == 'l1':
            return (self.lambda_param / len(theta_without_intercept)) * np.sign(theta_without_intercept)
        elif self.regularization == 'elastic':
            l2_term = (self.lambda_param * (1 - self.elastic_mix) / len(theta_without_intercept)) * theta_without_intercept
            l1_term = (self.lambda_param * self.elastic_mix / len(theta_without_intercept)) * np.sign(theta_without_intercept)
            return l2_term + l1_term
        return 0
    
    def fit(self, X, y):
        """
        Fit the logistic regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        # Store classes
        self.classes_ = np.unique(y)
        
        # Add intercept if needed
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        # Initialize parameters
        n_features = X.shape[1]
        self.theta = np.zeros(n_features)
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Compute predictions
            z = X @ self.theta
            h = self._sigmoid(z)
            
            # Compute gradient
            gradient = (1 / len(y)) * X.T @ (h - y)
            
            # Apply regularization
            gradient[1:] += self._regularization_term()
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Returns
        -------
        probabilities : array of shape (n_samples, n_classes)
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        z = X @ self.theta
        h = self._sigmoid(z)
        
        # For binary classification, return [P(y=0), P(y=1)]
        if len(self.classes_) == 2:
            return np.column_stack([1 - h, h])
        else:
            # For multi-class, use softmax
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        threshold : float, default=0.5
            Threshold for binary classification
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
        """
        probas = self.predict_proba(X)
        
        if len(self.classes_) == 2:
            return np.where(probas[:, 1] >= threshold, self.classes_[1], self.classes_[0])
        else:
            return self.classes_[np.argmax(probas, axis=1)]
    
    def score(self, X, y):
        """Return mean accuracy."""
        return np.mean(self.predict(X) == y)
    
    @property
    def coef_(self):
        """Return coefficients (excluding intercept)."""
        if self.fit_intercept:
            return self.theta[1:]
        return self.theta
    
    @property
    def intercept_(self):
        """Return intercept."""
        if self.fit_intercept:
            return self.theta[0]
        return 0
```

---

## 8. Practical Considerations

### 8.1 Feature Scaling

Logistic regression benefits from feature normalization:
- Z-score normalization
- Min-max scaling

**Why?** Gradient descent converges faster with normalized features.

```
Before normalization:    After normalization:
                       
Cost                    Cost
 │                        │
 │ ╲    ╲                 │ ╲╱╲
 │  ╲    ╲                │  ╲
 │   ╲___╲___             │   ╲___
 │        ╲               │      ╲
 │         ╲              │       ╲
 └──────────── θ         ──┴───────── θ
  Slow convergence      Fast convergence
```

### 8.2 Converting Probabilities to Classes

Default threshold: 0.5

But you can adjust based on:
- **Class imbalance**: Lower threshold to catch more positives
- **Cost asymmetry**: Different costs for false positives vs false negatives

```
ROC Curve: Trade-off between TPR and FPR at different thresholds

True Positive Rate
    │
 1.0┤    ● ● ●
    │   ●     ●
    │  ●       ●
 0.5┤ ●         ●
    │           ●
    │            ●●
 0.0┤              ●●
    └──────────────────── False Positive Rate
       0        0.5       1.0
       
       ↑        ↑
    Thresh=0.8  Thresh=0.2
```

### 8.3 Handling Multi-class

```python
class MultiClassLogisticRegression:
    """One-vs-Rest approach for multi-class classification."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.classifiers_ = {}
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        for cls in self.classes_:
            # Create binary labels
            y_binary = (y == cls).astype(int)
            
            # Train classifier
            clf = LogisticRegression(**self.kwargs)
            clf.fit(X, y_binary)
            self.classifiers_[cls] = clf
        
        return self
    
    def predict_proba(self, X):
        probas = []
        for cls in self.classes_:
            probas.append(self.classifiers_[cls].predict_proba(X)[:, 1])
        return np.column_stack(probas)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
```

---

## 9. Evaluation Metrics

### 9.1 Confusion Matrix

```
                    Predicted
                  Neg    Pos
Actual  Neg     TN    FP
        Pos     FN    TP
```

- **Accuracy**: $(TP + TN) / (TP + TN + FP + FN)$
- **Precision**: $TP / (TP + FP)$ - "Of predicted positive, how many correct?"
- **Recall**: $TP / (TP + FN)$ - "Of actual positive, how many caught?"
- **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$

### 9.2 ROC Curve and AUC

- **ROC**: Plot TPR vs FPR at various thresholds
- **AUC**: Area Under the Curve (1.0 = perfect, 0.5 = random)

### 9.3 Log Loss (Cross-Entropy)

$$LogLoss = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(p^{(i)}) + (1-y^{(i)})\log(1-p^{(i)})]$$

Lower is better - calibrated probability estimates.

---

## 10. Summary

| Topic | Key Points |
|-------|------------|
| **Hypothesis** | $h_\theta(x) = \sigma(\theta^T x)$ |
| **Cost Function** | Cross-entropy: $J(\theta) = -log-likelihood$ |
| **Gradient** | $\nabla_\theta J = \frac{1}{m}X^T(\sigma(X\theta) - y)$ |
| **Decision Boundary** | Linear: $\theta^T x = 0$ |
| **Regularization** | L1 (sparse), L2 (stable), Elastic Net (both) |
| **Multi-class** | One-vs-Rest or Softmax |

**When to use Logistic Regression:**
- Binary classification
- Need probability estimates
- Need interpretable model
- Baseline model for comparison
- Linearly separable classes

**Limitations:**
- Assumes linear decision boundary
- Can underfit for complex patterns
- Sensitive to outliers