# Regularization: Preventing Overfitting in Machine Learning

## Introduction

**Overfitting** occurs when a model learns not only the underlying patterns in training data but also the noise, leading to poor generalization on new data. **Regularization** is the technique of adding constraints to prevent this.

### Visual Intuition

```
                    Underfitting          Good Fit           Overfitting
                        │                    │                    │
    Function           │                    │                    │
        │              │    ╱╲              │    ╱╲              │   ╱╲╱╲
        │              │   ╱  ╲            │   ╱  ╲            │  ╱ ╲ ╱╲
        │              │  ╱    ╲           │  ╱    ╲           │ ╱   ╲   ╲
        │              │ ╱      ╲          │ ╱      ╲          │╱     ╲   ╲
        │             ╱╱        ╲╲       ╱╱        ╲╲        │       ╲   ╲
        │            ╱           ╲      ╱           ╲       ╱         ╲  ╲
 ───────┼────────────────────────────┼────────────────────────┼──────────┼──
        │                            │                    x   │          │
        │                            │                        │          │
   High Bias                 Balanced              High Variance
   (Too Simple)             Complexity             (Too Complex)
```

---

## 1. The Bias-Variance Tradeoff

### 1.1 Decomposition of Error

For any model, the expected prediction error can be decomposed:

$$E[(y - \hat{f}(x))^2] = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \text{Irreducible Error}$$

Where:
- **Bias**: Error from overly simplistic assumptions (underfitting)
- **Variance**: Error from too much sensitivity to training data (overfitting)
- **Irreducible Error**: Noise in the data itself

### 1.2 Visualization

```
                    Total Error
                      │
      Error          │        ╱╲
                      │       ╱  ╲
                      │      ╱    ╲
                      │     ╱      ╲
                      │    ╱        ╲
                      │   ╱          ╲
                      │  ╱            ╲
                      │ ╱              ╲
        ─────────────┼────────────────────────
                  Model Complexity
                   (Regularization →)
                      
              │←  Underfit  →│←  Overfit  →│
              
     Bias² + Variance Curve
     ╱╲
    ╱  ╲
   ╱    ╲ ← Bias²
  ╱      ╲
 ─┼─────────
  ╲      ╱ ← Variance
   ╲    ╱
    ╲  ╱
     ╲╱
     
     Optimal →←
```

### 1.3 Regularization's Role

Regularization reduces variance by:
1. Shrinking coefficients (L2)
2. Setting some coefficients to zero (L1)
3. Limiting model complexity
4. Adding noise to training (dropout, data augmentation)

---

## 2. L2 Regularization (Ridge Regression)

### 2.1 Definition

L2 adds the sum of squared coefficients to the cost function:

$$J_{L2}(\theta) = J_{original}(\theta) + \lambda \sum_{j=1}^{n} \theta_j^2$$

Where:
- $\lambda$ is the regularization strength (hyperparameter)
- $\theta_j$ are model parameters (not including bias)
- The sum excludes the bias term typically

### 2.2 Effect on Coefficients

```
Coefficient Magnitude vs λ (Ridge)

|Coeff│
 │    │
 │    ╲── ╲── ╲── ╲
 │      ╲    ╲    ╲
 │        ╲    ╲    ╲
 │          ╲    ╲    ╲
 │            ╲    ╲    ╲
 │              ╲    ╲    ╲
 ──────────────── λ ──────────
  0        0.1   1    10   100

  Coefficients shrink but never reach exactly 0
```

**Key Property:** All coefficients shrink toward but never reach zero.

### 2.3 Mathematical Properties

**For Linear Regression with L2 (Ridge):**

Closed-form solution:
$$\hat{\theta}_{ridge} = (X^T X + \lambda I)^{-1} X^T y$$

Where $I$ is the identity matrix (with 0 in the bias position).

**Why does this work?**
- Adding $\lambda I$ makes the matrix invertible even when $X^T X$ is singular
- This is especially useful when features are correlated (multicollinearity)

### 2.4 Implementation

```python
class RidgeRegression:
    """Ridge Regression (L2 Regularized Linear Regression)."""
    
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha  # Regularization strength (λ)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            # Add bias column
            X_with_bias = np.column_stack([np.ones(n_samples), X])
            n_params = n_features + 1
        else:
            X_with_bias = X
            n_params = n_features
        
        # Create regularization matrix (0 for bias term)
        reg_matrix = np.eye(n_params)
        if self.fit_intercept:
            reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        # Solve: (X'X + λI)θ = X'y
        XtX = X_with_bias.T @ X_with_bias
        Xty = X_with_bias.T @ y
        theta = np.linalg.solve(XtX + self.alpha * reg_matrix, Xty)
        
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0
            self.coef_ = theta
        
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
```

### 2.5 Choosing λ

**Cross-validation** is the standard approach:

```
k-Fold Cross-Validation

Fold 1: [●●●●|○○○○] → Train on ●, validate on ○
Fold 2: [○○●●|●●○○] → Train on ●, validate on ○
Fold 3: [○○○●|○○●○] → Train on ●, validate on ○
Fold 4: [○○○○|●●●●] → Train on ●, validate on ○

Average performance across folds for each λ
```

**Typical search grid:** [0.01, 0.1, 1, 10, 100, 1000]

---

## 3. L1 Regularization (Lasso)

### 3.1 Definition

L1 adds the sum of absolute values of coefficients:

$$J_{L1}(\theta) = J_{original}(\theta) + \lambda \sum_{j=1}^{n} |\theta_j|$$

### 3.2 Sparse Solutions

Unlike L2, L1 can set coefficients exactly to zero:

```
Lasso path (coefficients vs λ)

Coefficient Value
  │  β₁ ╱╲
  │    ╱  ╲
  │   ╱    ╲ ──── β₂
  │  ╱      ╲
  │ ╱        ╲ ──────────── β₃
  │╱          ╲
  ├─────────────────────────────── λ
  0         0.5        1.0
        
  At λ > λ₁: β₁ = 0 (feature eliminated)
  At λ > λ₂: β₂ = 0 (feature eliminated)
  At λ > λ₃: β₃ still non-zero
  
  → Feature selection built into the model!
```

### 3.3 Geometric Interpretation

**L2 (Ridge):** Constrains to an n-dimensional **sphere** (L2 norm = constant)
```
L2 constraint: θ₁² + θ₂² ≤ c

          │╲
          │ ╲
          │  ╲
          │   ╲
          │    ╲
──────────┴─────╲────────
          │      ╲
          │       ╲
          │        ╲
          
  Solution touches sphere at single point
  (No corner = no zero coefficients, except at origin)
```

**L1 (Lasso):** Constrains to an n-dimensional **cross-polytope** (L1 norm = constant)

```
L1 constraint: |θ₁| + |θ₂| ≤ c

        │╱╲
       ╱│ ╲
      ╱ │  ╲
     ╱  │   ╲
    ╱   │    ╲
───┼──────────┼────
    ╲   │    ╱
     ╲  │   ╱
      ╲ │  ╱
       ╲│ ╱
        ╲╱

  Solution touches corners → coefficients can be exactly zero!
```

### 3.4 Implementation

```python
class LassoRegression:
    """Lasso Regression (L1 Regularized Linear Regression)."""
    
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Standardize features (important for L1!)
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std[self._std == 0] = 1  # Prevent division by zero
        X_scaled = (X - self._mean) / self._std
        
        if self.fit_intercept:
            y_mean = np.mean(y)
            y_scaled = y - y_mean
        else:
            y_mean = 0
            y_scaled = y
        
        # Initialize coefficients
        theta = np.zeros(n_features)
        
        # Coordinate descent
        for _ in range(self.max_iter):
            theta_old = theta.copy()
            
            for j in range(n_features):
                # Compute residual excluding feature j
                residual = y_scaled - X_scaled @ theta + X_scaled[:, j] * theta[j]
                
                # Compute correlation with feature j
                rho_j = X_scaled[:, j] @ residual
                
                # Soft thresholding
                theta[j] = self._soft_threshold(rho_j / n_samples, self.alpha) / \
                           (X_scaled[:, j] @ X_scaled[:, j] / n_samples)
            
            # Check convergence
            if np.max(np.abs(theta - theta_old)) < self.tol:
                break
        
        # Store coefficients (in original scale)
        self.coef_ = theta / self._std
        self.intercept_ = y_mean - self._mean @ self.coef_
        
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
```

---

## 4. Elastic Net: Combining L1 and L2

### 4.1 Definition

Elastic Net combines L1 and L2 regularization:

$$J_{ElasticNet}(\theta) = J_{original}(\theta) + \lambda \left( \rho \sum_{j=1}^{n} |\theta_j| + \frac{1-\rho}{2} \sum_{j=1}^{n} \theta_j^2 \right)$$

Where:
- $\lambda$ is the overall regularization strength
- $\rho \in [0, 1]$ is the mix ratio (0 = pure L2, 1 = pure L1)

### 4.2 When to Use Elastic Net

| Scenario | Best Choice |
|----------|-------------|
| Many correlated features | Elastic Net |
| Want feature selection but stable | Elastic Net |
| Lasso selects too many features | Increase L2 proportion |
| Ridge underperforms | Increase L1 proportion |

### 4.3 Implementation

```python
class ElasticNet:
    """Elastic Net: L1 + L2 Regularization."""
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio  # ρ - mix ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Standardize
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std[self._std == 0] = 1
        X_scaled = (X - self._mean) / self._std
        y_mean = np.mean(y)
        y_scaled = y - y_mean
        
        theta = np.zeros(n_features)
        
        for _ in range(self.max_iter):
            theta_old = theta.copy()
            
            for j in range(n_features):
                residual = y_scaled - X_scaled @ theta + X_scaled[:, j] * theta[j]
                rho_j = X_scaled[:, j] @ residual
                
                # Elastic net: combination of L1 and L2
                # Soft threshold for L1 + scaling for L2
                alpha_l1 = self.alpha * self.l1_ratio
                alpha_l2 = self.alpha * (1 - self.l1_ratio)
                
                denom = (X_scaled[:, j] @ X_scaled[:, j]) / n_samples + alpha_l2
                theta[j] = np.sign(rho_j / n_samples) * \
                          max(abs(rho_j / n_samples) - alpha_l1, 0) / denom
            
            if np.max(np.abs(theta - theta_old)) < self.tol:
                break
        
        # Transform back
        self.coef_ = theta / self._std
        self.intercept_ = y_mean - self._mean @ self.coef_
        
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
```

---

## 5. Regularization in Other Models

### 5.1 Logistic Regression

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))] + \lambda R(\theta)$$

Where $R(\theta)$ can be L1, L2, or Elastic Net.

### 5.2 Neural Networks

**Weight Decay:** Similar to L2 but applied after each gradient step:
$$W^{(l)} \leftarrow (1 - \eta\lambda)W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$$

**Dropout:** Randomly deactivate neurons during training:
```python
# Training (inverted dropout)
mask = (np.random.rand(* activations.shape) > dropout_rate) / (1 - dropout_rate)
activations = activations * mask
```

### 5.3 Decision Trees

- **Max depth**: Limit tree depth
- **Min samples split**: Require minimum samples to split
- **Min samples leaf**: Require minimum samples in leaves
- **Max features**: Limit features considered for split

### 5.4 Support Vector Machines

The C parameter controls regularization:
- **Large C**: Hard margin (less regularization, more overfitting risk)
- **Small C**: Soft margin (more regularization, better generalization)

---

## 6. Practical Guidelines

### 6.1 When to Use Each Type

| Situation | Recommendation |
|-----------|----------------|
| Many features, some irrelevant | Lasso or Elastic Net |
| Many correlated features | Ridge or Elastic Net |
| Want interpretable features | Lasso |
| More features than samples | Elastic Net |
| Default start | Ridge |

### 6.2 Feature Preprocessing

**Critical:** Standardize/normalize features before regularization!

```
Why? Different scales → Unequal regularization effect

Example:
- Feature A: range [0, 1] (small coefficient possible)
- Feature B: range [0, 1000] (large coefficient possible)

With L2: Both get same λ
→ Feature B penalized more heavily (unfair!)

Solution: Standardize first
→ Both features have same scale → Equal treatment
```

### 6.3 Hyperparameter Tuning

**Grid Search vs Random Search:**

```
Grid Search:                        Random Search:
λ ∈ [0.01, 0.1, 1, 10]              λ ~ Uniform(0.01, 10)
                                     
Efficient for small grids          More efficient for large spaces
Exhaustive but slow                Can find good params faster
```

**Search Space Example:**
```python
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # For Elastic Net only
}
```

---

## 7. Summary Table

| Regularization | Effect on Coefficients | Use Case |
|----------------|------------------------|----------|
| **L2 (Ridge)** | Shrink toward zero, never reach | Correlated features, default |
| **L1 (Lasso)** | Can become exactly zero | Feature selection, sparse models |
| **Elastic Net** | Combination of both | Best of both worlds |
| **Dropout** | Randomly zero weights | Deep neural networks |
| **Early Stopping** | Stop training early | Iterative models |

**Key Insight:** Regularization trades off between:
- Fitting the training data (unregularized)
- Keeping parameters small (regularized)

The optimal balance is found via cross-validation.