# Chapter: Regression Algorithms

> **Learning Duration:** 3 Days  
> **Difficulty:** Intermediate  
> **Prerequisites:** Linear algebra, calculus, linear regression basics

---

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Understand regression fundamentals
- Implement Polynomial Regression
- Apply Ridge and Lasso Regularization
- Build Gradient Boosting Regressors
- Evaluate regression models appropriately

---

## 1.1 Beyond Linear Regression

### Limitations of Linear Regression

Linear regression assumes a linear relationship:

$$y = w_1 x_1 + w_2 x_2 + ... + b$$

But real-world data often has:
- **Curved relationships**: Quadratic, cubic
- **Interactions**: $y = w_1 x_1 + w_2 x_2 + w_3 x_1 x_2$
- **Non-linear patterns**: Exponential, logarithmic

### Solutions

| Method | Description | Use Case |
|--------|-------------|----------|
| Polynomial Regression | Add powers of features | Curved relationships |
| Ridge Regression | L2 regularization | Prevent overfitting |
| Lasso Regression | L1 regularization | Feature selection |
| Gradient Boosting | Ensemble of weak learners | Complex patterns |

---

## 1.2 Polynomial Regression

### Concept

Add polynomial features to capture non-linear relationships:

$$y = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + ...$$

### Example

For degree-2 polynomial with one feature:
- Original: $x$ → $[x, x^2]$
- Features: $[x_1, x_1^2]$

For two features degree-2:
- Original: $[x_1, x_2]$ → $[x_1, x_2, x_1^2, x_2^2, x_1 x_2]$

### Implementation

```python
class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.model = LinearRegression()  # Use our earlier implementation
    
    def transform(self, X):
        """Create polynomial features"""
        X_poly = [X]
        for d in range(2, self.degree + 1):
            X_poly.append(X ** d)
        return np.hstack(X_poly)
    
    def fit(self, X, y):
        X_poly = self.transform(X)
        self.model.fit(X_poly, y)
    
    def predict(self, X):
        X_poly = self.transform(X)
        return self.model.predict(X_poly)
```

### Trade-offs

- **Degree too low**: Underfitting (can't capture pattern)
- **Degree too high**: Overfitting (fits noise, poor generalization)

```
Degree 1:  ─────────────
Degree 2:  ╭───────────
Degree 3:  ╭╮──────────
Degree 10: ╭╮╭╮╭╮╭╮─── (overfitting!)
```

### Choosing Degree

1. **Cross-validation**: Try different degrees, pick best CV score
2. **Learning curves**: Plot train vs validation error
3. **Domain knowledge**: Sometimes degree is known from physics

---

## 1.3 Regularization

### The Problem

With many features or high-degree polynomials, models can overfit:

```
Training Error:  Low
Validation Error: High
```

### Solution: Regularization

Add penalty term to loss function to constrain weights:

$$J(w) = Loss(w) + λ \cdot Penalty(w)$$

### Ridge Regression (L2)

$$J(w) = \sum(y_i - ŷ_i)^2 + λ \sum w_j^2$$

- Shrinks weights toward zero
- Keeps all features (doesn't eliminate)
- Good when all features might be relevant

### Lasso Regression (L1)

$$J(w) = \sum(y_i - ŷ_i)^2 + λ \sum |w_j|$$

- Can drive weights exactly to zero
- Performs feature selection
- Good when some features are irrelevant

### Visual Comparison

```
Ridge (L2):           Lasso (L1):
  w₁                    w₁
  │                      │
  │ ╭──╮                 │╭
  │╱   ╲                ╱│
  ├────────→ λ      ├────────→ λ
  │
  All reduced      Some = 0
```

### Implementation

```python
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        n_features = X.shape[1]
        
        # Add bias
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        
        # Ridge: (X'X + λI)^(-1) X'y
        XTX = X_with_bias.T @ X_with_bias
        regularization = self.alpha * np.eye(n_features + 1)
        regularization[0, 0] = 0  # Don't regularize bias
        
        XTX_reg = XTX + regularization
        XTX_inv = np.linalg.inv(XTX_reg + 1e-10 * np.eye(n_features + 1))
        
        self.weights = XTX_inv @ X_with_bias.T @ y
        
        self.bias = self.weights[0]
        self.coefs = self.weights[1:]
    
    def predict(self, X):
        return X @ self.coefs + self.bias
```

---

## 1.4 Gradient Boosting

### Concept

Build trees sequentially, each correcting the errors of previous trees.

$$F(x) = F_0(x) + η \cdot h_1(x) + η \cdot h_2(x) + ...$$

Where:
- $F_0(x)$: Initial prediction (mean)
- $h_t(x)$: Tree fitted to residuals
- $η$: Learning rate (shrinkage)

### Algorithm

```
1. Initialize: F₀(x) = mean(y)
2. For t = 1 to n_trees:
   a. Compute pseudo-residuals: r_i = y_i - F_{t-1}(x_i)
   b. Fit tree h_t to residuals
   c. Update: F_t(x) = F_{t-1}(x) + η * h_t(x)
3. Final: F(x) = Σ η * h_t(x)
```

### Why It Works

1. **Sequential correction**: Each tree focuses on remaining errors
2. **Gradient descent**: Equivalent to gradient descent in function space
3. **Shrinkage**: Small learning rate prevents overfitting
4. **Weak learners**: Simple trees (shallow) prevent overfitting

### Implementation

```python
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # Initialize with mean
        self.initial_prediction = np.mean(y)
        predictions = np.full(n_samples, self.initial_prediction)
        
        for _ in range(self.n_estimators):
            # Compute residuals
            residuals = y - predictions
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update predictions
            tree_preds = tree.predict(X)
            predictions += self.learning_rate * tree_preds
    
    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
```

### Hyperparameters

| Parameter | Effect |
|-----------|--------|
| n_estimators | More trees = more complexity |
| learning_rate | Smaller = more trees needed, better generalization |
| max_depth | Deeper = more complex trees |
| min_samples_leaf | Prevents overfitting to noise |

---

## 1.5 Evaluation Metrics

### Mean Squared Error (MSE)

$$MSE = \frac{1}{n} \sum(y_i - ŷ_i)^2$$

- Penalizes large errors more
- Same units as target squared
- Sensitive to outliers

### Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{MSE}$$

- Same units as target
- More interpretable than MSE
- Preferred in practice

### Mean Absolute Error (MAE)

$$MAE = \frac{1}{n} \sum|y_i - ŷ_i|$$

- Robust to outliers
- Less sensitive to large errors
- Useful when outliers are noise

### R² Score (Coefficient of Determination)

$$R² = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y - ŷ)^2}{\sum(y - ȳ)^2}$$

- 1.0 = Perfect
- 0.0 = Always predicts mean
- Negative = Worse than mean

### Comparison

| Metric | Outlier Sensitive | Interpretable | Scale-Dependent |
|--------|-------------------|---------------|------------------|
| MSE | Yes | ✓ | Yes |
| RMSE | Yes | ✓ | Yes |
| MAE | No | ✓ | Yes |
| R² | - | - | No |

---

## 📝 Summary

### Key Takeaways

1. **Polynomial Regression**: Capture non-linear patterns
2. **Ridge (L2)**: Shrink weights, keep all features
3. **Lasso (L1)**: Feature selection, sparse solutions
4. **Gradient Boosting**: Sequential ensemble, powerful

### When to Use

| Scenario | Method |
|----------|--------|
| Simple curved relationship | Polynomial |
| Many features, potential overfitting | Ridge |
| Suspect irrelevant features | Lasso |
| Complex patterns, tabular data | Gradient Boosting |

### Formulas

| Concept | Formula |
|---------|---------|
| MSE | $\frac{1}{n} \sum(y - ŷ)^2$ |
| RMSE | $\sqrt{MSE}$ |
| MAE | $\frac{1}{n} \sum|y - ŷ|$ |
| Ridge Penalty | $λ \sum w_j^2$ |
| Lasso Penalty | $λ \sum |w_j|$ |
| R² | $1 - SS_{res}/SS_{tot}$ |

---

## ❓ Quick Check

1. What happens when polynomial degree is too high?
2. How does Ridge differ from Lasso in terms of feature selection?
3. Why is learning rate important in Gradient Boosting?
4. Which metric is most robust to outliers?

*Answers in solutions*