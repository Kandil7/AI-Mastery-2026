# Chapter: Classification Algorithms

> **Learning Duration:** 4 Days  
> **Difficulty:** Intermediate  
> **Prerequisites:** Linear regression, probability theory

---

## 🎯 Learning Objectives

By the end of this chapter, you will:
- Understand the fundamentals of classification
- Implement Logistic Regression from scratch
- Build Decision Trees for classification
- Create ensemble methods (Random Forest)
- Evaluate classifiers using appropriate metrics

---

## 1.1 What is Classification?

### Definition

**Classification** is a supervised learning task where we predict categorical labels for input data. Unlike regression (predicting continuous values), classification predicts discrete classes.

### Types of Classification

| Type | Description | Examples |
|------|-------------|----------|
| **Binary** | Two classes | Spam detection, Disease diagnosis |
| **Multi-class** | More than two classes | Image recognition, Letter recognition |
| **Multi-label** | Multiple labels per sample | Tagging, Content categorization |

### Key Terminology

- **Training Data**: Labeled examples (X, y)
- **Model**: Function that maps inputs to predictions
- **Training**: Finding parameters that minimize error
- **Inference**: Making predictions on new data
- **Generalization**: Performance on unseen data

---

## 1.2 Logistic Regression

### The Logistic Function

Logistic Regression uses the **sigmoid function** to squash predictions between 0 and 1:

$$σ(z) = \frac{1}{1 + e^{-z}}$$

Where:
- $z = w_1x_1 + w_2x_2 + ... + b$ (linear combination)
- $σ(z)$ outputs probability between 0 and 1

### Why Sigmoid?

```
         σ(z)
          │
    1.0   │    ╭────────────
          │   ╱
    0.5 ──┼──╯
          │ ╱
    0.0 ──┼─╯────────────────
          └──────────────→ z
             -5    0    5
```

Properties:
- S-shaped curve (Sigmoid)
- Output bounded [0, 1]
- Derivative: $σ'(z) = σ(z)(1 - σ(z))$
- Easy to differentiate for gradient descent

### Mathematical Formulation

**Hypothesis:**
$$h_θ(x) = σ(θ^T x) = \frac{1}{1 + e^{-(w^T x + b)}}$$

**Prediction:**
- If $h_θ(x) ≥ 0.5$: predict class 1
- If $h_θ(x) < 0.5$: predict class 0

**Decision Boundary:**
- The surface where $w^T x + b = 0$
- Can be linear or polynomial

### Cost Function

We use **Cross-Entropy Loss** (also called Log Loss):

$$J(θ) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(h_θ(x^{(i)})) + (1-y^{(i)})\log(1-h_θ(x^{(i)}))]$$

Why cross-entropy?
- Convex (has single global minimum)
- Large penalty for confident wrong predictions
- Derived from maximum likelihood estimation

### Gradient Descent Update

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (h_θ(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

Update rule:
$$w_j := w_j - α \cdot \frac{\partial J}{\partial w_j}$$

### Implementation

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        # Numerical stability
        return np.where(z >= 0,
                       1/(1 + np.exp(-z)),
                       np.exp(z)/(1 + np.exp(z)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_epochs):
            # Forward pass
            linear = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear)
            
            # Compute gradients
            dw = np.dot(X.T, (predictions - y)) / n_samples
            db = np.mean(predictions - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear)
        return (probabilities >= 0.5).astype(int)
```

---

## 1.3 Decision Trees

### Concept

Decision Trees split data based on feature values to create homogeneous groups.

```
        Is Age > 30?
           /    \
         Yes     No
         ↓       ↓
    Is Income    Is Income  
    > 50k?       > 50k?
     /  \          /  \
   Yes   No     Yes   No
   ↓     ↓       ↓     ↓
 Buy  Don't   Buy  Don't
```

### Key Terminology

- **Root**: First split
- **Node**: Decision point
- **Leaf**: Final prediction
- **Branch**: Path from root to leaf
- **Depth**: Number of splits

### Splitting Criteria

We need to measure "purity" of subsets. Two common metrics:

**1. Gini Impurity:**
$$Gini = 1 - \sum_{k=1}^{K} p_k^2$$

Where $p_k$ is proportion of class k in the node.

- 0 = Pure (all same class)
- 0.5 = Maximum impurity (equal mix)

**2. Information Gain (Entropy):**
$$Entropy = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

$$IG = Entropy_{parent} - \sum_{child} \frac{N_{child}}{N_{parent}} \cdot Entropy_{child}$$

### Algorithm

```
function build_tree(data, depth):
    if depth > max_depth or data is pure:
        return Leaf(majority_class)
    
    best_feature, best_threshold = find_best_split(data)
    
    left = split(data, best_feature, best_threshold)
    right = split(data, best_feature, best_threshold, invert=True)
    
    left_child = build_tree(left, depth + 1)
    right_child = build_tree(right, depth + 1)
    
    return Node(best_feature, best_threshold, left_child, right_child)
```

### Implementation

```python
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def find_best_split(self, X, y):
        best_gain = 0
        best_feature = 0
        best_threshold = 0
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if len(y[left_mask]) < 1 or len(y[right_mask]) < 1:
                    continue
                
                # Calculate information gain
                parent_gini = self.gini_impurity(y)
                n = len(y)
                n_left, n_right = len(y[left_mask]), len(y[right_mask])
                
                child_gini = (n_left/n * self.gini_impurity(y[left_mask]) +
                            n_right/n * self.gini_impurity(y[right_mask]))
                
                gain = parent_gini - child_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        # Stopping conditions
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {'class': np.bincount(y).argmax()}
        
        feature, threshold, gain = self.find_best_split(X, y)
        
        if gain == 0:
            return {'class': np.bincount(y).argmax()}
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth+1)
        
        return {'feature': feature, 'threshold': threshold,
                'left': left_tree, 'right': right_tree}
    
    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])
    
    def _predict(self, x, node):
        if 'class' in node:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict(x, node['left'])
        return self._predict(x, node['right'])
```

### Advantages & Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| Easy to interpret | Prone to overfitting |
| No feature scaling needed | Biased toward features with many levels |
| Handles missing values | Small changes in data can change tree |
| Non-linear relationships | Greedy (may miss global optimum) |

---

## 1.4 Random Forest

### Ensemble Learning

**Ensemble methods** combine multiple models to improve performance.

```
Single Model:      Ensemble:
    ╱╲                  ╱╲
   ╱  ╲                ╱  ╲ ← Vote/Average
  ╱    ╲      →       ╱    ╲
 ╱      ╲            ╱      ╲
╱        ╲          ╱        ╲
```

### Bagging (Bootstrap Aggregating)

1. **Bootstrap**: Sample with replacement
2. **Aggregate**: Combine predictions

Reduces variance without increasing bias.

### Random Forest Algorithm

1. Create $N$ bootstrap samples
2. Train $N$ decision trees (with random feature selection)
3. For prediction: majority vote (classification) or average (regression)

**Key difference from single tree**:
- At each node, only $k$ random features are considered (typically $k = √d$)
- This adds diversity and reduces correlation

### Implementation

```python
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=10, n_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train tree
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
    
    def predict(self, X):
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 0, predictions
        )
```

### Why Random Forest Works

1. **Variance Reduction**: Averaging reduces variance
2. **Decorrelation**: Random feature selection makes trees different
3. **Robustness**: Less sensitive to outliers than single tree

---

## 1.5 Evaluation Metrics

### Confusion Matrix

```
                 Predicted
              Pos     Neg
Actual  Pos   TP      FN
        Neg   FP      TN
```

- **True Positive (TP)**: Correctly predicted positive
- **False Negative (FN)**: Missed positive
- **False Positive (FP)**: Incorrectly predicted positive
- **True Negative (TN)**: Correctly predicted negative

### Accuracy

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

Simple but can be misleading for imbalanced data.

### Precision

$$Precision = \frac{TP}{TP + FP}$$

Of all positive predictions, how many are correct?

### Recall (Sensitivity)

$$Recall = \frac{TP}{TP + FN}$$

Of all actual positives, how many did we find?

### F1-Score

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

Harmonic mean of precision and recall. Good for imbalanced data.

### ROC-AUC

- **ROC Curve**: Plots TPR vs FPR at different thresholds
- **AUC**: Area Under the Curve
- 1.0 = Perfect, 0.5 = Random

---

## 📝 Summary

### Key Takeaways

1. **Logistic Regression**: Linear decision boundary, probabilistic outputs
2. **Decision Trees**: Interpretable, prone to overfitting
3. **Random Forest**: Ensemble of trees, more robust, less overfitting
4. **Evaluation**: Use multiple metrics, not just accuracy

### Formulas to Remember

| Concept | Formula |
|---------|---------|
| Sigmoid | $σ(z) = 1/(1 + e^{-z})$ |
| Cross-Entropy | $J = -[y\log(ŷ) + (1-y)\log(1-ŷ)]$ |
| Gini | $1 - \sum p_k^2$ |
| Information Gain | $Entropy_{parent} - \sum Entropy_{child}$ |
| F1 Score | $2PR/(P + R)$ |

---

## 🔄 What's Next

- **Regression**: Predicting continuous values
- **Clustering**: Unsupervised grouping
- **Dimensionality Reduction**: Compressing features

---

## ❓ Quick Check

1. Why is cross-entropy better than MSE for classification?
2. What is the difference between Gini and Entropy?
3. Why do we use random feature selection in Random Forest?
4. When would you prefer Logistic Regression over Random Forest?

*Answers in solutions*