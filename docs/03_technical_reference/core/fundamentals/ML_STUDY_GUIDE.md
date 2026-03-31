# Machine Learning Study Guide

A comprehensive guide covering mathematical foundations, key algorithms, and real-world applications.

---

## Part 1: Quick Reference Quiz

### Quiz Questions & Answers

| Q# | Question | Answer Summary |
|----|----------|----------------|
| 1 | Purpose of Supply Chain Management? | End-to-end flow from suppliers to consumers |
| 2 | Amazon's OR techniques? | Linear programming, inventory control, transportation modeling |
| 3 | PCA perspectives? | Maximum variance + minimum reconstruction error |
| 4 | Gradient Descent update rule? | `x_new = x_old - γ * ∇f(x_old)` |
| 5 | Naive Bayes assumption? | Conditional independence of features |
| 6 | EM algorithm steps? | E-step (responsibilities) → M-step (re-estimate params) |
| 7 | Kernel trick purpose? | Implicit high-dimensional inner products |

---

## Part 2: Mathematical Foundations

### 2.1 Linear Algebra for ML

#### Eigenvalues & Eigenvectors
For matrix A, eigenvector **x** satisfies: **Ax = λx**

```
A = [[2, 1],     Eigenvalues: λ₁=3, λ₂=1
     [1, 2]]     Eigenvectors: v₁=[1,1], v₂=[1,-1]
```

**Real-world**: Netflix uses eigendecomposition in SVD for recommendations.

#### Singular Value Decomposition (SVD)
Any matrix A can be decomposed: **A = UΣVᵀ**

- **U**: Left singular vectors (user features)
- **Σ**: Singular values (importance)
- **Vᵀ**: Right singular vectors (movie features)

```
User-Movie Matrix → SVD → Latent Factors → Predictions
     (sparse)              (genre, mood)    (fill gaps)
```

---

### 2.2 Principal Component Analysis (PCA)

**Goal**: Find directions of maximum variance

**Two perspectives**:
1. **Maximum Variance**: Find projection that preserves most variance
2. **Minimum Error**: Find projection that minimizes reconstruction error

**Math**:
- Compute covariance: `C = (1/n) * XᵀX`
- Eigendecompose: `C = VΛVᵀ`
- Principal components: columns of V (sorted by eigenvalue)

**Dimensionality Reduction**:
```
Original: X (n × d) 
Projected: Z = X * V_k (n × k)  where k << d
```

---

### 2.3 Gradient Descent

**Update Rule**:
```
θ := θ - α * ∇L(θ)
```

Where:
- θ = parameters
- α = learning rate
- ∇L = gradient of loss

**Variants**:
| Variant | Batch Size | Trade-off |
|---------|-----------|-----------|
| Batch GD | All data | Stable, slow |
| SGD | 1 sample | Noisy, fast |
| Mini-batch | 16-256 | Balanced |

**Convergence**:
```
Learning rate too high → Diverge
Learning rate too low  → Very slow convergence
```

---

### 2.4 Naive Bayes Classifier

**Bayes' Theorem**:
```
P(class|features) = P(features|class) * P(class) / P(features)
```

**Naive Assumption**: Features are conditionally independent
```
P(x₁,x₂,...,xₙ|class) = ∏ P(xᵢ|class)
```

**Spam Classification (Google)**:
```
P(spam|email) ∝ P(email|spam) * P(spam)

P(email|spam) = P("free"|spam) * P("click"|spam) * ...
```

---

### 2.5 Expectation-Maximization (EM)

**Purpose**: Estimate parameters when latent variables exist

**E-Step** (Expectation):
```
Compute responsibilities:
r_nk = P(z_k | x_n) = probability point n came from cluster k
```

**M-Step** (Maximization):
```
Re-estimate parameters:
μ_k = Σ(r_nk * x_n) / Σ(r_nk)
```

**GMM Application**:
1. Initialize K Gaussian components
2. E-step: Assign soft cluster memberships
3. M-step: Update means, covariances, weights
4. Repeat until convergence

---

### 2.6 Support Vector Machines

**Hard Margin SVM**:
```
max margin = 2/||w||
subject to: y_i(w·x_i + b) ≥ 1
```

**Kernel Trick**:
Replace `x·z` with `k(x,z)` to handle non-linear boundaries

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | xᵀz | Linearly separable |
| RBF | exp(-γ||x-z||²) | Non-linear, local |
| Polynomial | (xᵀz + c)ᵈ | Non-linear, global |

---

## Part 3: Real-World Applications

### 3.1 Amazon: Supply Chain Optimization

**Techniques Used**:
- Linear Programming for inventory allocation
- MILP for delivery scheduling
- DeepAR for demand forecasting

**Impact**:
```
Before OR: Manual planning, high waste
After OR:  Automated optimization
           - Reduced costs
           - Better inventory levels
           - Faster delivery
```

---

### 3.2 Netflix: Movie Recommendations

**SVD-based Collaborative Filtering**:
```
Rating Matrix R (users × movies)
     ↓ SVD
R ≈ U * Σ * Vᵀ

User u rating for movie m:
r_um ≈ Σ_k (U_uk * σ_k * V_mk)
```

**Latent Factors**: Genre preferences, mood, era, etc.

---

### 3.3 Google: Spam Filtering

**Naive Bayes for Email Classification**:
```
For each email:
  1. Tokenize words
  2. Calculate P(spam|words) vs P(ham|words)
  3. Classify based on higher probability
```

**Training**: Learn P(word|spam) from labeled examples

---

## Part 4: Essay Topics Summary

| Topic | Key Points |
|-------|------------|
| Math in ML | Linear algebra → PCA, SVD; Calculus → GD; Probability → Bayes |
| MLE vs MAP | MLE maximizes likelihood; MAP adds prior = regularization |
| SVD/PCA relationship | PCA = SVD on centered data |
| Supply Chain Evolution | Manual → OR/Analytics → AI/Robotics |
| CNN Architecture | Conv → Pool → FC; Activation for non-linearity |

---

## Part 5: Key Terminology Glossary

| Term | Definition |
|------|------------|
| **Backpropagation** | Chain rule applied layer-by-layer to compute gradients |
| **Conjugate Prior** | Prior that results in same-family posterior |
| **Eigenvalue** | Scalar λ where Ax = λx |
| **EM Algorithm** | E-step + M-step for latent variable models |
| **Gradient** | Vector of partial derivatives pointing uphill |
| **Kernel Trick** | Implicit high-dimensional inner products |
| **Latent Variable** | Unobserved variable inferred from data |
| **MAP Estimation** | MLE + prior = regularized estimation |
| **Overfitting** | Model captures noise, not signal |
| **PCA** | Find orthogonal directions of max variance |
| **Regularization** | Penalty to prevent overfitting |
| **SVD** | Matrix = UΣVᵀ decomposition |
| **SVM Margin** | Distance from hyperplane to nearest points |

---

## Quick Reference Formulas

```
PCA:         C = XᵀX/n, eigendecompose C
SVD:         A = UΣVᵀ  
Gradient:    θ = θ - α∇L(θ)
Bayes:       P(θ|x) ∝ P(x|θ)P(θ)
GMM E-step:  r_nk = π_k N(x_n|μ_k,Σ_k) / Σ_j π_j N(x_n|μ_j,Σ_j)
RBF Kernel:  k(x,z) = exp(-γ||x-z||²)
Hinge Loss:  max(0, 1 - y*f(x))
```
