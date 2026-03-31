# Machine Learning Glossary

Comprehensive terminology reference with mathematical definitions.

---

## A

### Activation Function
Function introducing non-linearity in neural networks.

| Function | Formula | Range |
|----------|---------|-------|
| ReLU | max(0, x) | [0, ∞) |
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) |

### Affine Space
Vector space offset from origin: `{v + b : v ∈ V}`

---

## B

### Backpropagation
Chain rule applied layer-by-layer to compute gradients.

```
∂L/∂w = ∂L/∂y · ∂y/∂z · ∂z/∂w
```

### Bayes' Theorem
```
P(θ|x) = P(x|θ) · P(θ) / P(x)

posterior = likelihood × prior / evidence
```

---

## C

### Conjugate Prior
Prior that produces same-family posterior.

| Likelihood | Conjugate Prior |
|------------|-----------------|
| Gaussian | Gaussian |
| Bernoulli | Beta |
| Poisson | Gamma |

### Covariance Matrix
```
C_ij = E[(X_i - μ_i)(X_j - μ_j)]
```

---

## D

### Determinant
Scalar encoding matrix properties: `det(A)` or `|A|`

- det(A) = 0 → A is singular
- det(AB) = det(A) · det(B)

---

## E

### Eigenvalue/Eigenvector
For matrix A: `Av = λv`
- v: eigenvector (direction preserved)
- λ: eigenvalue (scaling factor)

### EM Algorithm
```
E-step: r_nk = P(z_k | x_n)
M-step: θ = argmax Σ r_nk log P(x_n, z_k | θ)
```

---

## G

### Gradient
Vector of partial derivatives:
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```
Points toward steepest ascent.

### Gradient Descent
```
θ := θ - α · ∇L(θ)
```

---

## H

### Hinge Loss
SVM loss: `max(0, 1 - y·f(x))`

---

## K

### Kernel Trick
Implicit high-dimensional inner product:
```
k(x,z) = φ(x)ᵀφ(z)
```

| Kernel | Formula |
|--------|---------|
| Linear | xᵀz |
| RBF | exp(-γ‖x-z‖²) |
| Polynomial | (xᵀz + c)ᵈ |

---

## L

### Latent Variable
Unobserved variable inferred from data.

### Likelihood
```
L(θ|x) = P(x|θ)
```

---

## M

### MAP Estimation
```
θ_MAP = argmax P(θ|x) = argmax P(x|θ)P(θ)
```
MLE + prior = regularization

### MLE
```
θ_MLE = argmax P(x|θ)
```

---

## O

### Overfitting
Model captures noise instead of signal.

**Symptoms**: High training accuracy, low test accuracy

**Solutions**: Regularization, dropout, cross-validation

---

## P

### PCA
Find orthogonal directions of maximum variance.

```
1. Center: X_c = X - μ
2. Covariance: C = XᵀX/n
3. Eigendecompose: C = VΛVᵀ
4. Project: Z = XV_k
```

---

## R

### Regularization
Penalty to prevent overfitting.

| Type | Penalty | Effect |
|------|---------|--------|
| L1 (Lasso) | λΣ\|w\| | Sparsity |
| L2 (Ridge) | λΣw² | Small weights |

---

## S

### SVD
```
A = UΣVᵀ

U: left singular vectors (m×m)
Σ: singular values (diagonal)
V: right singular vectors (n×n)
```

### SVM Margin
```
margin = 2/‖w‖
```
Maximize margin subject to: `y_i(w·x_i + b) ≥ 1`

---

## Supply Chain Terms

### MILP
Mixed Integer Linear Programming - optimization with discrete choices.

### Operations Research
Analytical methods for better decisions: linear programming, inventory control.

### DeepAR
Amazon's probabilistic demand forecasting algorithm.

---

## Quick Formulas

```
PCA:      Z = XV_k where C = VΛVᵀ
SVD:      A = UΣVᵀ
GD:       θ = θ - α∇L
Bayes:    P(θ|x) ∝ P(x|θ)P(θ)
RBF:      k(x,z) = exp(-γ‖x-z‖²)
Softmax:  p_i = exp(z_i) / Σexp(z_j)
```
