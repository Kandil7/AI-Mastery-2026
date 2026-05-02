# Dimensionality Reduction

## Introduction

**Dimensionality reduction** transforms high-dimensional data into a lower-dimensional representation while preserving important properties. It's essential for:

- **Visualization**: 2D/3D visualization of high-dimensional data
- **Speed**: Faster training and inference
- **Memory**: Reduced storage requirements
- **Noise reduction**: Remove irrelevant features
- **Avoid curse of dimensionality**: Improve generalization

### The Curse of dimensionality

As dimensions increase:
1. Data becomes sparse
2. Distances become less meaningful
3. Overfitting risk increases
4. Computation becomes expensive

```
Data density in high dimensions

Dimensions: 1D      2D        3D        10D
             ●●●    ● ● ●    ●  ●      ●   ●
            ╱      ╱╲╱╲    ╱ ╲╱ ╲    ╱   ╲ ╱  ╲
           ●      ●   ●   ●    ●    ●     ●  ●
           
Effect:     High   Medium   Low     Very Low
```

---

## 1. Principal Component Analysis (PCA)

### 1.1 Intuition

PCA finds the directions (principal components) of maximum variance in data and projects onto them.

```
2D to 1D projection

x₂                      Original data:
 │                        ●  ●
 │                      ●    ●
 │                    ●  ●
 │                  ●
 │                ●
 │              ●
0┼─────────────────────────────── x₁

x₂                      Project onto PC1:
 │                        ╱
 │                      ╱
 │                    ╱
 │                  ╱
 │                ╱
 │              ╱
0┼───────────────────────────────

PC1 = Direction of maximum variance
PC2 = Perpendicular to PC1
```

### 1.2 Mathematical Foundation

**Goal:** Find orthonormal directions $u_1, u_2, ..., u_d$ such that variance along $u_i$ is maximized.

**Step 1: Center the data**
$$\bar{x} = \frac{1}{m}\sum_{i=1}^{m} x^{(i)}$$
$$X_{centered} = X - \bar{x}$$

**Step 2: Compute covariance matrix**
$$S = \frac{1}{m}\sum_{i=1}^{m}(x^{(i)}-\bar{x})(x^{(i)}-\bar{x})^T = \frac{1}{m}X_{centered}^T X_{centered}$$

**Step 3: Find eigenvalues and eigenvectors**
$$S v_i = \lambda_i v_i$$

- $\lambda_i$ = variance in direction $v_i$
- $v_i$ = principal component direction

**Step 4: Project onto top k components**
$$X_{reduced} = X_{centered} V_k$$

Where $V_k$ contains top k eigenvectors.

### 1.3 Variance Explained

The proportion of variance explained by each component:

$$\text{Variance explained by PC}_i = \frac{\lambda_i}{\sum_{j=1}^{d}\lambda_j}$$

```
Cumulative Variance Explained

Variance│
   100%┤╲
        ╲╲
    90%╲  ╲
        ╲  ╲
    80%╲    ╲
        ╲    ╲
    70%╲      ╲
        ╲      ╲
    60%╲        ╲_______________
        ╲          ╲
    50%╲            ╲
        ╲            ╲
    40%╲              ╲___
        ╲                  ╲___
    30%╲                      ╲___
        ╲                          ╲___
    20%╲                                ╲___
 ──────┼────────────────────────────────────────
        1   2   3   4   5   6   7   8   n_components
                ↑ 90% threshold
```

### 1.4 Implementation from Scratch

```python
import numpy as np

class PCA:
    """Principal Component Analysis implementation from scratch."""
    
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        """Fit PCA to data."""
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        # Using SVD is more numerically stable
        # For n_samples > n_features: SVD on X_centered
        # For n_features > n_samples: SVD on X_centered.T
        
        # Using eigendecomposition on covariance
        cov = (X_centered.T @ X_centered) / n_samples
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components
        if self.n_components is not None:
            self.components_ = eigenvectors[:, :self.n_components]
            self.explained_variance_ = eigenvalues[:self.n_components]
        else:
            self.components_ = eigenvectors
            self.explained_variance_ = eigenvalues
        
        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X):
        """Project data onto principal components."""
        X_centered = X - self.mean_
        
        if self.whiten:
            # Whiten: divide by sqrt of eigenvalues
            scale = np.sqrt(self.explained_variance_ + 1e-10)
            return (X_centered @ self.components_) / scale
        else:
            return X_centered @ self.components_
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_reduced):
        """Reconstruct data from reduced representation."""
        if self.whiten:
            scale = np.sqrt(self.explained_variance_ + 1e-10)
            X_centered = X_reduced * scale
        else:
            X_centered = X_reduced @ self.components_.T
        
        return X_centered + self.mean_
```

### 1.5 Choosing Number of Components

```python
# Method 1: Cumulative variance threshold
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

# Method 2: Elbow method (scree plot)
plt.plot(range(1, n_features + 1), pca.explained_variance_, 'bo-')
plt.xlabel('Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')

# Method 3: Cross-validation (reconstruction error)
```

---

## 2. Kernel PCA

### 2.1 Motivation

Standard PCA finds linear subspaces. Kernel PCA can find non-linear manifolds.

```
Linear vs Non-linear PCA

Original 2D data (swiss roll):      Standard PCA (linear):

     ╱╲                              │
   ╱    ╲                           │
 ╱        ╲__                     ╱╲╱╲
│          ╲__                   ╱   ╲__
╲            ╲__                 ╱       ╲__
 ╲            ╲__               ╱         ╲__
   ╲            ╲__           ╱           ╲__
     ╲              ╲_____╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱

Result: Still looks curved!        Result: Flattens the manifold!

Kernel PCA with RBF kernel:
     ╱╲                              │
   ╱    ╲                           │ (unrolled!)
 ╱        ╲__                     ──┼───────
│          ╲__                    
╲            ╲__                   
 ╲            ╲__                  
   ╲            ╲__               
     ╲              ╲_______
```

### 2.2 Kernel Trick

Instead of computing $\phi(x)$, compute kernel directly:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

Common kernels:
- **Linear**: $K(x, z) = x^T z$
- **Polynomial**: $K(x, z) = (\gamma x^T z + r)^d$
- **RBF (Gaussian)**: $K(x, z) = \exp(-\gamma \|x - z\|^2)$

### 2.3 Implementation

```python
class KernelPCA:
    """Kernel PCA implementation."""
    
    def __init__(self, n_components=None, kernel='rbf', gamma=None, degree=3):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.alphas_ = None
        self.lambdas_ = None
        
    def _compute_kernel(self, X):
        """Compute kernel matrix."""
        n = X.shape[0]
        
        if self.kernel == 'linear':
            return X @ X.T
        
        elif self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[1]
            
            # RBF kernel: exp(-gamma * ||x - z||^2)
            pairwise_sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + \
                               np.sum(X**2, axis=1).reshape(1, -1) - 2 * (X @ X.T)
            return np.exp(-self.gamma * pairwise_sq_dists)
        
        elif self.kernel == 'poly':
            return (self.gamma * X @ X.T + 1) ** self.degree
        
        return X @ X.T
    
    def fit(self, X):
        """Fit Kernel PCA."""
        # Center the kernel matrix
        n = X.shape[0]
        one_n = np.ones((n, n)) / n
        K = self._compute_kernel(X)
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store
        self.lambdas_ = eigenvalues[:self.n_components]
        self.alphas_ = eigenvectors[:, :self.n_components]
        
        # Normalize
        self.alphas_ = self.alphas_ / np.sqrt(self.lambdas_ + 1e-10)
        
        return self
    
    def transform(self, X):
        """Transform new data."""
        # Compute kernel with training data
        if self.kernel == 'linear':
            K = X @ self.X_train_.T
        elif self.kernel == 'rbf':
            # Need to compute kernel for each new point
            pass  # Simplified
        
        return K @ self.alphas_
```

---

## 3. t-SNE (t-Distributed Stochastic Neighbor Embedding)

### 3.1 Purpose

t-SNE is primarily for **visualization** - converting high-dimensional data to 2D or 3D while preserving local structure.

### 3.2 Algorithm

```
t-SNE Algorithm

1. Compute pairwise similarities in high-dimensional space:
   p_{j|i} = exp(-||x_i - x_j||² / 2σ_i²) / Σ_k exp(-||x_i - x_k||² / 2σ_i²)

2. Define similar distribution q in low-dimensional space:
   q_{ij} = (1 + ||y_i - y_j||²)^(-1) / Σ_kl (1 + ||y_k - y_l||²)^(-1)

3. Minimize KL divergence: KL(P || Q)
   → Use gradient descent to find y

4. Result: 2D embedding that preserves local structure
```

### 3.3 Key Properties

- **Non-parametric**: No closed-form solution
- **Stochastic**: Different runs may give different results
- **Slow**: O(n²) - limited to ~10,000 points
- **Preserves local structure**: Similar points stay close

### 3.4 Visualization Example

```
MNIST digits (784D → 2D) with t-SNE

     0     1     2     3
    ╱╲   ╱╲   ╱╲   ╱╲
   ╱ ╲  ╱ ╲  ╱ ╲  ╱ ╲
  ╱   ╲╱   ╲╱   ╲╱   ╲╱
  █   ██   ██   ██   ██
  │    │    │    │    │
  
  Each digit forms a cluster!
```

### 3.5 Implementation

```python
class TSNE:
    """t-SNE implementation (simplified)."""
    
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, 
                 n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
    def _compute_perplexity(self, distances, sigma):
        """Compute perplexity of distribution."""
        # Simplified - in practice use binary search
        probabilities = np.exp(-distances ** 2 / (2 * sigma ** 2))
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        # Shannon entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        perplexity = np.exp(entropy)
        
        return perplexity
    
    def fit_transform(self, X):
        """Fit and transform using t-SNE."""
        n = X.shape[0]
        
        # Compute pairwise distances
        distances = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2)
        
        # Find sigma for each point to achieve target perplexity
        # (simplified - use binary search in practice)
        sigmas = np.ones(n) * np.std(distances)
        
        # Compute P matrix
        P = np.exp(-distances / (2 * sigmas[:, np.newaxis] ** 2))
        np.fill_diagonal(P, 0)
        P = P / P.sum()
        P = (P + P.T) / (2 * n)  # Symmetrize
        
        # Initialize random low-dimensional embeddings
        Y = np.random.randn(n, self.n_components) * 0.0001
        
        # Gradient descent
        for i in range(self.n_iter):
            # Compute q distribution
            distances_y = np.sum((Y[:, np.newaxis] - Y[np.newaxis, :]) ** 2, axis=2)
            q = 1 / (1 + distances_y)
            np.fill_diagonal(q, 0)
            q = q / q.sum()
            
            # Compute gradient
            grad = 4 * (P - q) @ (Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) * \
                   (1 + distances_y)[:, :, np.newaxis]
            grad = grad.sum(axis=1)
            
            # Update with momentum
            if i == 0:
                momentum = np.zeros_like(Y)
            else:
                momentum = 0.8 * momentum - self.learning_rate * grad
            
            Y += momentum
            
            # Learning rate schedule
            if i >= 250:
                self.learning_rate *= 0.99
        
        return Y
```

---

## 4. Comparison of Methods

| Method | Linear/Non-linear | Use Case | Limitations |
|--------|-------------------|----------|--------------|
| **PCA** | Linear | Compression, denoising | Linear only |
| **Kernel PCA** | Non-linear | Non-linear manifolds | Expensive |
| **t-SNE** | Non-linear | Visualization | Slow, stochastic |
| **UMAP** | Non-linear | Visualization | Newer, less proven |
| **LLE** | Non-linear | Manifold learning | Sensitive to noise |
| **Autoencoders** | Non-linear | Deep learning | Needs training |

### When to Use Which:

1. **General compression** → PCA
2. **Visualization** → t-SNE or UMAP
3. **Non-linear structure** → Kernel PCA or Autoencoder
4. **Speed** → UMAP > t-SNE
5. **Interpretability** → PCA (eigenvectors have meaning)

---

## 5. Summary

| Concept | Key Points |
|---------|-------------|
| **PCA** | Linear, variance-based, interpretable components |
| **Kernel PCA** | Non-linear via kernel trick |
| **t-SNE** | Visualization, preserves local structure |
| **Variance explained** | Choose components to retain ~90% variance |
| **Whitening** | Scale components to unit variance |

**Key Insight:** Dimensionality reduction is lossy - information is lost. The goal is to preserve what's important (variance or structure) while discarding noise.