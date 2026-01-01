# 📐 Math Notes

> **📓 For a comprehensive deep-dive with Python implementations, see:**
> [`deep_ml_mathematics_complete.ipynb`](../notebooks/01_mathematical_foundations/deep_ml_mathematics_complete.ipynb)

---

## Linear Algebra

### Matrix Operations

$$
A \cdot B = C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
$$

### Dot Product

$$
\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i \cdot b_i = |\vec{a}| |\vec{b}| \cos\theta
$$

### Cosine Similarity

$$
\text{sim}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| |\vec{b}|}
$$

### Eigenvalues & PageRank

$$
A v = \lambda v \quad \text{(Eigenvalue equation)}
$$

$$
r = d \cdot M \cdot r + \frac{1-d}{n} \quad \text{(PageRank iteration)}
$$

### SVD (Singular Value Decomposition)

$$
A = U \Sigma V^T
$$

---

## Calculus

### Gradient Descent

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta)
$$

### Chain Rule (Backpropagation)

$$
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w_i}
$$

### Self-Attention (Transformers)

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

---

## Optimization

### Lagrangian

$$
\mathcal{L}(x, \lambda) = f(x) - \lambda g(x)
$$

### SVM Dual Form

$$
\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
$$

---

## Probability & Statistics

### Bayes' Theorem

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

### Softmax

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

### Entropy

$$
H(X) = -\sum_{x} p(x) \log p(x)
$$

### KL Divergence

$$
D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

### VAE ELBO

$$
\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
$$

---

## Bayesian Optimization

### Gaussian Process

$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$

### RBF Kernel

$$
k(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2l^2}\right)
$$

### Expected Improvement

$$
\text{EI}(x) = \mathbb{E}[\max(f(x) - f^*, 0)]
$$

---

*Reference: See the full notebook at `notebooks/01_mathematical_foundations/deep_ml_mathematics_complete.ipynb`*
