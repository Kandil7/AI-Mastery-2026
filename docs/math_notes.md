# 📐 Math Notes

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
