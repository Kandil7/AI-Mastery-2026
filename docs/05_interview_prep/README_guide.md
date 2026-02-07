# AI/ML Interview Preparation Guide

Comprehensive preparation for AI/ML engineering interviews.

---

## 1. Technical Interview Topics

### 1.1 Machine Learning Fundamentals

**Q1: Explain the bias-variance tradeoff.**

```
High Bias: Model is too simple, underfits data
  - Example: Linear model for non-linear data
  - Fix: Use more complex model, add features

High Variance: Model is too complex, overfits data
  - Example: Deep tree memorizing training data
  - Fix: Regularization, more data, simpler model

Optimal: Balance between the two
  Total Error = Bias² + Variance + Irreducible Error
```

**Q2: Derive the gradient for logistic regression.**

```python
# Logistic Regression Loss (Binary Cross-Entropy)
# L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
# where ŷ = σ(wx + b) = 1/(1 + e^(-z))

# Gradient derivation:
# ∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂w
#       = (ŷ - y) · x

# Implementation:
def logistic_gradient(X, y, weights):
    z = X @ weights
    predictions = 1 / (1 + np.exp(-z))
    gradient = X.T @ (predictions - y) / len(y)
    return gradient
```

**Q3: What is the difference between L1 and L2 regularization?**

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| Penalty | λΣ\|w\| | λΣw² |
| Effect | Sparse weights (feature selection) | Small weights (shrinkage) |
| Geometry | Diamond constraint | Circular constraint |
| Use case | Feature selection | Multicollinearity |

---

### 1.2 Deep Learning

**Q4: Explain vanishing/exploding gradients.**

```python
# Problem: In deep networks, gradients can become very small or large

# Vanishing (Sigmoid activation):
# σ'(x) = σ(x)(1-σ(x)) ≤ 0.25
# After n layers: gradient ≤ 0.25^n → 0

# Solutions:
# 1. ReLU activation: ∂ReLU/∂x = 1 (for x > 0)
# 2. Residual connections: gradient = 1 + ∂F/∂x
# 3. Batch normalization
# 4. Proper weight initialization (Xavier, He)

def he_init(fan_in, fan_out):
    """He initialization for ReLU networks"""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std
```

**Q5: Implement attention mechanism from scratch.**

```python
def scaled_dot_product_attention(Q, K, V):
    """
    Q: Query (batch, seq_q, d_k)
    K: Key (batch, seq_k, d_k)
    V: Value (batch, seq_k, d_v)
    
    Returns: (batch, seq_q, d_v)
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores
    scores = Q @ K.transpose(-2, -1)  # (batch, seq_q, seq_k)
    scores = scores / np.sqrt(d_k)   # Scale
    
    # Softmax
    attention_weights = softmax(scores, axis=-1)
    
    # Apply to values
    output = attention_weights @ V
    
    return output, attention_weights
```

**Q6: What is the difference between RNN, LSTM, and Transformer?**

| Model | Memory | Parallelizable | Long-range |
|-------|--------|----------------|------------|
| RNN | Hidden state | No | Poor |
| LSTM | Cell + Hidden | No | Good |
| Transformer | Attention | Yes | Excellent |

---

### 1.3 System Design

**Q7: Design a recommendation system for 10M users, 1M items.**

```
Requirements:
- 10M users, 1M items
- < 100ms latency
- 10K requests/second

Architecture:
┌─────────────────────────────────────────────┐
│                 Load Balancer               │
└─────────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────┐
│              API Gateway (Rate Limit)       │
└─────────────────────────────────────────────┘
                      │
┌──────────┬──────────┬──────────┬──────────┐
│  API-1   │  API-2   │  API-3   │  API-N   │
└──────────┴──────────┴──────────┴──────────┘
         │              │              │
┌────────────────┐ ┌─────────┐ ┌──────────────┐
│  Redis Cache   │ │  HNSW   │ │  User Store  │
│ (hot recs)     │ │ (items) │ │  (postgres)  │
└────────────────┘ └─────────┘ └──────────────┘

Storage:
- User embeddings: 10M × 128 × 4 bytes = 5GB (Redis)
- Item embeddings: 1M × 128 × 4 bytes = 500MB (HNSW)
- Item metadata: PostgreSQL with read replicas

Latency Breakdown:
- Cache lookup: 5ms
- HNSW search: 10ms
- Ranking: 20ms
- Response: 5ms
- Total: ~40ms p50, ~100ms p95
```

**Q8: Design a real-time fraud detection system.**

```
Requirements:
- 10K transactions/second
- < 50ms decision latency
- 99.9% availability

Architecture:
Transaction → Kafka → Feature Engine → ML Model → Decision
     │                      │              │
     └──────────────────────┼──────────────┘
                            ↓
                    ┌───────────────┐
                    │ Feature Store │
                    └───────────────┘

Feature Categories:
1. Transaction: amount, merchant, time
2. Velocity: count/sum in 1h, 24h, 7d
3. Behavioral: avg amount, typical merchants
4. Device: IP, device fingerprint

Model Strategy:
- Stage 1: Rules engine (known patterns)
- Stage 2: XGBoost (fast, interpretable)
- Stage 3: Neural network (complex patterns)

SLA:
- p50: 15ms
- p95: 35ms
- p99: 50ms
```

---

## 2. Coding Challenges

### Challenge 1: Implement K-Means from Scratch

```python
def kmeans(X, k, max_iters=100):
    """
    K-Means clustering implementation
    
    Args:
        X: Data points (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum iterations
    
    Returns:
        centroids: (k, n_features)
        labels: (n_samples,)
    """
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].copy()
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if (labels == i).any() 
            else centroids[i]
            for i in range(k)
        ])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels
```

### Challenge 2: Implement Binary Search Tree for KNN

```python
class KDTree:
    """K-D Tree for efficient nearest neighbor search"""
    
    def __init__(self, points, depth=0):
        n = len(points)
        if n == 0:
            self.node = None
            return
        
        k = len(points[0])
        axis = depth % k
        
        sorted_points = sorted(points, key=lambda x: x[axis])
        mid = n // 2
        
        self.node = sorted_points[mid]
        self.left = KDTree(sorted_points[:mid], depth + 1)
        self.right = KDTree(sorted_points[mid + 1:], depth + 1)
        self.axis = axis
    
    def nearest(self, point, best=None, best_dist=float('inf')):
        if self.node is None:
            return best, best_dist
        
        dist = np.linalg.norm(np.array(point) - np.array(self.node))
        if dist < best_dist:
            best = self.node
            best_dist = dist
        
        # Check which branch to search first
        if point[self.axis] < self.node[self.axis]:
            first, second = self.left, self.right
        else:
            first, second = self.right, self.left
        
        best, best_dist = first.nearest(point, best, best_dist)
        
        # Check if we need to search other branch
        if abs(point[self.axis] - self.node[self.axis]) < best_dist:
            best, best_dist = second.nearest(point, best, best_dist)
        
        return best, best_dist
```

### Challenge 3: Implement Word2Vec (Skip-gram)

```python
class SkipGram:
    """Simple Skip-gram Word2Vec implementation"""
    
    def __init__(self, vocab_size, embed_dim):
        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_out = np.random.randn(embed_dim, vocab_size) * 0.01
    
    def forward(self, center_word_idx):
        hidden = self.W_in[center_word_idx]
        scores = hidden @ self.W_out
        probs = self.softmax(scores)
        return hidden, probs
    
    def train_step(self, center_idx, context_idx, lr=0.01):
        hidden, probs = self.forward(center_idx)
        
        # Gradient
        grad_out = probs.copy()
        grad_out[context_idx] -= 1
        
        # Update weights
        self.W_out -= lr * np.outer(hidden, grad_out)
        self.W_in[center_idx] -= lr * self.W_out @ grad_out
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - x.max())
        return exp_x / exp_x.sum()
```

---

## 3. Behavioral Questions

### STAR Format Responses

**Q: Tell me about a time you optimized a slow ML pipeline.**

```
Situation: Our RAG system had p95 latency of 2.5s, causing user drop-off.

Task: Reduce latency to under 500ms while maintaining retrieval quality.

Action:
1. Profiled the pipeline: embedding (1s), search (1s), LLM (0.5s)
2. Pre-computed embeddings for document corpus
3. Replaced brute-force search with HNSW index
4. Added Redis cache for frequent queries
5. Implemented async embedding for new documents

Result:
- Latency reduced from 2.5s to 580ms (p95)
- Retrieval faithfulness maintained at 92%
- User engagement increased 40%
```

**Q: Describe a project where you balanced technical excellence with business constraints.**

```
Situation: Business wanted GPT-4 quality, but budget was $5K/month.

Task: Build production LLM system within cost constraints.

Action:
1. Benchmarked open-source models vs GPT-4
2. Found Mistral-7B achieved 85% of GPT-4 quality
3. Quantized to 4-bit, reducing GPU requirements 4x
4. Implemented intelligent routing (easy → small model, hard → large)
5. Added caching layer for repeated queries

Result:
- Final cost: $1,200/month (vs $15K for GPT-4)
- User satisfaction: 4.2/5 (vs 4.5/5 target)
- System handled 10x traffic growth
```

---

## 4. Quick Reference Cards

### ML Algorithm Complexity

| Algorithm | Training | Inference | Space |
|-----------|----------|-----------|-------|
| Linear Regression | O(nd²) | O(d) | O(d) |
| Logistic Regression | O(ndk) | O(d) | O(d) |
| Decision Tree | O(n²d) | O(log n) | O(n) |
| Random Forest | O(n²dk) | O(k log n) | O(kn) |
| SVM | O(n³) | O(sv·d) | O(n²) |
| KNN | O(1) | O(nd) | O(nd) |
| Neural Network | O(epochs·n·params) | O(params) | O(params) |

### Common Activation Functions

| Function | Formula | Derivative | Use Case |
|----------|---------|------------|----------|
| Sigmoid | 1/(1+e⁻ˣ) | σ(1-σ) | Binary output |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 1-tanh² | RNN hidden |
| ReLU | max(0,x) | 1 if x>0 | Hidden layers |
| Softmax | eˣⁱ/Σeˣʲ | p(1-p) | Multi-class |

### Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| MSE | (y-ŷ)²/n | Regression |
| Cross-Entropy | -Σy·log(ŷ) | Classification |
| Hinge | max(0, 1-y·ŷ) | SVM |
| Huber | MSE if \|e\|<δ else MAE | Robust regression |

---

## 5. Practice Problems

### Easy
1. Implement gradient descent for linear regression
2. Build a decision stump (1-level tree)
3. Implement TF-IDF from scratch

### Medium
4. Implement backpropagation for a 2-layer network
5. Build a simple RNN cell
6. Implement batch normalization forward/backward

### Hard
7. Implement multi-head attention
8. Build a complete VAE
9. Implement HNSW index

---

## 6. Interview Checklist

### Before Interview
- [ ] Review company's ML products/papers
- [ ] Practice coding on whiteboard/CoderPad
- [ ] Prepare 3-5 STAR stories
- [ ] Review system design patterns

### During Interview
- [ ] Clarify requirements before coding
- [ ] Think out loud
- [ ] Start with brute force, then optimize
- [ ] Test with edge cases

### Questions to Ask
- "What does the ML infrastructure look like?"
- "How do you handle model deployment and monitoring?"
- "What's the typical team composition for ML projects?"
- "How do you balance research vs production work?"
