# Machine Learning Coding Problems - Interview Questions

A collection of hands-on coding problems commonly asked in ML interviews.

---

## Table of Contents
1. [Implement Linear Regression](#1-implement-linear-regression)
2. [K-Nearest Neighbors](#2-k-nearest-neighbors)
3. [k-Means Clustering](#3-k-means-clustering)
4. [Binary Cross-Entropy Loss](#4-binary-cross-entropy-loss)
5. [Softmax and Cross-Entropy](#5-softmax-and-cross-entropy)
6. [Attention Mechanism](#6-attention-mechanism)
7. [BM25 Retrieval](#7-bm25-retrieval)

---

## 1. Implement Linear Regression

### Problem
Implement linear regression from scratch using gradient descent.

### Solution

```python
import numpy as np

class LinearRegression:
    """
    Linear Regression using Gradient Descent.
    
    Model: y = Xw + b
    Loss: MSE = (1/n) * Σ(y - ŷ)²
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iter):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute loss
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)
            
            # Compute gradients
            dw = (-2/n_samples) * np.dot(X.T, (y - y_pred))
            db = (-2/n_samples) * np.sum(y - y_pred)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Test
if __name__ == "__main__":
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.flatten() + np.random.randn(100)
    
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    print(f"Weights: {model.weights[0]:.2f} (expected ~3)")
    print(f"Bias: {model.bias:.2f} (expected ~4)")
```

---

## 2. K-Nearest Neighbors

### Problem
Implement KNN classifier from scratch.

### Solution

```python
import numpy as np
from collections import Counter

class KNN:
    """
    K-Nearest Neighbors classifier.
    
    Algorithm:
    1. Compute distances to all training points
    2. Find k nearest neighbors
    3. Vote on majority class
    """
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict_one(self, x):
        # Compute distances
        distances = [self._euclidean_distance(x, x_train) 
                     for x_train in self.X_train]
        
        # Get k nearest neighbor indices
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=100, n_features=2, 
                               n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy:.2%}")
```

---

## 3. k-Means Clustering

### Problem
Implement k-means clustering from scratch.

### Solution

```python
import numpy as np

class KMeans:
    """
    K-Means clustering algorithm.
    
    Algorithm:
    1. Initialize k centroids randomly
    2. Assign points to nearest centroid
    3. Update centroids as mean of assigned points
    4. Repeat until convergence
    """
    
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].copy()
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            self.labels = self._assign_clusters(X)
            
            # Save old centroids
            old_centroids = self.centroids.copy()
            
            # Update centroids
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = cluster_points.mean(axis=0)
            
            # Check convergence
            if np.sum((self.centroids - old_centroids) ** 2) < self.tol:
                break
        
        return self
    
    def _assign_clusters(self, X):
        distances = np.array([
            np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            for centroid in self.centroids
        ])
        return np.argmin(distances, axis=0)
    
    def predict(self, X):
        return self._assign_clusters(X)
    
    def inertia(self, X):
        """Sum of squared distances to nearest centroid."""
        total = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            total += np.sum((cluster_points - self.centroids[i]) ** 2)
        return total


# Test
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate clustered data
    cluster1 = np.random.randn(30, 2) + [2, 2]
    cluster2 = np.random.randn(30, 2) + [-2, -2]
    cluster3 = np.random.randn(30, 2) + [2, -2]
    X = np.vstack([cluster1, cluster2, cluster3])
    
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    
    print(f"Centroids:\n{kmeans.centroids}")
    print(f"Inertia: {kmeans.inertia(X):.2f}")
```

---

## 4. Binary Cross-Entropy Loss

### Problem
Implement binary cross-entropy loss with gradient.

### Solution

```python
import numpy as np

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary Cross-Entropy Loss.
    
    BCE = -[y*log(p) + (1-y)*log(1-p)]
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        epsilon: Small value to avoid log(0)
    
    Returns:
        Mean BCE loss
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

def binary_cross_entropy_gradient(y_true, y_pred, epsilon=1e-15):
    """
    Gradient of BCE with respect to y_pred.
    
    d(BCE)/d(y_pred) = -y/p + (1-y)/(1-p)
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / len(y_true)


# Test
if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred_good = np.array([0.9, 0.1, 0.8, 0.95, 0.2])
    y_pred_bad = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    print(f"BCE (good predictions): {binary_cross_entropy(y_true, y_pred_good):.4f}")
    print(f"BCE (bad predictions):  {binary_cross_entropy(y_true, y_pred_bad):.4f}")
```

---

## 5. Softmax and Cross-Entropy

### Problem
Implement softmax and multi-class cross-entropy loss.

### Solution

```python
import numpy as np

def softmax(x):
    """
    Numerically stable softmax.
    
    softmax(x)_i = exp(x_i) / Σexp(x_j)
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """
    Multi-class Cross-Entropy Loss.
    
    CE = -Σ y_true * log(y_pred)
    
    Args:
        y_true: One-hot encoded labels [batch, classes]
        y_pred: Predicted probabilities [batch, classes]
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

def cross_entropy_gradient(y_true, y_pred):
    """Gradient when y_pred comes from softmax."""
    return (y_pred - y_true) / len(y_true)


# Test
if __name__ == "__main__":
    # Logits for 3 samples, 4 classes
    logits = np.array([
        [2.0, 1.0, 0.1, 0.5],
        [0.1, 3.0, 0.2, 0.1],
        [0.5, 0.2, 2.5, 0.1]
    ])
    
    # True labels (one-hot)
    y_true = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    
    probs = softmax(logits)
    loss = cross_entropy_loss(y_true, probs)
    
    print(f"Probabilities:\n{probs.round(3)}")
    print(f"Cross-Entropy Loss: {loss:.4f}")
```

---

## 6. Attention Mechanism

### Problem
Implement scaled dot-product attention from scratch.

### Solution

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Args:
        Q: Query matrix [batch, seq_len, d_k]
        K: Key matrix [batch, seq_len, d_k]
        V: Value matrix [batch, seq_len, d_v]
        mask: Optional attention mask
    
    Returns:
        Attention output and weights
    """
    d_k = K.shape[-1]
    
    # Compute attention scores
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    
    # Softmax to get attention weights
    weights = softmax(scores)
    
    # Weighted sum of values
    output = np.matmul(weights, V)
    
    return output, weights


# Test
if __name__ == "__main__":
    np.random.seed(42)
    
    batch_size = 2
    seq_len = 4
    d_k = 8
    d_v = 8
    
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights sum to 1: {np.allclose(weights.sum(axis=-1), 1)}")
```

---

## 7. BM25 Retrieval

### Problem
Implement BM25 retrieval from scratch.

### Solution

```python
import numpy as np
from collections import defaultdict
import math

class BM25:
    """
    BM25 retrieval algorithm.
    
    score(D, Q) = Σ IDF(qi) * [tf(qi, D) * (k1 + 1)] / 
                  [tf(qi, D) + k1 * (1 - b + b * |D|/avgdl)]
    """
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.avgdl = 0
        self.corpus = []
        self.N = 0
    
    def fit(self, corpus):
        """Index the corpus."""
        self.corpus = [doc.lower().split() for doc in corpus]
        self.N = len(self.corpus)
        
        # Calculate document lengths and term frequencies
        for doc in self.corpus:
            self.doc_lengths.append(len(doc))
            seen = set()
            for term in doc:
                if term not in seen:
                    self.doc_freqs[term] += 1
                    seen.add(term)
        
        self.avgdl = sum(self.doc_lengths) / self.N
    
    def _idf(self, term):
        """Inverse document frequency."""
        df = self.doc_freqs.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def _score(self, query_terms, doc_idx):
        """BM25 score for a document."""
        doc = self.corpus[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0
        term_freqs = defaultdict(int)
        for term in doc:
            term_freqs[term] += 1
        
        for term in query_terms:
            tf = term_freqs.get(term, 0)
            idf = self._idf(term)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query, top_k=5):
        """Search the corpus."""
        query_terms = query.lower().split()
        
        scores = [(i, self._score(query_terms, i)) for i in range(self.N)]
        scores.sort(key=lambda x: -x[1])
        
        return scores[:top_k]


# Test
if __name__ == "__main__":
    documents = [
        "Machine learning is AI",
        "Deep learning uses neural networks",
        "NLP processes natural language",
        "Neural networks learn patterns"
    ]
    
    bm25 = BM25()
    bm25.fit(documents)
    
    results = bm25.search("neural networks", top_k=2)
    
    print("Search results for 'neural networks':")
    for idx, score in results:
        print(f"  [{score:.3f}] {documents[idx]}")
```

---

## 📚 Tips for Coding Interviews

1. **Clarify requirements** - Ask about edge cases, expected input size
2. **Start simple** - Get a working solution before optimizing
3. **Test as you go** - Write small tests for each function
4. **Explain your approach** - Talk through your thinking
5. **Know time/space complexity** - Be ready to analyze your solution
6. **Handle edge cases** - Empty inputs, single elements, etc.
