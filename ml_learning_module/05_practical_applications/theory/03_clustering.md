# Clustering: Unsupervised Learning

## Introduction

**Clustering** is an unsupervised learning technique that groups similar data points together without predefined labels. The goal is to discover natural groupings within the data.

### When to Use Clustering

| Scenario | Example | Clustering Approach |
|----------|---------|---------------------|
| Customer segmentation | Group customers by behavior | K-Means, Hierarchical |
| Document organization | Organize news articles | K-Means, DBSCAN |
| Image compression | Reduce color palette | K-Means |
| Anomaly detection | Find unusual patterns | DBSCAN, Isolation Forest |
| Market research | Discover product categories | Hierarchical |

### Key Assumptions
- Data has natural groupings
- Similar points should be in same cluster
- Clusters are spatially coherent

---

## 1. K-Means Clustering

### 1.1 Algorithm

K-Means partitions data into K clusters by minimizing within-cluster variance:

```
K-Means Algorithm

1. Initialize K cluster centers (centroids)
2. Repeat until convergence:
   a. Assignment step: Assign each point to nearest centroid
   b. Update step: Recompute centroids as mean of assigned points
3. Return cluster assignments and centroids
```

### 1.2 Mathematical Formulation

**Objective:** Minimize sum of squared distances to cluster centers

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

Where:
- $C_k$ = cluster k
- $\mu_k$ = centroid of cluster k
- $\|x_i - \mu_k\|^2$ = squared Euclidean distance

### 1.3 Step-by-Step Visualization

```
Iteration 1:                     Iteration 2:
                               
x₂                              x₂
 │                                │
6┼ ● ●    ●                    6┼ ● ●    ●
 │●   ●  ● ●                   │●   ●  ● ●
 │  ●  ●  ●                    │  ●  ●  ●
 │ ●  ●   ●                     │ ●  ●   ●
 │●   ●  ● ●                   │●   ●  ● ●
4┼────────●─◎─               4┼───◎────────
 │        ●                     │        ●
 │     ●  ●                     │     ●  ●
 │                                │     
0┼────────────────             0┼────────────────
  0    2    4    6    8  x₁      0    2    4    6    8  x₁

  Assign points              Update centroids
  to nearest center            to cluster means
```

### 1.4 Implementation from Scratch

```python
import numpy as np

class KMeans:
    """K-Means Clustering implementation from scratch."""
    
    def __init__(self, n_clusters=3, max_iter=100, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_init = n_init
        self.centroids = None
        self.labels = None
        self.inertia_ = None
    
    def _initialize_centroids(self, X):
        """Initialize centroids using K-Means++ algorithm."""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Choose first centroid randomly
        centroids = [X[np.random.randint(n_samples)]]
        
        # Choose remaining centroids with probability proportional to distance
        for _ in range(self.n_clusters - 1):
            distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) 
                                  for x in X])
            probabilities = distances / distances.sum()
            cumulative = np.cumsum(probabilities)
            r = np.random.rand()
            idx = np.searchsorted(cumulative, r)
            centroids.append(X[idx % n_samples])
        
        return np.array(centroids)
    
    def _assign_clusters(self, X, centroids):
        """Assign each point to nearest centroid."""
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _compute_centroids(self, X, labels):
        """Compute new centroids as mean of cluster points."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                centroids[k] = X[labels == k].mean(axis=0)
        return centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """Compute sum of squared distances to centroids."""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += ((cluster_points - centroids[k]) ** 2).sum()
        return inertia
    
    def fit(self, X):
        """Fit K-Means to data."""
        best_labels = None
        best_centroids = None
        best_inertia = float('inf')
        
        # Run multiple initializations
        for _ in range(self.n_init):
            centroids = self._initialize_centroids(X)
            
            for _ in range(self.max_iter):
                # Assignment step
                labels = self._assign_clusters(X, centroids)
                
                # Update step
                new_centroids = self._compute_centroids(X, labels)
                
                # Check for convergence
                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids
            
            # Compute inertia
            inertia = self._compute_inertia(X, labels, centroids)
            
            # Keep best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centroids = centroids.copy()
        
        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia_ = best_inertia
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels
```

### 1.5 Choosing the Number of Clusters

**Elbow Method:**

```python
# Plot inertia vs number of clusters
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Look for "elbow" point
# Where adding more clusters gives diminishing returns
```

```
Inertia vs K (Elbow Method)

Inertia│
        │
        │      ╱╲
        │     ╱  ╲
        │    ╱    ╲
        │   ╱      ╲___
        │  ╱          ╲___
        │ ╱              ╲___
        │╱                    ╲___
 ──────┼──────────────────────────────
        1   2   3   4   5   6   7   K
                 ↑
               Elbow point (K=3)
```

**Silhouette Score:**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = average distance to other points in same cluster
- $b(i)$ = average distance to points in nearest other cluster

Range: [-1, 1], higher is better.

---

## 2. Hierarchical Clustering

### 2.1 Types of Hierarchical Clustering

| Type | Approach | Use Case |
|------|----------|----------|
| **Agglomerative** | Start with each point, merge iteratively | Most common |
| **Divisive** | Start with all points, split iteratively | Less common |

### 2.2 Agglomerative Algorithm

```
Agglomerative Hierarchical Clustering

Step 1: Start with N clusters (each point is a cluster)
Step 2: Repeat:
   a. Find two closest clusters
   b. Merge them into one cluster
Step 3: Stop when only one cluster remains

Result: Dendrogram showing hierarchy
```

### 2.3 Linkage Methods

**Single Linkage:** Distance = minimum distance between any points in two clusters
- Can create elongated clusters (chaining effect)

**Complete Linkage:** Distance = maximum distance between any points
- Tends to create compact spherical clusters

**Average Linkage:** Distance = mean distance between all pairs
- Good balance between single and complete

**Ward Linkage:** Minimize increase in total within-cluster variance
- Tends to create similar-sized clusters

### 2.4 Dendrogram

```
                    ┌─────────┐
                    │ Cluster │
                    └────┬────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    ┌───┴───┐       ┌───┴───┐       ┌───┴───┐
    │       │       │       │       │       │
  ╱╲      ╱╲     ╱╲      ╱╲     ╱╲      ╱╲
 ●      ●    ●  ●    ●  ●    ●  ●      ●
 │      │    │  │    │  │    │  │      │
 ▼      ▼    ▼  ▼    ▼  ▼    ▼  ▼      ▼
 
 Cut at different heights → different numbers of clusters
```

### 2.5 Implementation

```python
class HierarchicalClustering:
    """Agglomerative Hierarchical Clustering."""
    
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        
    def _compute_distance(self, X, method='ward'):
        """Compute pairwise distances."""
        n = X.shape[0]
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if method == 'euclidean':
                    d = np.linalg.norm(X[i] - X[j])
                elif method == 'ward':
                    # Ward distance: difference in sum of squares
                    d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                distances[i, j] = d
                distances[j, i] = d
        
        return distances
    
    def _merge_clusters(self, distances, clusters):
        """Find and merge closest clusters."""
        n = len(clusters)
        min_dist = float('inf')
        merge_idx = None
        
        for i in range(n):
            for j in range(i + 1, n):
                if distances[clusters[i][0], clusters[j][0]] < min_dist:
                    min_dist = distances[clusters[i][0], clusters[j][0]]
                    merge_idx = (i, j)
        
        # Merge clusters
        new_clusters = [c for idx, c in enumerate(clusters) if idx not in merge_idx]
        new_clusters.append(clusters[merge_idx[0]] + clusters[merge_idx[1]])
        
        return new_clusters
    
    def fit(self, X):
        """Fit hierarchical clustering."""
        n = X.shape[0]
        
        # Start with each point as its own cluster
        clusters = [[i] for i in range(n)]
        
        # Compute distances
        distances = self._compute_distance(X, self.linkage)
        
        # Merge until we have k clusters
        while len(clusters) > self.n_clusters:
            clusters = self._merge_clusters(distances, clusters)
        
        # Assign labels
        self.labels_ = np.zeros(n, dtype=int)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = label
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
```

---

## 3. DBSCAN (Density-Based Clustering)

### 3.1 Motivation

K-Means and hierarchical clustering assume spherical clusters. DBSCAN finds clusters of arbitrary shape.

### 3.2 Key Concepts

- **Core point**: Has at least min_samples points within ε distance
- **Border point**: Within ε of a core point but not a core point itself
- **Noise point**: Neither core nor border point

```
DBSCAN Concepts

         ○
        ○ ○     ○ = Core point (dense region)
    ○ ───── ○   
    ○   ○   ○   ● = Border point
        ●       
           ●    × = Noise point (outlier)
      ×   ●
          ×
```

### 3.3 Algorithm

```
DBSCAN Algorithm

Input: Data X, ε (epsilon), min_samples

1. For each unvisited point:
   a. Find neighbors within ε
   b. If enough neighbors → start new cluster
      - Add all reachable points to cluster
      - Find their neighbors recursively
   c. Otherwise → mark as noise (may be border later)
2. Assign noise points to nearby clusters if border
3. Return cluster labels
```

### 3.4 Implementation

```python
class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise."""
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def _region_query(self, X, point_idx):
        """Find all points within eps of point."""
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]
    
    def _expand_cluster(self, X, labels, point_idx, cluster_id):
        """Expand cluster from seed point."""
        seeds = self._region_query(X, point_idx)
        
        while len(seeds) > 0:
            new_point = seeds[0]
            seeds = seeds[1:]
            
            if labels[new_point] == -1:  # Was noise, now border
                labels[new_point] = cluster_id
            elif labels[new_point] == 0:  # Unvisited
                labels[new_point] = cluster_id
                
                # Add new seeds
                new_seeds = self._region_query(X, new_point)
                if len(new_seeds) >= self.min_samples:
                    seeds = np.concatenate([seeds, new_seeds])
    
    def fit(self, X):
        """Fit DBSCAN."""
        n = X.shape[0]
        labels = np.zeros(n, dtype=int)  # 0 = unvisited
        
        cluster_id = 0
        
        for point_idx in range(n):
            if labels[point_idx] != 0:
                continue
            
            neighbors = self._region_query(X, point_idx)
            
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1  # Noise
            else:
                cluster_id += 1
                labels[point_idx] = cluster_id
                self._expand_cluster(X, labels, point_idx, cluster_id)
        
        # Convert to standard format (noise = -1, clusters = 0, 1, 2, ...)
        self.labels_ = labels - 1  # Now 0, 1, 2,... for clusters, -1 for noise
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
```

### 3.5 DBSCAN vs K-Means

| Aspect | K-Means | DBSCAN |
|--------|---------|--------|
| Clusters | Spherical | Arbitrary shape |
| Outliers | Assigned to nearest | Explicitly detected |
| Parameters | K | eps, min_samples |
| Cluster count | Pre-specified | Data-dependent |

---

## 4. Gaussian Mixture Models (GMM)

### 4.1 Probabilistic Approach

GMM assumes data is generated from a mixture of Gaussian distributions:

$$P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

Where:
- $\pi_k$ = mixing coefficient (weights)
- $\mu_k$ = mean of k-th Gaussian
- $\Sigma_k$ = covariance of k-th Gaussian

### 4.2 EM Algorithm

```
EM for GMM

1. Initialize: Randomly assign points to clusters, compute parameters
2. E-step: Compute probability of each point belonging to each cluster
   P(z_k|x) = π_k * N(x|μ_k, Σ_k) / Σ_j π_j * N(x|μ_j, Σ_j)
3. M-step: Update parameters using computed probabilities
4. Repeat until convergence
```

### 4.3 Soft Clustering

GMM provides **probabilistic cluster assignments** (soft clustering):

```
K-Means (hard):       GMM (soft):

Point A:              Point A:
- Cluster 1: 100%      - Cluster 1: 80%
- Cluster 2: 0%        - Cluster 2: 20%
                      
Useful for uncertainty measurement!
```

---

## 5. Evaluation Metrics

### 5.1 Internal Metrics (without ground truth)

**Inertia (Within-cluster sum of squares):**
- Lower is better
- Use elbow method to find optimal K

**Silhouette Score:**
$$s = \frac{b - a}{\max(a, b)}$$
- Range: [-1, 1]
- Higher is better
- Good for comparing algorithms

**Davies-Bouldin Index:**
$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \frac{s_i + s_j}{d_{ij}}$$
- Lower is better

### 5.2 External Metrics (with ground truth)

**Adjusted Rand Index (ARI):**
- Range: [-1, 1]
- 1 = perfect match
- 0 = random

**Normalized Mutual Information (NMI):**
- Range: [0, 1]
- 1 = perfect match

---

## 6. Summary

| Algorithm | Best For | Weakness |
|-----------|----------|----------|
| **K-Means** | Large data, spherical clusters | Assumes equal variance, requires K |
| **Hierarchical** | Tree structure, unknown K | Computationally expensive |
| **DBSCAN** | Arbitrary shapes, outliers | Sensitive to parameters |
| **GMM** | Probabilistic, overlapping clusters | Can get stuck in local optima |

**Choosing an Algorithm:**

1. Know K? → K-Means or GMM
2. Unknown K? → Hierarchical or DBSCAN
3. Non-spherical shapes? → DBSCAN
4. Need probabilities? → GMM
5. Large data? → K-Means or Mini-batch K-Means