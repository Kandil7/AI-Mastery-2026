"""
Clustering Implementation
=========================

Unsupervised learning algorithms for grouping data.

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ClusteringResult:
    """Store clustering results"""

    labels: np.ndarray
    centroids: np.ndarray
    inertia: float  # Sum of squared distances to closest centroid
    n_iterations: int


class KMeans:
    """
    K-Means Clustering Algorithm

    Partitions data into K clusters by minimizing within-cluster variance.

    Algorithm:
        1. Initialize K centroids randomly
        2. Repeat until convergence:
           a. Assign each point to nearest centroid
           b. Update centroids to mean of cluster points

    Convergence: When assignments don't change or max iterations reached.

    Time Complexity: O(n * k * t) where n=samples, k=clusters, t=iterations
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iterations = 0

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using K-means++"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]

        # Choose first centroid randomly
        centroids = [X[np.random.randint(n_samples)]]

        # Choose remaining centroids with probability proportional to distance²
        for _ in range(1, self.n_clusters):
            distances = np.array(
                [min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X]
            )
            probs = distances / distances.sum()
            centroids.append(X[np.random.choice(n_samples, p=probs)])

        return np.array(centroids)

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute distance from each point to each centroid"""
        # Use broadcasting: (n_samples, 1, n_features) - (1, n_clusters, n_features)
        # = (n_samples, n_clusters, n_features)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.sum(diff**2, axis=2)

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Fit K-Means to data

        Args:
            X: Input data, shape (n_samples, n_features)
        """
        n_samples = X.shape[0]

        # Initialize centroids
        self.centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iter):
            # Compute distances
            distances = self._compute_distances(X, self.centroids)

            # Assign to nearest centroid
            new_labels = np.argmin(distances, axis=1)

            # Check for convergence
            if self.labels is not None and np.array_equal(new_labels, self.labels):
                break

            self.labels = new_labels

            # Update centroids
            new_centroids = np.zeros((self.n_clusters, X.shape[1]))
            for k in range(self.n_clusters):
                cluster_points = X[self.labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)

            # Check centroid movement
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if centroid_shift < self.tol:
                break

        self.n_iterations = iteration + 1

        # Compute inertia
        distances = self._compute_distances(X, self.centroids)
        self.inertia = np.sum(np.min(distances, axis=1))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)


class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

    Finds clusters based on density, capable of finding arbitrary shapes
    and identifying outliers.

    Parameters:
        eps: Maximum distance between two points to be neighbors
        min_samples: Points needed to form a core point

    Types of points:
        - Core point: Has at least min_samples neighbors within eps
        - Border point: Within eps of a core point but not a core point
        - Outlier: Neither core nor border
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def _find_neighbors(self, X: np.ndarray, point_idx: int) -> np.ndarray:
        """Find all neighbors within eps of point"""
        point = X[point_idx]
        distances = np.linalg.norm(X - point, axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        core_indices: List[int],
        cluster_id: int,
    ):
        """Expand cluster from core points"""
        for core_idx in core_indices:
            # Get neighbors
            neighbors = self._find_neighbors(X, core_idx)

            # Unvisited neighbor becomes part of cluster
            if labels[neighbors] == -1:
                labels[neighbors] = cluster_id

                # If neighbor is also core, add to queue
                if neighbors.size >= self.min_samples:
                    for n in neighbors:
                        if n not in core_indices:
                            core_indices.append(n)

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Fit DBSCAN to data
        """
        n_samples = X.shape[0]

        # Initialize labels: -1 = unvisited
        labels = np.full(n_samples, -1, dtype=int)
        cluster_id = 0

        # Find core points and form clusters
        core_points = []
        for i in range(n_samples):
            if labels[i] != -1:
                continue

            neighbors = self._find_neighbors(X, i)

            if len(neighbors) >= self.min_samples:
                # Core point - start new cluster
                labels[i] = cluster_id
                core_points = list(neighbors)

                # Expand cluster
                self._expand_cluster(X, labels, core_points, cluster_id)
                cluster_id += 1

        self.labels = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict - for new data, assign to nearest cluster"""
        # For simplicity, just return labels for training data
        # For new data, would need to implement properly
        return self.labels


class HierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering

    Bottom-up approach that starts with each point as its own cluster
    and successively merges the closest pairs.

    Linkage methods:
        - single: Minimum pairwise distance
        - complete: Maximum pairwise distance
        - average: Mean pairwise distance
    """

    def __init__(self, n_clusters: int = 3, linkage: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels = None

    def _compute_linkage(
        self, distances: np.ndarray, cluster_sizes: np.ndarray
    ) -> np.ndarray:
        """Compute linkage matrix for different methods"""
        # Simplified - just return distances
        return distances

    def fit(self, X: np.ndarray) -> "HierarchicalClustering":
        """
        Fit hierarchical clustering
        """
        n_samples = X.shape[0]

        # Start with each point as its own cluster
        clusters = list(range(n_samples))
        cluster_sizes = np.ones(n_samples)

        # Initialize labels
        labels = np.arange(n_samples)

        # Compute initial distance matrix
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = np.linalg.norm(X[i] - X[j])
                distances[i, j] = d
                distances[j, i] = d

        # Merge clusters until we have n_clusters
        while len(np.unique(labels)) > self.n_clusters:
            # Find minimum distance pair
            min_dist = float("inf")
            merge = (0, 0)

            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] != labels[j] and distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        merge = (i, j)

            # Merge clusters
            labels[labels == labels[merge[1]]] = labels[merge[0]]

        self.labels = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels"""
        return self.labels


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict:
    """Evaluate clustering quality"""

    n_clusters = len(np.unique(labels))

    # Compute inertia
    unique_labels = np.unique(labels)
    inertia = 0

    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            inertia += np.sum((cluster_points - centroid) ** 2)

    # Compute cluster sizes
    from collections import Counter

    cluster_sizes = Counter(labels)

    return {
        "n_clusters": n_clusters,
        "inertia": inertia,
        "cluster_sizes": dict(cluster_sizes),
        "silhouette": None,  # Would need more complex computation
    }


def demo_clustering():
    """Demonstrate clustering algorithms"""
    print("=" * 60)
    print("Clustering Demo")
    print("=" * 60)

    # Create sample data with 3 clusters
    np.random.seed(42)
    n_samples = 150

    # Cluster 1
    X1 = np.random.randn(50, 2) + np.array([-3, -3])
    # Cluster 2
    X2 = np.random.randn(50, 2) + np.array([0, 3])
    # Cluster 3
    X3 = np.random.randn(50, 2) + np.array([3, -3])

    X = np.vstack([X1, X2, X3])
    true_labels = np.array([0] * 50 + [1] * 50 + [2] * 50)

    print(f"Dataset: {len(X)} samples, 2 features, 3 true clusters")

    # ========================================
    # Demo 1: K-Means
    # ========================================
    print("\n--- K-Means ---")

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    labels_kmeans = kmeans.labels

    eval_kmeans = evaluate_clustering(X, labels_kmeans)
    print(f"Clusters found: {eval_kmeans['n_clusters']}")
    print(f"Inertia: {eval_kmeans['inertia']:.4f}")
    print(f"Cluster sizes: {eval_kmeans['cluster_sizes']}")

    # ========================================
    # Demo 2: DBSCAN
    # ========================================
    print("\n--- DBSCAN ---")

    dbscan = DBSCAN(eps=0.8, min_samples=5)
    dbscan.fit(X)
    labels_dbscan = dbscan.labels

    n_noise = np.sum(labels_dbscan == -1)
    n_clusters = len(np.unique(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)

    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")

    # ========================================
    # Demo 3: Hierarchical
    # ========================================
    print("\n--- Hierarchical ---")

    hc = HierarchicalClustering(n_clusters=3)
    hc.fit(X)
    labels_hc = hc.labels

    eval_hc = evaluate_clustering(X, labels_hc)
    print(f"Clusters found: {eval_hc['n_clusters']}")
    print(f"Cluster sizes: {eval_hc['cluster_sizes']}")

    # ========================================
    # Compare with true labels
    # ========================================
    print("\n--- Comparison ---")
    print("Note: Perfect match would give accuracy ~1.0")
    print(
        f"K-Means matches true: {np.sum(labels_kmeans == true_labels)}/{len(true_labels)}"
    )


if __name__ == "__main__":
    demo_clustering()

    print("\n" + "=" * 60)
    print("Clustering complete!")
    print("=" * 60)
