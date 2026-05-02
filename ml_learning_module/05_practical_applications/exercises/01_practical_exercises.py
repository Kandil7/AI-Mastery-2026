"""
Practical Applications - Exercises
====================================

Practice problems for classification, regression, clustering.

Author: AI-Mastery-2026
"""

import numpy as np


# ============================================================================
# EXERCISE 1: Confusion Matrix
# ============================================================================


def exercise_confusion_matrix():
    """
    Exercise 1: Calculate Confusion Matrix

    Given:
    y_true = [1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 1]

    Calculate:
    - True Positives (TP)
    - False Positives (FP)
    - True Negatives (TN)
    - False Negatives (FN)

    Build 2x2 confusion matrix:
    [[TN, FP], [FN, TP]]
    """
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1])

    # Your code here
    pass


# ============================================================================
# EXERCISE 2: Precision and Recall
# ============================================================================


def exercise_precision_recall():
    """
    Exercise 2: Calculate Precision and Recall

    From confusion matrix:
    TP = 4, FP = 1, FN = 1, TN = 2

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    """
    TP = 4
    FP = 1
    FN = 1
    TN = 2

    # Your code here
    pass


# ============================================================================
# EXERCISE 3: K-Means Initialization
# ============================================================================


def exercise_kmeans_init():
    """
    Exercise 3: K-Means++ Initialization

    Given points:
    X = [[1,1], [2,1], [1,2], [10,10], [11,11], [12,11]]

    Select first centroid as [1,1]

    For second centroid:
    - Calculate distance to first centroid
    - Use distance squared as probability
    - Sample second centroid proportionally

    Show which point gets selected with highest probability
    """
    X = np.array([[1, 1], [2, 1], [1, 2], [10, 10], [11, 11], [12, 11]])
    centroids = [X[0]]  # First centroid

    # Your code here
    print("K-Means++ initialization:")
    print("-" * 40)
    # Show distances and probabilities


# ============================================================================
# EXERCISE 4: Elbow Method
# ============================================================================


def exercise_elbow_method():
    """
    Exercise 4: Determine Optimal K using Elbow Method

    Given inertia values for different K:
    K=1: 500
    K=2: 200
    K=3: 100
    K=4: 90
    K=5: 85

    Which K is optimal? (Look for "elbow" - where decrease slows)
    """
    inertias = {1: 500, 2: 200, 3: 100, 4: 90, 5: 85}

    print("Elbow Method Analysis:")
    print("-" * 40)
    for k, inertia in inertias.items():
        print(f"K={k}: inertia={inertia}")

    # Your analysis here
    print("\nThe elbow appears at K=3 because...")


# ============================================================================
# EXERCISE 5: PCA Variance
# ============================================================================


def exercise_pca_variance():
    """
    Exercise 5: Calculate Explained Variance

    Eigenvalues: [10, 5, 2, 1, 0.5]

    Calculate:
    - Total variance
    - Variance ratio for each component
    - Cumulative variance for first 2 components
    """
    eigenvalues = np.array([10, 5, 2, 1, 0.5])

    # Your code here
    total = sum(eigenvalues)
    ratios = eigenvalues / total

    print(f"Total variance: {total}")
    print(f"Ratios: {ratios}")
    print(f"Cumulative (first 2): {sum(ratios[:2]):.2%}")


# ============================================================================
# EXERCISE 6: DBSCAN Neighbors
# ============================================================================


def exercise_dbscan_neighbors():
    """
    Exercise 6: Determine Core Points in DBSCAN

    Given eps=1.5, min_samples=3:

    Points and their neighbors (within eps):
    A: [B, C]
    B: [A, C]
    C: [A, B, D]
    D: [C, E]
    E: [D]

    Which are core points? (Need ≥ min_samples neighbors)
    """
    neighbors = {
        "A": ["B", "C"],
        "B": ["A", "C"],
        "C": ["A", "B", "D"],
        "D": ["C", "E"],
        "E": ["D"],
    }
    min_samples = 3

    print("DBSCAN Core Points:")
    print("-" * 40)
    # Your code here
    print("Core points: C only (has 3 neighbors)")


# ============================================================================
# EXERCISE 7: Random Forest Voting
# ============================================================================


def exercise_rf_voting():
    """
    Exercise 7: Random Forest Majority Vote

    5 trees predict: [0, 1, 1, 0, 1]

    Final prediction = most common class
    """
    predictions = np.array([0, 1, 1, 0, 1])

    # Your code here
    from collections import Counter

    votes = Counter(predictions)
    final = votes.most_common(1)[0][0]
    print(f"Votes: {dict(votes)}")
    print(f"Final prediction: {final}")


# ============================================================================
# EXERCISE 8: Gradient Boosting Residuals
# ============================================================================


def exercise_gb_residuals():
    """
    Exercise 8: Calculate Pseudo-Residuals for Gradient Boosting

    Given:
    y_true = [3, 5, 7, 9, 11]
    y_pred = [2.5, 4.5, 6.5, 8.5, 10.5]

    Residuals = y_true - y_pred

    Next tree fits these residuals
    """
    y_true = np.array([3, 5, 7, 9, 11])
    y_pred = np.array([2.5, 4.5, 6.5, 8.5, 10.5])

    residuals = y_true - y_pred
    print(f"Residuals: {residuals}")


# ============================================================================
# EXERCISE 9: Ridge vs Lasso
# ============================================================================


def exercise_ridge_vs_lasso():
    """
    Exercise 9: Compare Ridge and Lasso Effects

    Given weights after training with different regularization:

    Without reg: [5, -3, 2, 0.5, -1]
    With Ridge:  [4.5, -2.8, 1.8, 0.4, -0.9]
    With Lasso:  [4, -1, 0.5, 0, 0]

    Which features does Lasso eliminate? What does Ridge do?
    """
    weights = {
        "none": [5, -3, 2, 0.5, -1],
        "ridge": [4.5, -2.8, 1.8, 0.4, -0.9],
        "lasso": [4, -1, 0.5, 0, 0],
    }

    print("Ridge vs Lasso Analysis:")
    print("-" * 40)
    print("Original:  ", weights["none"])
    print("Ridge:     ", weights["ridge"])
    print("Lasso:     ", weights["lasso"])
    print("\nLasso eliminated features at indices 3 and 4")
    print("Ridge shrank all weights but kept all")


# ============================================================================
# EXERCISE 10: Silhouette Score Intuition
# ============================================================================


def exercise_silhouette():
    """
    Exercise 10: Interpret Silhouette Score

    Silhouette = (b - a) / max(a, b)
    - a: avg distance to points in same cluster
    - b: avg distance to points in nearest other cluster

    What does each value mean?
    - Score close to 1:
    - Score close to 0:
    - Score close to -1:
    """
    print("Silhouette Score Interpretation:")
    print("-" * 40)
    print("""
    - Score → 1: Points well-clustered, far from other clusters
    - Score → 0: Points near cluster boundaries
    - Score → -1: Points assigned to wrong clusters
    
    Good clustering: silhouette > 0.5
    """)


# ============================================================================
# SOLUTIONS
# ============================================================================


def solutions():
    """Print solutions"""

    print("=" * 60)
    print("SOLUTIONS")
    print("=" * 60)

    # Solution 1
    print("\n--- Exercise 1: Confusion Matrix ---")
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1])

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    print(f"[[TN={TN}, FP={FP}], [FN={FN}, TP={TP}]]")

    # Solution 2
    print("\n--- Exercise 2: Precision & Recall ---")
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    # Solution 5
    print("\n--- Exercise 5: PCA Variance ---")
    eigenvalues = np.array([10, 5, 2, 1, 0.5])
    total = sum(eigenvalues)
    ratios = eigenvalues / total
    print(f"Total variance: {total}")
    print(f"Ratios: {[f'{r:.2f}' for r in ratios]}")
    print(f"Cumulative (2 components): {sum(ratios[:2]):.2%}")


if __name__ == "__main__":
    print("Running practical exercises...")
    exercise_ridge_vs_lasso()
    # solutions()  # Uncomment to see solutions
