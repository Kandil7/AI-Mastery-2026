"""
Vector Operations - Exercises
=============================

Practice problems for vectors and vector spaces.

Author: AI-Mastery-2026
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# EXERCISE 1: Vector Creation and Basic Operations
# ============================================================================


def exercise_vector_creation():
    """
    Exercise 1: Create vectors and perform basic operations

    Create the following vectors using numpy:
    - a = [1, 2, 3, 4, 5]
    - b = [5, 4, 3, 2, 1]

    Perform and print:
    1. a + b
    2. a - b
    3. 3 * a
    4. a * b (element-wise)
    """
    # Your code here
    pass


def exercise_dot_product():
    """
    Exercise 2: Dot Product

    Given:
    - v1 = [1, 2, 3]
    - v2 = [4, 5, 6]

    Calculate:
    1. Dot product using numpy
    2. Manual calculation
    3. Verify they match
    """
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    # Your code here
    pass


# ============================================================================
# EXERCISE 2: Vector Norms
# ============================================================================


def exercise_l1_norm():
    """
    Exercise 3: L1 Norm

    Calculate L1 norm of v = [3, -4, 5]

    Expected: |3| + |-4| + |5| = 12
    """
    v = np.array([3, -4, 5])

    # Your code here
    pass


def exercise_l2_norm():
    """
    Exercise 4: L2 Norm

    Calculate L2 norm of v = [3, 4]

    Expected: sqrt(3² + 4²) = 5
    """
    v = np.array([3, 4])

    # Your code here
    pass


def exercise_normalize_vector():
    """
    Exercise 5: Normalize a Vector

    Given v = [3, 4], create a unit vector in the same direction.
    A unit vector has L2 norm = 1.

    Formula: v_unit = v / ||v||₂

    Expected result: [0.6, 0.8]
    """
    v = np.array([3, 4])

    # Your code here
    pass


# ============================================================================
# EXERCISE 3: Cosine Similarity
# ============================================================================


def exercise_cosine_similarity():
    """
    Exercise 6: Cosine Similarity

    Calculate cosine similarity between:
    - v1 = [1, 0]
    - v2 = [1, 1]

    Formula: cos(θ) = (v1 · v2) / (||v1|| * ||v2||)

    Expected: ~0.707
    """
    v1 = np.array([1, 0])
    v2 = np.array([1, 1])

    # Your code here
    pass


# ============================================================================
# EXERCISE 4: Linear Independence
# ============================================================================


def exercise_check_independence():
    """
    Exercise 7: Check Linear Independence

    Determine if the following sets are linearly independent:

    Set 1: v1 = [1, 0], v2 = [0, 1]
    Set 2: v1 = [1, 2], v2 = [2, 4]

    Hint: Create a matrix with vectors as columns and check rank
    """
    # Set 1
    set1_v1 = np.array([1, 0])
    set1_v2 = np.array([0, 1])

    # Set 2
    set2_v1 = np.array([1, 2])
    set2_v2 = np.array([2, 4])

    # Your code here
    pass


# ============================================================================
# EXERCISE 5: Projection
# ============================================================================


def exercise_projection():
    """
    Exercise 8: Vector Projection

    Project vector w = [2, 2] onto vector v = [3, 0]

    Formula: proj_v(w) = (w · v / v · v) * v

    Expected: [2, 0]
    """
    v = np.array([3, 0])
    w = np.array([2, 2])

    # Your code here
    pass


# ============================================================================
# EXERCISE 6: Orthogonality
# ============================================================================


def exercise_orthogonal():
    """
    Exercise 9: Check Orthogonality

    Determine if these pairs are orthogonal (dot product = 0):
    - a = [1, 1], b = [1, -1]
    - a = [1, 2], b = [2, 1]
    """
    # Your code here
    pass


# ============================================================================
# EXERCISE 7: Visualization
# ============================================================================


def visualize_exercise():
    """
    Exercise 10: Visualize vectors

    Create a plot showing:
    - Vector v = [3, 2] in blue
    - Vector w = [1, 4] in green
    - Their sum v + w in red

    Use arrow() to draw arrows from origin
    """
    v = np.array([3, 2])
    w = np.array([1, 4])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Your code here - add arrows

    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 7)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Vector Addition Visualization")
    ax.grid(True)
    ax.set_aspect("equal")

    plt.show()


# ============================================================================
# SOLUTIONS (Uncomment to check answers)
# ============================================================================


def solutions():
    """Print solutions to all exercises"""

    print("=" * 60)
    print("SOLUTIONS")
    print("=" * 60)

    # Exercise 1
    print("\n--- Exercise 1: Vector Creation ---")
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([5, 4, 3, 2, 1])
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"3 * a = {3 * a}")
    print(f"a * b = {a * b}")

    # Exercise 2
    print("\n--- Exercise 2: Dot Product ---")
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    dot = np.dot(v1, v2)
    manual = 1 * 4 + 2 * 5 + 3 * 6
    print(f"np.dot(v1, v2) = {dot}")
    print(f"Manual: 1×4 + 2×5 + 3×6 = {manual}")
    print(f"Match: {dot == manual}")

    # Exercise 3
    print("\n--- Exercise 3: L1 Norm ---")
    v = np.array([3, -4, 5])
    l1 = np.sum(np.abs(v))
    print(f"L1 norm = |3| + |-4| + |5| = {l1}")

    # Exercise 4
    print("\n--- Exercise 4: L2 Norm ---")
    v = np.array([3, 4])
    l2 = np.linalg.norm(v)
    print(f"L2 norm = √(3² + 4²) = {l2}")

    # Exercise 5
    print("\n--- Exercise 5: Normalization ---")
    v = np.array([3, 4])
    v_unit = v / np.linalg.norm(v)
    print(f"Unit vector: {v_unit}")
    print(f"New norm: {np.linalg.norm(v_unit)}")

    # Exercise 6
    print("\n--- Exercise 6: Cosine Similarity ---")
    v1 = np.array([1, 0])
    v2 = np.array([1, 1])
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"Cosine similarity = {cos_sim:.4f}")

    # Exercise 7
    print("\n--- Exercise 7: Linear Independence ---")
    set1 = np.column_stack([np.array([1, 0]), np.array([0, 1])])
    set2 = np.column_stack([np.array([1, 2]), np.array([2, 4])])
    print(f"Set 1 rank: {np.linalg.matrix_rank(set1)} (independent)")
    print(f"Set 2 rank: {np.linalg.matrix_rank(set2)} (dependent)")

    # Exercise 8
    print("\n--- Exercise 8: Projection ---")
    v = np.array([3, 0])
    w = np.array([2, 2])
    proj = (np.dot(w, v) / np.dot(v, v)) * v
    print(f"Projection of w onto v = {proj}")

    # Exercise 9
    print("\n--- Exercise 9: Orthogonality ---")
    a1, b1 = np.array([1, 1]), np.array([1, -1])
    a2, b2 = np.array([1, 2]), np.array([2, 1])
    print(f"a=[1,1], b=[1,-1]: dot = {np.dot(a1, b1)} (orthogonal)")
    print(f"a=[1,2], b=[2,1]: dot = {np.dot(a2, b2)} (not orthogonal)")


if __name__ == "__main__":
    print("Running exercises...")

    # Uncomment to see solutions
    # solutions()

    print("\nExercises ready! Uncomment solutions() call to see answers.")
