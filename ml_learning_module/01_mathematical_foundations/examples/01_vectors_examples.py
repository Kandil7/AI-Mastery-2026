"""
Vector Operations - Worked Examples
====================================

This file contains step-by-step worked examples for vector operations.

Author: AI-Mastery-2026
"""

import numpy as np
import matplotlib.pyplot as plt


def example_vector_creation():
    """Example 1: Creating Vectors"""
    print("=" * 60)
    print("Example 1: Creating Vectors")
    print("=" * 60)

    # Method 1: Using numpy array
    v1 = np.array([1, 2, 3])
    print(f"v1 = {v1}")

    # Method 2: Using zeros
    v2 = np.zeros(5)
    print(f"v2 (zeros) = {v2}")

    # Method 3: Using ones
    v3 = np.ones(4)
    print(f"v3 (ones) = {v3}")

    # Method 4: Using linspace
    v4 = np.linspace(0, 10, 5)  # 5 equally spaced points
    print(f"v4 (linspace) = {v4}")

    # Method 5: Using random
    np.random.seed(42)
    v5 = np.random.randn(3)  # Standard normal
    print(f"v5 (random) = {v5}")

    print("\nNote: In ML, we typically use numpy arrays for efficiency.")


def example_vector_operations():
    """Example 2: Basic Vector Operations"""
    print("\n" + "=" * 60)
    print("Example 2: Basic Vector Operations")
    print("=" * 60)

    v = np.array([1, 2, 3])
    w = np.array([4, 5, 6])

    # Addition
    print(f"v = {v}")
    print(f"w = {w}")
    print(f"v + w = {v + w}")  # Element-wise

    # Subtraction
    print(f"v - w = {v - w}")

    # Scalar multiplication
    print(f"2 * v = {2 * v}")
    print(f"-1 * w = {-1 * w}")

    # Element-wise multiplication (Hadamard product)
    print(f"v * w = {v * w}")  # [4, 10, 18]

    # Element-wise division
    print(f"v / w = {v / w}")


def example_dot_product():
    """Example 3: Dot Product"""
    print("\n" + "=" * 60)
    print("Example 3: Dot Product")
    print("=" * 60)

    v = np.array([1, 2, 3])
    w = np.array([4, 5, 6])

    # Method 1: Using numpy
    dot1 = np.dot(v, w)
    print(f"np.dot(v, w) = {dot1}")

    # Method 2: Using @ operator
    dot2 = v @ w
    print(f"v @ w = {dot2}")

    # Method 3: Manual calculation
    dot3 = sum(v[i] * w[i] for i in range(len(v)))
    print(f"Manual: {dot3}")

    # Verification
    print(f"\nVerification: 1*4 + 2*5 + 3*6 = {1 * 4 + 2 * 5 + 3 * 6}")


def example_cross_product():
    """Example 4: Cross Product"""
    print("\n" + "=" * 60)
    print("Example 4: Cross Product")
    print("=" * 60)

    # 3D vectors
    v = np.array([1, 0, 0])
    w = np.array([0, 1, 0])

    cross = np.cross(v, w)
    print(f"v = {v} (x-axis)")
    print(f"w = {w} (y-axis)")
    print(f"v × w = {cross} (z-axis)")

    # Verify perpendicular
    print(f"\nVerification:")
    print(f"v · (v × w) = {np.dot(v, cross)}")  # Should be 0
    print(f"w · (v × w) = {np.dot(w, cross)}")  # Should be 0


def example_norms():
    """Example 5: Vector Norms"""
    print("\n" + "=" * 60)
    print("Example 5: Vector Norms")
    print("=" * 60)

    v = np.array([3, 4])
    print(f"v = {v}")

    # L2 Norm (Euclidean)
    l2 = np.linalg.norm(v)
    print(f"\nL2 Norm (Euclidean):")
    print(f"||v||₂ = √(3² + 4²) = √(9 + 16) = √25 = {l2}")

    # L1 Norm (Manhattan)
    l1 = np.sum(np.abs(v))
    print(f"\nL1 Norm (Manhattan):")
    print(f"||v||₁ = |3| + |4| = {l1}")

    # L-infinity Norm (Max)
    linf = np.max(np.abs(v))
    print(f"\nL-infinity Norm (Max):")
    print(f"||v||∞ = max(|3|, |4|) = {linf}")

    # General Lp norm
    p = 3
    lp = np.sum(np.abs(v) ** p) ** (1 / p)
    print(f"\nL{p} norm: ||v||_{p} = {lp}")


def example_cosine_similarity():
    """Example 6: Cosine Similarity"""
    print("\n" + "=" * 60)
    print("Example 6: Cosine Similarity")
    print("=" * 60)

    # Two similar vectors (pointing in similar direction)
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 4, 6])

    # Two opposite vectors
    v3 = np.array([1, 2, 3])
    v4 = np.array([-1, -2, -3])

    # Two orthogonal vectors
    v5 = np.array([1, 0, 0])
    v6 = np.array([0, 1, 0])

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"Similar vectors: cosine = {cosine_sim(v1, v2):.4f}")
    print(f"Opposite vectors: cosine = {cosine_sim(v3, v4):.4f}")
    print(f"Orthogonal vectors: cosine = {cosine_sim(v5, v6):.4f}")


def example_linear_independence():
    """Example 7: Linear Independence"""
    print("\n" + "=" * 60)
    print("Example 7: Linear Independence")
    print("=" * 60)

    # Linear independent set
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])

    print("Set 1: v1 = (1, 0), v2 = (0, 1)")
    print("These are linearly independent - they form the standard basis")

    # Linear dependent set
    w1 = np.array([1, 2])
    w2 = np.array([2, 4])  # w2 = 2 * w1

    print("\nSet 2: w1 = (1, 2), w2 = (2, 4)")
    print("These are linearly dependent - w2 = 2 * w1")

    # Check using rank
    matrix = np.column_stack([w1, w2])
    print(f"Rank of matrix = {np.linalg.matrix_rank(matrix)}")
    print("Rank < 2 means dependent")


def example_span():
    """Example 8: Span of Vectors"""
    print("\n" + "=" * 60)
    print("Example 8: Span of Vectors")
    print("=" * 60)

    # In 2D, two linearly independent vectors span all of ℝ²
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])

    print("Span of v1=(1,0) and v2=(0,1) = all of ℝ²")

    # One vector spans a line
    v = np.array([1, 2])
    print(f"\nSpan of v={v} = line through origin in direction (1,2)")

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 8))

    # Original vectors
    ax.arrow(
        0,
        0,
        1,
        2,
        head_width=0.1,
        head_length=0.1,
        fc="blue",
        ec="blue",
        label="v = (1,2)",
    )

    # Span: linear combinations
    t = np.linspace(-2, 2, 100)
    span_x = t * 1
    span_y = t * 2
    ax.plot(span_x, span_y, "r--", label="Span of v", alpha=0.7)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Span of a Single Vector in ℝ²")
    ax.legend()
    ax.set_aspect("equal")
    plt.grid(True)
    plt.show()


def visualize_vector_operations():
    """Visualize vector operations"""
    print("\n" + "=" * 60)
    print("Visualizing Vector Operations")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Plot 1: Vector Addition
    ax1 = axes[0, 0]
    v = np.array([2, 1])
    w = np.array([1, 2])

    ax1.arrow(
        0,
        0,
        v[0],
        v[1],
        head_width=0.1,
        head_length=0.1,
        fc="blue",
        ec="blue",
        label="v",
    )
    ax1.arrow(
        0,
        0,
        w[0],
        w[1],
        head_width=0.1,
        head_length=0.1,
        fc="green",
        ec="green",
        label="w",
    )
    ax1.arrow(
        v[0],
        v[1],
        w[0],
        w[1],
        head_width=0.1,
        head_length=0.1,
        fc="green",
        ec="green",
        alpha=0.5,
    )
    ax1.arrow(
        0,
        0,
        v[0] + w[0],
        v[1] + w[1],
        head_width=0.1,
        head_length=0.1,
        fc="red",
        ec="red",
        label="v+w",
    )
    ax1.set_xlim(-1, 4)
    ax1.set_ylim(-1, 4)
    ax1.set_title("Vector Addition (Parallelogram Rule)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect("equal")

    # Plot 2: Scalar Multiplication
    ax2 = axes[0, 1]
    v = np.array([2, 1])

    for c in [0.5, 1, 1.5, 2]:
        ax2.arrow(
            0,
            0,
            c * v[0],
            c * v[1],
            head_width=0.1,
            head_length=0.1,
            fc=f"C{int(c * 2)}",
            ec=f"C{int(c * 2)}",
            label=f"c={c}",
        )
    ax2.set_xlim(-1, 5)
    ax2.set_ylim(-1, 3)
    ax2.set_title("Scalar Multiplication")
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect("equal")

    # Plot 3: Dot Product and Projection
    ax3 = axes[1, 0]
    v = np.array([3, 1])
    w = np.array([2, 2])

    # Project w onto v
    projection = (np.dot(v, w) / np.dot(v, v)) * v

    ax3.arrow(
        0,
        0,
        v[0],
        v[1],
        head_width=0.15,
        head_length=0.1,
        fc="blue",
        ec="blue",
        label="v",
    )
    ax3.arrow(
        0,
        0,
        w[0],
        w[1],
        head_width=0.15,
        head_length=0.1,
        fc="green",
        ec="green",
        label="w",
    )
    ax3.arrow(
        0,
        0,
        projection[0],
        projection[1],
        head_width=0.15,
        head_length=0.1,
        fc="red",
        ec="red",
        label="proj_w_v",
    )
    ax3.set_xlim(-1, 4)
    ax3.set_ylim(-1, 3)
    ax3.set_title("Dot Product & Projection")
    ax3.legend()
    ax3.grid(True)
    ax3.set_aspect("equal")

    # Plot 4: L2 vs L1 norm
    ax4 = axes[1, 1]
    point = np.array([3, 4])

    # L2 norm path
    ax4.plot([0, point[0]], [0, point[1]], "b-", linewidth=2, label="L2 norm path")

    # L1 norm path (Manhattan)
    ax4.plot([0, point[0]], [0, 0], "r--", linewidth=1.5)
    ax4.plot(
        [point[0], point[0]], [0, point[1]], "r--", linewidth=1.5, label="L1 norm path"
    )

    ax4.scatter([point[0]], [point[1]], c="purple", s=100, zorder=5)
    ax4.annotate(f"({point[0]}, {point[1]})", (point[0] + 0.2, point[1] + 0.2))

    ax4.set_xlim(-1, 5)
    ax4.set_ylim(-1, 5)
    ax4.set_title("L2 (blue) vs L1 (red) Norm")
    ax4.legend()
    ax4.grid(True)
    ax4.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Running all examples...")

    example_vector_creation()
    example_vector_operations()
    example_dot_product()
    example_cross_product()
    example_norms()
    example_cosine_similarity()
    example_linear_independence()
    example_span()
    visualize_vector_operations()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
