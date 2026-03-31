"""
Interview Coding Challenges - Matrix Operations
================================================
Common matrix operation problems from ML/AI interviews.

Each challenge includes:
- Problem statement
- Naive solution
- Optimized solution
- Time/space complexity analysis

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Tuple, Optional


# ============================================================
# CHALLENGE 1: Matrix Multiplication
# ============================================================

def matrix_multiply_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Problem: Implement matrix multiplication without using np.dot or @
    
    Naive O(n³) solution using triple nested loops.
    
    Args:
        A: Matrix of shape (m, n)
        B: Matrix of shape (n, p)
    
    Returns:
        Result matrix of shape (m, p)
    
    Time: O(m * n * p)
    Space: O(m * p)
    """
    m, n = A.shape
    n2, p = B.shape
    
    assert n == n2, "Matrix dimensions must match"
    
    C = np.zeros((m, p))
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C


def matrix_multiply_optimized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Optimized using NumPy broadcasting and vectorization.
    
    Key insight: Use np.einsum for flexible tensor operations.
    
    Time: O(m * n * p) but with SIMD parallelism
    Space: O(m * p)
    """
    return np.einsum('ik,kj->ij', A, B)


# ============================================================
# CHALLENGE 2: Matrix Transpose In-Place
# ============================================================

def transpose_square_inplace(A: np.ndarray) -> np.ndarray:
    """
    Problem: Transpose a square matrix in-place.
    
    Key insight: Swap elements across the diagonal.
    Only iterate through upper triangle.
    
    Time: O(n²)
    Space: O(1) - in-place
    """
    n = A.shape[0]
    
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j], A[j, i] = A[j, i], A[i, j]
    
    return A


# ============================================================
# CHALLENGE 3: Find Saddle Points
# ============================================================

def find_saddle_points(A: np.ndarray) -> List[Tuple[int, int]]:
    """
    Problem: Find all saddle points in a matrix.
    A saddle point is minimum in its row and maximum in its column.
    
    Time: O(m * n)
    Space: O(m + n) for precomputed min/max
    """
    m, n = A.shape
    saddle_points = []
    
    # Precompute row minimums and column maximums
    row_mins = np.min(A, axis=1)
    col_maxs = np.max(A, axis=0)
    
    for i in range(m):
        for j in range(n):
            if A[i, j] == row_mins[i] and A[i, j] == col_maxs[j]:
                saddle_points.append((i, j))
    
    return saddle_points


# ============================================================
# CHALLENGE 4: Rotate Matrix 90°
# ============================================================

def rotate_90_clockwise(A: np.ndarray) -> np.ndarray:
    """
    Problem: Rotate NxN matrix 90 degrees clockwise.
    
    Key insight: Transpose + Reverse rows = 90° clockwise
    
    Time: O(n²)
    Space: O(1) if done in-place
    """
    # Transpose
    A = A.T
    
    # Reverse each row
    A = np.flip(A, axis=1)
    
    return A


def rotate_90_inplace(A: np.ndarray) -> np.ndarray:
    """
    In-place rotation using layer-by-layer approach.
    
    Rotate one layer at a time, from outside to inside.
    """
    n = A.shape[0]
    
    for layer in range(n // 2):
        first = layer
        last = n - 1 - layer
        
        for i in range(first, last):
            offset = i - first
            
            # Save top
            top = A[first, i]
            
            # Left -> Top
            A[first, i] = A[last - offset, first]
            
            # Bottom -> Left
            A[last - offset, first] = A[last, last - offset]
            
            # Right -> Bottom
            A[last, last - offset] = A[i, last]
            
            # Top -> Right
            A[i, last] = top
    
    return A


# ============================================================
# CHALLENGE 5: Spiral Matrix Traversal
# ============================================================

def spiral_order(A: np.ndarray) -> List[int]:
    """
    Problem: Return elements in spiral order.
    
    Key insight: Track boundaries (top, bottom, left, right)
    and shrink them as we traverse.
    
    Time: O(m * n)
    Space: O(1) excluding output
    """
    if A.size == 0:
        return []
    
    result = []
    top, bottom = 0, A.shape[0] - 1
    left, right = 0, A.shape[1] - 1
    
    while top <= bottom and left <= right:
        # Traverse right
        for j in range(left, right + 1):
            result.append(A[top, j])
        top += 1
        
        # Traverse down
        for i in range(top, bottom + 1):
            result.append(A[i, right])
        right -= 1
        
        # Traverse left
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(A[bottom, j])
            bottom -= 1
        
        # Traverse up
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(A[i, left])
            left += 1
    
    return result


# ============================================================
# CHALLENGE 6: Matrix Chain Multiplication Order
# ============================================================

def matrix_chain_order(dims: List[int]) -> Tuple[int, str]:
    """
    Problem: Find optimal parenthesization for matrix chain multiplication.
    
    Given dimensions [p0, p1, p2, ..., pn], matrix i has dimensions p[i-1] x p[i].
    Find order to minimize total scalar multiplications.
    
    Dynamic Programming Solution.
    
    Time: O(n³)
    Space: O(n²)
    """
    n = len(dims) - 1
    
    # m[i][j] = minimum cost to multiply matrices i to j
    m = [[0] * n for _ in range(n)]
    
    # s[i][j] = split point for optimal solution
    s = [[0] * n for _ in range(n)]
    
    # l is chain length
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            m[i][j] = float('inf')
            
            for k in range(i, j):
                cost = m[i][k] + m[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    
    def build_parenthesization(i: int, j: int) -> str:
        if i == j:
            return f"M{i}"
        else:
            return f"({build_parenthesization(i, s[i][j])} x {build_parenthesization(s[i][j]+1, j)})"
    
    return m[0][n-1], build_parenthesization(0, n-1)


# ============================================================
# CHALLENGE 7: Strassen's Matrix Multiplication
# ============================================================

def strassen_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Problem: Implement Strassen's algorithm for matrix multiplication.
    
    Reduces complexity from O(n³) to O(n^2.807) using divide and conquer.
    
    Note: Practical only for large matrices due to overhead.
    
    Time: O(n^log₂7) ≈ O(n^2.807)
    Space: O(n² log n) due to recursion
    """
    n = A.shape[0]
    
    # Base case - use regular multiplication for small matrices
    if n <= 64:
        return A @ B
    
    # Pad to power of 2 if necessary
    if n % 2 != 0:
        A_pad = np.zeros((n+1, n+1))
        B_pad = np.zeros((n+1, n+1))
        A_pad[:n, :n] = A
        B_pad[:n, :n] = B
        result = strassen_multiply(A_pad, B_pad)
        return result[:n, :n]
    
    # Split matrices into quadrants
    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]
    
    # Compute 7 products (instead of 8)
    M1 = strassen_multiply(A11 + A22, B11 + B22)
    M2 = strassen_multiply(A21 + A22, B11)
    M3 = strassen_multiply(A11, B12 - B22)
    M4 = strassen_multiply(A22, B21 - B11)
    M5 = strassen_multiply(A11 + A12, B22)
    M6 = strassen_multiply(A21 - A11, B11 + B12)
    M7 = strassen_multiply(A12 - A22, B21 + B22)
    
    # Combine results
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    # Assemble result
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22
    
    return C


# ============================================================
# TESTS
# ============================================================

def run_tests():
    """Run all challenge tests."""
    print("Running Matrix Operation Challenges Tests...")
    
    # Test 1: Matrix multiplication
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 2)
    expected = A @ B
    result = matrix_multiply_naive(A, B)
    assert np.allclose(result, expected), "Matrix multiply failed"
    print("✓ Matrix multiplication")
    
    # Test 2: Transpose
    A = np.random.randn(4, 4).copy()
    expected = A.T.copy()
    result = transpose_square_inplace(A.copy())
    assert np.allclose(result, expected), "Transpose failed"
    print("✓ In-place transpose")
    
    # Test 3: Rotate 90
    A = np.array([[1, 2], [3, 4]])
    expected = np.array([[3, 1], [4, 2]])
    result = rotate_90_clockwise(A.copy())
    assert np.allclose(result, expected), "Rotate failed"
    print("✓ Rotate 90°")
    
    # Test 4: Spiral order
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = spiral_order(A)
    assert result == [1, 2, 3, 6, 9, 8, 7, 4, 5], "Spiral order failed"
    print("✓ Spiral traversal")
    
    # Test 5: Matrix chain
    dims = [10, 30, 5, 60]
    cost, order = matrix_chain_order(dims)
    assert cost == 4500, "Matrix chain order failed"
    print("✓ Matrix chain order")
    
    # Test 6: Strassen
    A = np.random.randn(64, 64)
    B = np.random.randn(64, 64)
    expected = A @ B
    result = strassen_multiply(A, B)
    assert np.allclose(result, expected, rtol=1e-5), "Strassen failed"
    print("✓ Strassen multiplication")
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    run_tests()
