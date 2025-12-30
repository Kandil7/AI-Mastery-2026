"""
Unit tests for AI logic modules.
"""

import pytest
import numpy as np


class TestMatrixOperations:
    """Tests for matrix operations."""

    def test_dot_product(self):
        """Test dot product calculation."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.dot(a, b)
        assert result == 32

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        result = np.matmul(A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result, expected)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert similarity == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
