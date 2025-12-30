"""
Unit tests for linear algebra and math operations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

from src.core.math_operations import (
    dot_product, magnitude, normalize, cosine_similarity,
    euclidean_distance, manhattan_distance,
    matrix_multiply, transpose, identity_matrix, trace,
    power_iteration, gram_schmidt, qr_decomposition,
    covariance_matrix, PCA,
    softmax, sigmoid, relu
)


class TestVectorOperations:
    """Tests for vector operations."""
    
    def test_dot_product(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        assert dot_product(v1, v2) == 32
    
    def test_dot_product_orthogonal(self):
        v1 = [1, 0]
        v2 = [0, 1]
        assert dot_product(v1, v2) == 0
    
    def test_magnitude(self):
        v = [3, 4]
        assert magnitude(v) == pytest.approx(5.0)
    
    def test_magnitude_unit(self):
        v = [1, 0, 0]
        assert magnitude(v) == pytest.approx(1.0)
    
    def test_normalize(self):
        v = [3, 4]
        normalized = normalize(v)
        assert magnitude(normalized) == pytest.approx(1.0)
    
    def test_cosine_similarity_same(self):
        v = [1, 2, 3]
        assert cosine_similarity(v, v) == pytest.approx(1.0)
    
    def test_cosine_similarity_opposite(self):
        v1 = [1, 0]
        v2 = [-1, 0]
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)
    
    def test_cosine_similarity_orthogonal(self):
        v1 = [1, 0]
        v2 = [0, 1]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)
    
    def test_euclidean_distance(self):
        v1 = [0, 0]
        v2 = [3, 4]
        assert euclidean_distance(v1, v2) == pytest.approx(5.0)
    
    def test_manhattan_distance(self):
        v1 = [0, 0]
        v2 = [3, 4]
        assert manhattan_distance(v1, v2) == pytest.approx(7.0)


class TestMatrixOperations:
    """Tests for matrix operations."""
    
    def test_matrix_multiply(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        result = matrix_multiply(A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_matrix_multiply_identity(self):
        A = [[1, 2], [3, 4]]
        I = [[1, 0], [0, 1]]
        result = matrix_multiply(A, I)
        np.testing.assert_array_almost_equal(result, A)
    
    def test_transpose(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])
        result = transpose(A)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)
    
    def test_identity_matrix(self):
        I = identity_matrix(3)
        expected = np.eye(3)
        np.testing.assert_array_equal(I, expected)
    
    def test_trace(self):
        A = [[1, 2], [3, 4]]
        assert trace(A) == 5


class TestDecomposition:
    """Tests for matrix decomposition."""
    
    def test_power_iteration(self):
        # Symmetric matrix with known eigenvalues
        A = np.array([[4, 1], [1, 3]])
        eigenvalue, eigenvector = power_iteration(A)
        
        # Dominant eigenvalue should be close to actual
        actual_eigenvalues = np.linalg.eigvalsh(A)
        assert eigenvalue == pytest.approx(max(actual_eigenvalues), rel=0.01)
    
    def test_gram_schmidt(self):
        vectors = np.array([[1, 1], [1, 0]], dtype=float)
        orthonormal = gram_schmidt(vectors)
        
        # Check orthogonality
        dot = np.dot(orthonormal[0], orthonormal[1])
        assert dot == pytest.approx(0.0, abs=1e-10)
        
        # Check unit length
        for v in orthonormal:
            assert np.linalg.norm(v) == pytest.approx(1.0)
    
    def test_qr_decomposition(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        Q, R = qr_decomposition(A)
        
        # Q should be orthogonal
        QtQ = Q.T @ Q
        np.testing.assert_array_almost_equal(QtQ, np.eye(2), decimal=5)
        
        # A = QR
        reconstructed = Q @ R
        np.testing.assert_array_almost_equal(reconstructed, A, decimal=5)


class TestPCA:
    """Tests for PCA."""
    
    def test_pca_dimensionality(self):
        X = np.random.randn(100, 10)
        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(X)
        
        assert X_reduced.shape == (100, 3)
    
    def test_pca_explained_variance(self):
        X = np.random.randn(100, 5)
        pca = PCA(n_components=5)
        pca.fit(X)
        
        # Total explained variance should sum to ~1
        total_var_ratio = sum(pca.explained_variance_ratio_)
        assert total_var_ratio == pytest.approx(1.0, rel=0.01)
    
    def test_pca_inverse_transform(self):
        X = np.random.randn(50, 4)
        pca = PCA(n_components=4)
        X_reduced = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)
        
        # With all components, should reconstruct exactly
        np.testing.assert_array_almost_equal(X, X_reconstructed, decimal=5)


class TestActivations:
    """Tests for activation functions."""
    
    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert np.sum(result) == pytest.approx(1.0)
    
    def test_softmax_all_positive(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = softmax(x)
        assert np.all(result > 0)
    
    def test_sigmoid_range(self):
        x = np.array([-10, 0, 10])
        result = sigmoid(x)
        assert np.all((result >= 0) & (result <= 1))
    
    def test_sigmoid_at_zero(self):
        assert sigmoid(np.array([0]))[0] == pytest.approx(0.5)
    
    def test_relu_positive(self):
        x = np.array([1, 2, 3])
        result = relu(x)
        np.testing.assert_array_equal(result, x)
    
    def test_relu_negative(self):
        x = np.array([-1, -2, -3])
        result = relu(x)
        np.testing.assert_array_equal(result, [0, 0, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
