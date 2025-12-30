"""
Comprehensive test suite for the AI-Mastery-2026 project.
This file includes tests for all major components of the project.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.math_operations import (
    dot_product, magnitude, normalize, cosine_similarity,
    euclidean_distance, manhattan_distance,
    matrix_multiply, transpose, identity_matrix, trace,
    power_iteration, gram_schmidt, qr_decomposition,
    covariance_matrix, PCA,
    softmax, sigmoid, relu
)

from src.core.probability import (
    Gaussian, Bernoulli, Categorical, Uniform, Exponential,
    entropy, cross_entropy, kl_divergence, js_divergence,
    bayes_theorem, posterior_update
)

from src.ml.classical import (
    LinearRegressionScratch, LogisticRegressionScratch,
    KNNScratch, DecisionTreeScratch, RandomForestScratch,
    GaussianNBScratch
)

from src.ml.deep_learning import (
    Dense, Activation, Dropout, BatchNormalization,
    MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    NeuralNetwork
)

from src.llm.rag import (
    Document, RetrievalResult, TextChunker,
    EmbeddingModel, Retriever, Reranker,
    ContextAssembler, RAGPipeline
)

from src.llm.attention import (
    scaled_dot_product_attention, MultiHeadAttention,
    TransformerBlock, FeedForwardNetwork, LayerNorm
)

from src.llm.fine_tuning import (
    LoRALayer, LinearWithLoRA, AdapterLayer,
    quantize_nf4, dequantize_nf4
)

from src.production.caching import (
    LRUCache, RedisCache, EmbeddingCache, PredictionCache
)

from src.production.monitoring import (
    DriftDetector, PerformanceMonitor, AlertManager,
    ks_test, psi, chi_square_test
)

from src.production.deployment import (
    ModelSerializer, ModelVersionManager, HealthChecker,
    GracefulShutdown
)


class TestCoreMath:
    """Tests for core mathematical operations."""

    def test_vector_operations(self):
        """Test vector operations."""
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        assert dot_product(v1, v2) == 32
        assert magnitude([3, 4]) == pytest.approx(5.0)
        normalized = normalize([3, 4])
        assert magnitude(normalized) == pytest.approx(1.0)
        assert cosine_similarity(v1, v1) == pytest.approx(1.0)

    def test_matrix_operations(self):
        """Test matrix operations."""
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        result = matrix_multiply(A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_almost_equal(result, expected)

        I = identity_matrix(3)
        expected_I = np.eye(3)
        np.testing.assert_array_equal(I, expected_I)

    def test_pca(self):
        """Test PCA implementation."""
        X = np.random.randn(100, 5)
        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(X)
        assert X_reduced.shape == (100, 3)

    def test_activations(self):
        """Test activation functions."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert np.sum(result) == pytest.approx(1.0, abs=1e-6)
        assert np.all(result > 0)


class TestProbability:
    """Tests for probability distributions and information theory."""

    def test_gaussian(self):
        """Test Gaussian distribution."""
        dist = Gaussian(mean=0, std=1)
        samples = dist.sample(1000)
        assert abs(np.mean(samples)) < 0.5  # Reasonable tolerance
        assert abs(np.std(samples) - 1.0) < 0.5

    def test_bernoulli(self):
        """Test Bernoulli distribution."""
        dist = Bernoulli(p=0.7)
        samples = dist.sample(1000)
        assert abs(np.mean(samples) - 0.7) < 0.1

    def test_information_theory(self):
        """Test information theory functions."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(entropy(p, base=2) - 2.0) < 0.01  # log2(4) = 2

        # KL divergence of same distribution should be 0
        assert abs(kl_divergence(p, p)) < 0.01


class TestClassicalML:
    """Tests for classical ML algorithms."""

    def test_linear_regression(self):
        """Test LinearRegressionScratch."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])  # y = 2x

        model = LinearRegressionScratch(method='closed_form')
        model.fit(X, y)

        # Should find y = 2x approximately
        assert model.weights[0] == pytest.approx(2.0, abs=0.1)

    def test_logistic_regression(self):
        """Test LogisticRegressionScratch."""
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

        model = LogisticRegressionScratch(n_iterations=100)
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy > 0.6  # Should achieve reasonable accuracy

    def test_decision_tree(self):
        """Test DecisionTreeScratch."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])  # AND function

        model = DecisionTreeScratch(max_depth=5)
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy >= 0.75  # Should get most examples right

    def test_random_forest(self):
        """Test RandomForestScratch."""
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

        model = RandomForestScratch(n_estimators=10, max_depth=5)
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy > 0.7


class TestDeepLearning:
    """Tests for deep learning components."""

    def test_dense_layer(self):
        """Test Dense layer."""
        layer = Dense(5, 3)
        X = np.random.randn(10, 5)
        output = layer.forward(X)
        assert output.shape == (10, 3)

    def test_activation_layer(self):
        """Test Activation layer."""
        activation = Activation('relu')
        X = np.array([[-2, -1], [0, 1], [2, 3]])
        output = activation.forward(X)
        expected = np.array([[0, 0], [0, 1], [2, 3]])
        np.testing.assert_array_equal(output, expected)

    def test_neural_network(self):
        """Test NeuralNetwork."""
        model = NeuralNetwork()
        model.add(Dense(4, 8))
        model.add(Activation('relu'))
        model.add(Dense(8, 1))
        model.add(Activation('sigmoid'))
        model.compile(loss=BinaryCrossEntropyLoss(), learning_rate=0.01)

        X = np.random.randn(20, 4)
        y = np.random.randint(0, 2, (20, 1))

        history = model.fit(X, y, epochs=3, batch_size=10, verbose=False)
        assert len(history['loss']) == 3


class TestAttention:
    """Tests for attention mechanisms."""

    def test_scaled_dot_product_attention(self):
        """Test scaled dot product attention."""
        Q = torch.randn(2, 10, 64)
        K = torch.randn(2, 15, 64)
        V = torch.randn(2, 15, 64)

        output, attention_weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == (2, 10, 64)
        assert attention_weights.shape == (2, 10, 15)

    def test_multi_head_attention(self):
        """Test MultiHeadAttention."""
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        X = torch.randn(4, 20, 512)
        output = mha(X, X, X)
        assert output.shape == (4, 20, 512)

    def test_transformer_block(self):
        """Test TransformerBlock."""
        block = TransformerBlock(d_model=256, num_heads=8, d_ff=512)
        X = torch.randn(2, 15, 256)
        output = block(X)
        assert output.shape == (2, 15, 256)


class TestRAG:
    """Tests for RAG pipeline."""

    def test_document(self):
        """Test Document class."""
        doc = Document(content="Test content", metadata={"author": "test"})
        assert doc.content == "Test content"
        assert doc.metadata["author"] == "test"

    def test_rag_pipeline(self):
        """Test RAGPipeline."""
        rag = RAGPipeline()

        docs = [
            Document(content="AI is artificial intelligence"),
            Document(content="Machine learning is part of AI")
        ]
        rag.add_documents(docs)

        response = rag.query("What is AI?", k=2, return_sources=True)

        assert 'answer' in response
        assert 'context' in response
        assert 'sources' in response
        assert len(response['sources']) <= 2


class TestFineTuning:
    """Tests for fine-tuning techniques."""

    def test_lora_layer(self):
        """Test LoRALayer."""
        lora = LoRALayer(in_features=128, out_features=64, r=8)
        x = np.random.randn(10, 128)
        output = lora.forward(x)
        assert output.shape == (10, 64)

    def test_quantization(self):
        """Test quantization functions."""
        weights = np.random.randn(100, 50)
        quantized, scale = quantize_nf4(weights)
        dequantized = dequantize_nf4(quantized, scale)
        
        # Should be close but not exact due to quantization
        assert np.allclose(weights, dequantized, atol=0.5)


class TestProduction:
    """Tests for production components."""

    def test_caching(self):
        """Test caching components."""
        cache = LRUCache(max_size=100)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.exists("key1") == True

    def test_drift_detection(self):
        """Test drift detection."""
        detector = DriftDetector(method='ks', threshold=0.05)
        
        # Set reference data
        ref_data = np.random.normal(0, 1, (100, 2))
        detector.set_reference(ref_data)
        
        # Current data (similar to reference)
        cur_data = np.random.normal(0, 1, (50, 2))
        results = detector.detect_drift(cur_data)
        
        # Should not detect drift in similar data
        assert all(not r.drift_detected for r in results)

    def test_model_serialization(self):
        """Test model serialization."""
        # Create a simple model-like object
        model = {"weights": np.random.randn(10, 5), "bias": np.random.randn(5)}
        
        # Save and load
        serializer = ModelSerializer()
        path = serializer.save(model, "test_model.pkl", format="pickle")
        
        loaded_model = serializer.load(path, format="pickle")
        
        assert np.allclose(model["weights"], loaded_model["weights"])
        assert np.allclose(model["bias"], loaded_model["bias"])


def run_all_tests():
    """Run all tests in the suite."""
    test_classes = [
        TestCoreMath,
        TestProbability,
        TestClassicalML,
        TestDeepLearning,
        TestAttention,
        TestRAG,
        TestFineTuning,
        TestProduction
    ]
    
    for test_class in test_classes:
        test_instance = test_class()
        for attr_name in dir(test_instance):
            if attr_name.startswith('test_'):
                print(f"Running {test_class.__name__}.{attr_name}...")
                try:
                    method = getattr(test_instance, attr_name)
                    method()
                    print(f"  ✓ {attr_name} passed")
                except Exception as e:
                    print(f"  ✗ {attr_name} failed: {e}")
                    return False
    
    print("\nAll tests passed! ✓")
    return True


if __name__ == "__main__":
    run_all_tests()