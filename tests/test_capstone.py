"""
Comprehensive test suite for the GitHub Issue Classifier Capstone Project.

Tests cover:
- Model training pipeline
- API endpoints
- Data preprocessing
- Model predictions
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestDataGeneration:
    """Test synthetic data generation."""
    
    def test_dataset_balance(self):
        """Ensure dataset has balanced classes."""
        from scripts.capstone.train_issue_classifier import generate_synthetic_issues
        
        issues = generate_synthetic_issues(n_samples=1000)
        
        labels = [issue['label'] for issue in issues]
        unique, counts = np.unique(labels, return_counts=True)
        
        # Check all 4 classes present
        assert len(unique) == 4
        
        # Check reasonable balance (within 50 samples of each other)
        assert max(counts) - min(counts) < 50
    
    def test_text_quality(self):
        """Ensure generated text contains key phrases."""
        from scripts.capstone.train_issue_classifier import generate_synthetic_issues
        
        issues = generate_synthetic_issues(n_samples=100)
        
        # Bug issues should contain error-related words
        bugs = [i for i in issues if i['label'] == 'bug']
        bug_texts = ' '.join([b['text'].lower() for b in bugs])
        
        assert any(word in bug_texts for word in ['error', 'crash', 'fails', 'broken'])
        
        # Feature requests should contain improvement words
        features = [i for i in issues if i['label'] == 'feature']
        feature_texts = ' '.join([f['text'].lower() for f in features])
        
        assert any(word in feature_texts for word in ['add', 'support', 'improve', 'feature'])


class TestPreprocessing:
    """Test text preprocessing pipeline."""
    
    def test_tfidf_vectorization(self):
        """Ensure TF-IDF creates correct shape."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        texts = [
            "This is a test document",
            "Another test document here",
            "Third document for testing"
        ]
        
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts)
        
        assert X.shape == (3, 100)
        assert X.dtype == np.float64
    
    def test_text_normalization(self):
        """Test text cleaning."""
        text = "Hello, WORLD!!! This is a TEST???"
        
        # Simple normalization
        normalized = text.lower().replace('!', '').replace('?', '')
        
        assert 'WORLD' not in normalized
        assert 'world' in normalized
        assert '!' not in normalized


class TestModelTraining:
    """Test neural network training."""
    
    def test_model_architecture(self):
        """Verify model can be built with correct shapes."""
        from src.ml.deep_learning import NeuralNetwork, Dense, Activation, Dropout
        
        model = NeuralNetwork()
        model.add(Dense(100, 64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, 32))
        model.add(Activation('relu'))
        model.add(Dense(32, 4))
        model.add(Activation('softmax'))
        
        assert len(model.layers) == 7
        assert model.layers[0].output_size == 64
        assert model.layers[-1].output_size == 4
    
    def test_forward_pass(self):
        """Test model can process a batch."""
        from src.ml.deep_learning import NeuralNetwork, Dense, Activation
        
        model = NeuralNetwork()
        model.add(Dense(10, 5))
        model.add(Activation('relu'))
        model.add(Dense(5, 3))
        model.add(Activation('softmax'))
        
        # Create dummy input
        X = np.random.randn(32, 10)  # Batch of 32 samples
        
        output = model.predict(X)
        
        assert output.shape == (32, 3)
        assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)  # Softmax sums to 1


class TestAPI:
    """Test FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.production.issue_classifier_api import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_classify_endpoint(self, client, monkeypatch):
        """Test /classify endpoint."""
        # Mock model loading to avoid dependency
        class MockModel:
            def predict(self, X):
                return np.array([[0.1, 0.6, 0.2, 0.1]])  # Features class scores
        
        # Mock vectorizer
        class MockVectorizer:
            def transform(self, X):
                return X  # Pass through
        
        # Patch the model and vectorizer
        import src.production.issue_classifier_api as api_module
        monkeypatch.setattr(api_module, 'model', MockModel())
        monkeypatch.setattr(api_module, 'vectorizer', MockVectorizer())
        
        response = client.post(
            "/classify",
            json={"text": "Add dark mode support to the application"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "label" in data
        assert "confidence" in data
        assert "all_probabilities" in data
        assert data["label"] in ["bug", "feature", "question", "documentation"]
    
    def test_classify_empty_text(self, client):
        """Test /classify with empty text."""
        response = client.post(
            "/classify",
            json={"text": ""}
        )
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_model_info_endpoint(self, client):
        """Test /model/info endpoint."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check expected fields are present
        assert "architecture" in data or "model_type" in data
    
    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint (Prometheus format)."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Prometheus metrics are plain text
        assert "# HELP" in response.text or "# TYPE" in response.text


class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_model_save_load(self, tmp_path):
        """Test model can be saved and loaded."""
        from src.ml.deep_learning import NeuralNetwork, Dense, Activation
        import pickle
        
        # Build model
        model = NeuralNetwork()
        model.add(Dense(10, 5))
        model.add(Activation('relu'))
        model.add(Dense(5, 3))
        model.add(Activation('softmax'))
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        
        # Get model parameters
        params = []
        for layer in model.layers:
            if hasattr(layer, 'get_params'):
                params.append(layer.get_params())
            else:
                params.append(None)
        
        with open(model_path, 'wb') as f:
            pickle.dump(params, f)
        
        # Load model
        with open(model_path, 'rb') as f:
            loaded_params = pickle.load(f)
        
        assert len(loaded_params) == len(params)
        
        # Check first layer weights match
        if loaded_params[0] is not None:
            original_weights = np.array(params[0]['weights'])
            loaded_weights = np.array(loaded_params[0]['weights'])
            assert np.allclose(original_weights, loaded_weights)


class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete training → saving → loading → prediction."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from src.ml.deep_learning import NeuralNetwork, Dense, Activation, CrossEntropyLoss
        import pickle
        
        # Step 1: Create dummy training data
        texts = [
            "Bug: application crashes",
            "Bug: error message shown",
            "Feature: add dark mode",
            "Feature: support new format",  "Question: how to install",
            "Question: where is the config"
        ]
        labels = [0, 0, 1, 1, 2, 2]  # 0=bug, 1=feature, 2=question
        
        # Step 2: Vectorize
        vectorizer = TfidfVectorizer(max_features=50)
        X = vectorizer.fit_transform(texts).toarray()
        y = np.array(labels)
        
        # Step 3: Build and train model
        model = NeuralNetwork()
        model.add(Dense(50, 32))
        model.add(Activation('relu'))
        model.add(Dense(32, 3))
        model.add(Activation('softmax'))
        
        model.compile(loss=CrossEntropyLoss(), learning_rate=0.01)
        model.fit(X, y, epochs=10, batch_size=2, verbose=False)
        
        # Step 4: Save model and vectorizer
        model_path = tmp_path / "model.pkl"
        vectorizer_path = tmp_path / "vectorizer.pkl"
        
        params = [layer.get_params() if hasattr(layer, 'get_params') else None 
                  for layer in model.layers]
        
        with open(model_path, 'wb') as f:
            pickle.dump(params, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        # Step 5: Load and predict
        with open(model_path, 'rb') as f:
            loaded_params = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            loaded_vectorizer = pickle.load(f)
        
        # Rebuild model with loaded params
        new_model = NeuralNetwork()
        new_model.add(Dense(50, 32))
        new_model.add(Activation('relu'))
        new_model.add(Dense(32, 3))
        new_model.add(Activation('softmax'))
        
        for layer, params in zip(new_model.layers, loaded_params):
            if params is not None and hasattr(layer, 'set_params'):
                layer.set_params(params)
        
        # Test prediction
        test_text = ["Bug: system freezes"]
        X_test = loaded_vectorizer.transform(test_text).toarray()
        prediction = new_model.predict(X_test)
        
        assert prediction.shape == (1, 3)
        assert np.argmax(prediction) == 0  # Should predict "bug" class


# Performance benchmarks
class TestPerformance:
    """Performance and latency tests."""
    
    def test_inference_latency(self):
        """Ensure inference is fast enough for production."""
        import time
        from src.ml.deep_learning import NeuralNetwork, Dense, Activation
        
        model = NeuralNetwork()
        model.add(Dense(500, 128))
        model.add(Activation('relu'))
        model.add(Dense(128, 4))
        model.add(Activation('softmax'))
        
        # Warm up
        X = np.random.randn(1, 500)
        _ = model.predict(X)
        
        # Benchmark
        latencies = []
        for _ in range(100):
            start = time.time()
            _ = model.predict(X)
            latencies.append((time.time() - start) * 1000)  # Convert to ms
        
        p95_latency = np.percentile(latencies, 95)
        
        print(f"\nInference Latency p95: {p95_latency:.2f}ms")
        
        # Target: <50ms p95 for single prediction
        assert p95_latency < 50, f"Latency too high: {p95_latency:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
