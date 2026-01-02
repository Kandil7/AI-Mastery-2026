"""
Tests for GNN Integration and Explainable AI modules.
"""

import pytest
import numpy as np


# ============================================================================
# GNN Integration Tests
# ============================================================================

class TestGNNIntegration:
    """Tests for Graph Neural Network integration methods."""
    
    def test_generate_synthetic_graph(self):
        """Test synthetic graph generation."""
        from src.core.gnn_integration import generate_synthetic_graph
        
        graph = generate_synthetic_graph(num_nodes=100, num_classes=3)
        
        assert graph.num_nodes == 100
        assert len(np.unique(graph.y)) == 3
        assert graph.train_mask.sum() + graph.test_mask.sum() == 100
    
    def test_graph_data_properties(self):
        """Test GraphData properties."""
        from src.core.gnn_integration import generate_synthetic_graph
        
        graph = generate_synthetic_graph(num_nodes=50, feature_dim=8)
        
        assert graph.num_nodes == 50
        assert graph.num_features == 8
        assert graph.num_edges >= 0
    
    def test_numpy_gcn_layer_forward(self):
        """Test NumPy GCN layer forward pass."""
        from src.core.gnn_integration import NumpyGCNLayer
        
        layer = NumpyGCNLayer(in_features=16, out_features=8)
        
        # Simple adjacency (identity = self-loops only)
        x = np.random.randn(10, 16).astype(np.float32)
        adj = np.eye(10)
        
        output = layer.forward(x, adj)
        
        assert output.shape == (10, 8)
        assert np.all(np.isfinite(output))
    
    def test_bayesian_gcn_layer_forward(self):
        """Test Bayesian GCN layer forward pass."""
        from src.core.gnn_integration import BayesianGCNLayer
        
        layer = BayesianGCNLayer(in_features=16, out_features=8, num_samples=5)
        
        x = np.random.randn(10, 16).astype(np.float32)
        adj = np.eye(10)
        
        output, uncertainty = layer.forward(x, adj)
        
        assert output.shape == (10, 8)
        assert uncertainty.shape == (10,)
        assert np.all(uncertainty >= 0)
    
    def test_bayesian_gcn_layer_kl_divergence(self):
        """Test KL divergence computation."""
        from src.core.gnn_integration import BayesianGCNLayer
        
        layer = BayesianGCNLayer(in_features=8, out_features=4)
        
        kl = layer.kl_divergence()
        
        assert kl >= 0
        assert np.isfinite(kl)
    
    def test_bayesian_gcn_model(self):
        """Test full Bayesian GCN model."""
        from src.core.gnn_integration import BayesianGCN, generate_synthetic_graph
        
        graph = generate_synthetic_graph(num_nodes=50, num_classes=2)
        
        model = BayesianGCN(
            input_dim=graph.num_features,
            hidden_dim=16,
            output_dim=2,
            num_samples=3
        )
        
        logits, uncertainty = model.forward(graph.x, graph.edge_index)
        
        assert logits.shape == (50, 2)
        assert uncertainty.shape == (50,)
    
    def test_bayesian_gcn_predict(self):
        """Test Bayesian GCN prediction."""
        from src.core.gnn_integration import BayesianGCN, generate_synthetic_graph
        
        graph = generate_synthetic_graph(num_nodes=30)
        
        model = BayesianGCN(
            input_dim=graph.num_features,
            hidden_dim=8,
            output_dim=len(np.unique(graph.y))
        )
        
        prediction = model.predict(graph)
        
        assert len(prediction.predictions) == 30
        assert len(prediction.uncertainty) == 30
        assert all(p in [0, 1, 2] for p in prediction.predictions)
    
    def test_bayesian_gcn_train(self):
        """Test Bayesian GCN training."""
        from src.core.gnn_integration import BayesianGCN, generate_synthetic_graph
        
        graph = generate_synthetic_graph(num_nodes=50)
        
        model = BayesianGCN(
            input_dim=graph.num_features,
            hidden_dim=16,
            output_dim=len(np.unique(graph.y)),
            num_samples=3
        )
        
        losses = model.train_step(graph, num_epochs=10)
        
        assert len(losses) == 10
        assert all(np.isfinite(l) for l in losses)
    
    def test_bayesian_gcn_evaluate(self):
        """Test Bayesian GCN evaluation."""
        from src.core.gnn_integration import BayesianGCN, generate_synthetic_graph
        
        graph = generate_synthetic_graph(num_nodes=50)
        
        model = BayesianGCN(
            input_dim=graph.num_features,
            hidden_dim=16,
            output_dim=len(np.unique(graph.y))
        )
        
        metrics = model.evaluate(graph)
        
        assert 'test_accuracy' in metrics
        assert 'confident_accuracy' in metrics
        assert 'mean_uncertainty' in metrics
        assert 0 <= metrics['test_accuracy'] <= 1


# ============================================================================
# Explainable AI Tests
# ============================================================================

class TestExplainableAI:
    """Tests for Explainable AI methods."""
    
    def test_tree_shap_fit(self):
        """Test TreeSHAP explainer fitting."""
        from src.core.explainable_ai import TreeSHAP
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        explainer = TreeSHAP(model, ['f1', 'f2', 'f3', 'f4', 'f5'])
        explainer.fit(X)
        
        assert explainer.expected_value is not None
    
    def test_tree_shap_values(self):
        """Test SHAP value computation."""
        from src.core.explainable_ai import TreeSHAP
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.randn(50, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        explainer = TreeSHAP(model)
        explainer.fit(X)
        
        shap_values = explainer.shap_values(X[:5], num_samples=20)
        
        # Should have shape (5, 4, 2) for binary classification
        assert shap_values.shape[0] == 5
        assert shap_values.shape[1] == 4
    
    def test_tree_shap_explain_instance(self):
        """Test single instance explanation."""
        from src.core.explainable_ai import TreeSHAP
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.randn(50, 3)
        y = (X.sum(axis=1) > 0).astype(int)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        explainer = TreeSHAP(model, ['a', 'b', 'c'])
        explainer.fit(X)
        
        explanation = explainer.explain_instance(X[0], num_samples=20)
        
        assert len(explanation.feature_names) == 3
        assert len(explanation.shap_values) == 3
        assert 0 <= explanation.prediction <= 1
    
    def test_explainable_model_generate_data(self):
        """Test medical data generation."""
        from src.core.explainable_ai import ExplainableModel
        
        model = ExplainableModel()
        data = model.generate_medical_data(n_samples=100)
        
        assert data['X'].shape == (100, 10)
        assert len(data['y']) == 100
        assert len(data['feature_names']) == 10
    
    def test_explainable_model_train(self):
        """Test explainable model training."""
        from src.core.explainable_ai import ExplainableModel
        
        model = ExplainableModel(model_type='decision_tree')
        data = model.generate_medical_data(n_samples=200)
        
        metrics = model.train(data['X'], data['y'], data['feature_names'])
        
        assert 'accuracy' in metrics
        assert model.is_trained
    
    def test_explainable_model_predict_with_explanation(self):
        """Test prediction with explanation."""
        from src.core.explainable_ai import ExplainableModel
        
        model = ExplainableModel()
        data = model.generate_medical_data(n_samples=200)
        model.train(data['X'], data['y'], data['feature_names'])
        
        explanations = model.predict_with_explanation(data['X'][:3], num_samples=20)
        
        assert len(explanations) == 3
        for exp in explanations:
            assert len(exp.feature_names) == 10
            assert len(exp.shap_values) == 10
    
    def test_explainable_model_global_importance(self):
        """Test global feature importance."""
        from src.core.explainable_ai import ExplainableModel
        
        model = ExplainableModel()
        data = model.generate_medical_data(n_samples=200)
        model.train(data['X'], data['y'], data['feature_names'])
        
        global_exp = model.get_global_importance(data['X'][:50], num_samples=20)
        
        assert len(global_exp.feature_importance) == 10
        assert all(v >= 0 for v in global_exp.feature_importance.values())
    
    def test_explainable_model_text_explanation(self):
        """Test text explanation generation."""
        from src.core.explainable_ai import ExplainableModel
        
        model = ExplainableModel()
        data = model.generate_medical_data(n_samples=200)
        model.train(data['X'], data['y'], data['feature_names'])
        
        explanation = model.predict_with_explanation(data['X'][:1], num_samples=20)[0]
        text = model.explain_prediction_text(explanation)
        
        assert 'Prediction' in text
        assert 'factors' in text.lower()


# ============================================================================
# Demo Function Tests
# ============================================================================

class TestDemoFunctions:
    """Test that demo functions run without error."""
    
    def test_gnn_integration_demo(self):
        """Test GNN integration demo runs."""
        from src.core.gnn_integration import gnn_integration_demo
        
        results = gnn_integration_demo()
        
        assert 'graph' in results
        assert 'model' in results
        assert 'metrics' in results
    
    def test_explainable_ai_demo(self):
        """Test XAI demo runs."""
        from src.core.explainable_ai import explainable_ai_demo
        
        results = explainable_ai_demo()
        
        assert 'model' in results
        assert 'data' in results
        assert 'global_explanation' in results


# ============================================================================
# Integration Tests
# ============================================================================

class TestModuleImports:
    """Test that all modules can be imported."""
    
    def test_import_gnn_integration(self):
        """Test GNN integration module import."""
        from src.core import gnn_integration
        
        assert hasattr(gnn_integration, 'BayesianGCN')
        assert hasattr(gnn_integration, 'generate_synthetic_graph')
        assert hasattr(gnn_integration, 'GraphData')
    
    def test_import_explainable_ai(self):
        """Test explainable AI module import."""
        from src.core import explainable_ai
        
        assert hasattr(explainable_ai, 'ExplainableModel')
        assert hasattr(explainable_ai, 'TreeSHAP')
        assert hasattr(explainable_ai, 'FeatureExplanation')
