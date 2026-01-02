"""Quick verification of Phase 11: GNN Integration and Explainable AI."""
import sys
sys.path.insert(0, '.')
import numpy as np

print("=" * 60)
print("PHASE 11: GNN & XAI VERIFICATION")
print("=" * 60)

# Test 1: GNN Integration Module
print("\n[1] GNN Integration Module...")
from src.core.gnn_integration import (
    BayesianGCN,
    generate_synthetic_graph,
    BayesianGCNLayer
)

# Generate graph
graph = generate_synthetic_graph(num_nodes=50, num_classes=2)
print(f"    Generated graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

# Test Bayesian layer
layer = BayesianGCNLayer(in_features=16, out_features=8, num_samples=3)
adj = np.eye(50)
output, uncertainty = layer.forward(graph.x, adj)
assert output.shape == (50, 8), f"Wrong output shape: {output.shape}"
assert len(uncertainty) == 50, f"Wrong uncertainty shape"
print("    ✓ Bayesian GCN layer works")

# Test full model
model = BayesianGCN(
    input_dim=graph.num_features,
    hidden_dim=16,
    output_dim=2,
    num_samples=3
)
prediction = model.predict(graph)
assert len(prediction.predictions) == 50
assert len(prediction.uncertainty) == 50
print("    ✓ Bayesian GCN model works")

# Test training
losses = model.train_step(graph, num_epochs=5)
assert len(losses) == 5
print("    ✓ Training works")
print("    ✓ GNN integration module OK")

# Test 2: Explainable AI Module
print("\n[2] Explainable AI Module...")
from src.core.explainable_ai import (
    ExplainableModel,
    TreeSHAP
)

# Create and train model
model = ExplainableModel(model_type='decision_tree')
data = model.generate_medical_data(n_samples=100)
assert data['X'].shape == (100, 10)
print("    ✓ Medical data generation works")

model.train(data['X'], data['y'], data['feature_names'])
assert model.is_trained
print("    ✓ Model training works")

# Test explanation
explanation = model.predict_with_explanation(data['X'][:1], num_samples=10)[0]
assert len(explanation.shap_values) == 10
assert 0 <= explanation.prediction <= 1
print("    ✓ Local explanation works")

# Test global importance
global_exp = model.get_global_importance(data['X'][:30], num_samples=10)
assert len(global_exp.feature_importance) == 10
print("    ✓ Global importance works")

# Test text explanation
text = model.explain_prediction_text(explanation)
assert 'Prediction' in text
print("    ✓ Text explanation works")
print("    ✓ Explainable AI module OK")

# Summary
print("\n" + "=" * 60)
print("ALL PHASE 11 MODULES VERIFIED ✓")
print("=" * 60)
print("\nModules created:")
print("  - src/core/gnn_integration.py (~480 lines)")
print("  - src/core/explainable_ai.py (~450 lines)")
print("  - tests/test_gnn_xai.py (~270 lines)")
print("\nNotebook updated:")
print("  - Chapter 16: GNN Integration (4 cells)")
print("  - Chapter 17: Explainable AI (4 cells)")
print("  - Interview Questions (1 cell)")
print("\nDocumentation updated:")
print("  - docs/USER_GUIDE.md (sections 5.14-5.15)")
print("  - docs/interview_prep.md (GNN/XAI questions)")
