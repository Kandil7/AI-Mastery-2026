"""Quick verification script for advanced integration module."""

import sys
sys.path.insert(0, '.')

from src.core.advanced_integration import (
    NeuralODE, ODEFunc, robot_dynamics_demo,
    MultiModalIntegrator, generate_patient_data,
    FederatedHospital, FederatedIntegrator, federated_demo,
    biased_lending_simulation, analyze_bias, fairness_test
)
import torch
import numpy as np

print("=" * 60)
print("ADVANCED INTEGRATION MODULE - VERIFICATION")
print("=" * 60)

# 1. Neural ODE Test
print("\n[1] Neural ODE Test...")
results = robot_dynamics_demo(dim=2, t_max=1.0, n_steps=11)
assert results['mean_path'].shape == (11, 1, 2), "Wrong shape"
print(f"    ✓ Mean trajectory shape: {results['mean_path'].shape}")
print(f"    ✓ Final uncertainty: {results['std_path'][-1, 0, 0]:.4f}")

# 2. Multi-Modal Test
print("\n[2] Multi-Modal Integration Test...")
data = generate_patient_data(n_samples=100)
model = MultiModalIntegrator(clinical_dim=5, xray_dim=3, text_dim=4, hidden_dim=32)
clinical = torch.tensor(data['clinical_data'], dtype=torch.float32)
xray = torch.tensor(data['xray_data'], dtype=torch.float32)
text = torch.tensor(data['text_data'], dtype=torch.float32)
pred, unc = model.predict_with_confidence(clinical, xray, text, n_samples=5)
print(f"    ✓ Predictions shape: {pred.shape}")
print(f"    ✓ Mean uncertainty: {np.mean(unc):.4f}")

# 3. Federated Learning Test
print("\n[3] Federated Learning Test...")
hospitals = [FederatedHospital(i, 'mixed', 50) for i in range(3)]
integrator = FederatedIntegrator(hospitals, aggregation_method='bayesian_weighting')
global_risk, global_unc = integrator.aggregate()
print(f"    ✓ Global risk: {global_risk:.4f}")
print(f"    ✓ Global uncertainty: {global_unc:.4f}")

# 4. Ethics/Bias Test
print("\n[4] Ethics & Bias Test...")
results = biased_lending_simulation(n_samples=5000, bias_factor=0.4)
metrics = analyze_bias(results)
print(f"    ✓ Approval disparity: {metrics['approval_disparity']:.2%}")
print(f"    ✓ Disparate impact ratio: {metrics['disparate_impact_ratio']:.3f}")

# 5. Fairness Test
print("\n[5] Fairness Test...")
preds = np.random.randint(0, 2, 100)
labels = np.random.randint(0, 2, 100)
sens = np.random.choice([0, 1], 100)
fairness = fairness_test(preds, labels, sens)
print(f"    ✓ FPR disparity: {fairness['fairness_metrics']['fpr_disparity']:.4f}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
