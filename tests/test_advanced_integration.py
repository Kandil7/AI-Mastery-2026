"""
Tests for Advanced Integration Methods.

This module tests:
- Neural ODEs with uncertainty quantification
- Multi-Modal Integration with MC Dropout
- Federated Learning aggregation strategies
- Ethics and bias detection functions
"""

import numpy as np
import pytest
import torch

from src.core.advanced_integration import (
    # Neural ODEs
    NeuralODE,
    ODEFunc,
    robot_dynamics_demo,
    # Multi-Modal
    MultiModalIntegrator,
    generate_patient_data,
    # Federated Learning
    FederatedHospital,
    FederatedIntegrator,
    UncertainNN,
    federated_demo,
    # Ethics
    biased_lending_simulation,
    analyze_bias,
    fairness_test,
)


# =============================================================================
# Neural ODE Tests
# =============================================================================


class TestNeuralODE:
    """Test suite for Neural ODE implementation."""

    def test_ode_func_forward(self):
        """ODEFunc should produce output of same dimension as input."""
        func = ODEFunc(dim=4, hidden_dim=32)
        x = torch.randn(8, 4)
        t = torch.tensor(0.0)
        output = func(t, x)

        assert output.shape == x.shape

    def test_neural_ode_euler(self):
        """NeuralODE with Euler method should produce trajectory."""
        func = ODEFunc(dim=2)
        model = NeuralODE(func, method="euler")

        x0 = torch.randn(4, 2)
        t = torch.linspace(0, 1, 11)
        trajectory = model(x0, t)

        assert trajectory.shape == (11, 4, 2)

    def test_neural_ode_rk4(self):
        """NeuralODE with RK4 method should produce trajectory."""
        func = ODEFunc(dim=3)
        model = NeuralODE(func, method="rk4")

        x0 = torch.randn(2, 3)
        t = torch.linspace(0, 2, 21)
        trajectory = model(x0, t)

        assert trajectory.shape == (21, 2, 3)

    def test_integrate_with_uncertainty(self):
        """Uncertainty integration should return mean, std, and samples."""
        func = ODEFunc(dim=2)
        model = NeuralODE(func)

        x0 = torch.tensor([[0.0, 1.0]])
        t = torch.linspace(0, 1, 11)

        mean_path, std_path, trajectories = model.integrate_with_uncertainty(
            x0, t, num_samples=10
        )

        assert mean_path.shape == (11, 1, 2)
        assert std_path.shape == (11, 1, 2)
        assert trajectories.shape[0] == 10
        # Std should be non-negative
        assert np.all(std_path >= 0)

    def test_robot_dynamics_demo(self):
        """Robot dynamics demo should return valid results."""
        results = robot_dynamics_demo(dim=2, t_max=1.0, n_steps=11)

        assert "mean_path" in results
        assert "std_path" in results
        assert "trajectories" in results
        assert "t_span" in results
        assert len(results["t_span"]) == 11


# =============================================================================
# Multi-Modal Integration Tests
# =============================================================================


class TestMultiModalIntegration:
    """Test suite for Multi-Modal Integration."""

    def test_generate_patient_data(self):
        """Patient data generation should create valid arrays."""
        data = generate_patient_data(n_samples=100)

        assert data["clinical_data"].shape == (100, 5)
        assert data["xray_data"].shape == (100, 3)
        assert data["text_data"].shape == (100, 4)
        assert data["labels"].shape == (100,)
        assert set(np.unique(data["labels"])).issubset({0, 1})

    def test_multimodal_integrator_forward(self):
        """MultiModalIntegrator forward pass should work."""
        model = MultiModalIntegrator(
            clinical_dim=5, xray_dim=3, text_dim=4, hidden_dim=32
        )

        clinical = torch.randn(8, 5)
        xray = torch.randn(8, 3)
        text = torch.randn(8, 4)

        mean, uncertainty = model(clinical, xray, text)

        assert mean.shape == (8,)
        assert uncertainty.shape == (8,)
        # Mean should be in [0, 1] (sigmoid)
        assert torch.all(mean >= 0) and torch.all(mean <= 1)
        # Uncertainty should be positive
        assert torch.all(uncertainty > 0)

    def test_predict_with_confidence(self):
        """MC Dropout prediction should return valid uncertainty."""
        model = MultiModalIntegrator(
            clinical_dim=5, xray_dim=3, text_dim=4, hidden_dim=32
        )

        clinical = torch.randn(4, 5)
        xray = torch.randn(4, 3)
        text = torch.randn(4, 4)

        mean_pred, total_unc = model.predict_with_confidence(
            clinical, xray, text, n_samples=10
        )

        assert mean_pred.shape == (4,)
        assert total_unc.shape == (4,)
        # All values should be valid
        assert np.all(np.isfinite(mean_pred))
        assert np.all(np.isfinite(total_unc))
        assert np.all(total_unc >= 0)


# =============================================================================
# Federated Learning Tests
# =============================================================================


class TestFederatedLearning:
    """Test suite for Federated Learning."""

    def test_federated_hospital_creation(self):
        """FederatedHospital should generate valid local data."""
        hospital = FederatedHospital(1, "elderly", n_patients=100)

        assert hospital.data.hospital_id == 1
        assert hospital.data.distribution == "elderly"
        assert len(hospital.data.age) == 100
        # Elderly hospital should have higher average age
        assert np.mean(hospital.data.age) > 60

    def test_hospital_local_estimate(self):
        """Hospital should compute valid local estimates."""
        hospital = FederatedHospital(0, "young", n_patients=50)
        estimate = hospital.compute_local_estimate()

        assert "hospital_id" in estimate
        assert "local_risk" in estimate
        assert "local_uncertainty" in estimate
        assert "sample_size" in estimate
        assert 0 <= estimate["local_risk"] <= 1
        assert estimate["local_uncertainty"] > 0

    def test_uncertain_nn(self):
        """UncertainNN should output predictions and uncertainties."""
        model = UncertainNN(input_dim=4, hidden_dim=16)
        x = torch.randn(8, 4)

        mean, uncertainty = model(x)

        assert mean.shape == (8,)
        assert uncertainty.shape == (8,)
        assert torch.all(mean >= 0) and torch.all(mean <= 1)
        assert torch.all(uncertainty > 0)

    def test_federated_integrator_aggregation(self):
        """FederatedIntegrator should aggregate hospital estimates."""
        hospitals = [
            FederatedHospital(0, "young", 100),
            FederatedHospital(1, "elderly", 100),
            FederatedHospital(2, "mixed", 100),
        ]
        integrator = FederatedIntegrator(hospitals, "bayesian_weighting")

        global_risk, global_unc = integrator.aggregate()

        assert 0 <= global_risk <= 1
        assert global_unc > 0
        assert len(integrator.history) == 1

    def test_aggregation_methods(self):
        """Different aggregation methods should produce different results."""
        hospitals = [
            FederatedHospital(0, "young", 50),
            FederatedHospital(1, "elderly", 200),
        ]

        results = {}
        for method in ["simple_average", "sample_weighted", "bayesian_weighting"]:
            integrator = FederatedIntegrator(hospitals, method)
            risk, _ = integrator.aggregate()
            results[method] = risk

        # Sample weighted should be closer to elderly hospital (larger sample)
        # Simple average should be between the two
        assert results["simple_average"] != results["sample_weighted"]

    def test_federated_demo(self):
        """Federated demo should run without errors."""
        results = federated_demo(n_hospitals=3, n_rounds=2)

        assert "results" in results
        assert "true_risk" in results
        assert "hospital_info" in results
        assert len(results["results"]) == 4  # 4 methods


# =============================================================================
# Ethics and Bias Tests
# =============================================================================


class TestEthicsAndBias:
    """Test suite for Ethics and Bias detection."""

    def test_biased_lending_simulation(self):
        """Lending simulation should create valid data with bias."""
        results = biased_lending_simulation(n_samples=1000, bias_factor=0.5)

        assert results["sensitive_attr"].shape == (1000,)
        assert results["approved"].shape == (1000,)
        assert results["true_worth"].shape == (1000,)
        assert results["perceived_worth"].shape == (1000,)

        # Check sensitive attribute distribution
        assert 0.2 < np.mean(results["sensitive_attr"]) < 0.4

    def test_analyze_bias(self):
        """Bias analysis should detect disparities."""
        results = biased_lending_simulation(n_samples=5000, bias_factor=0.4)
        metrics = analyze_bias(results)

        assert "approval_rate_group0" in metrics
        assert "approval_rate_group1" in metrics
        assert "approval_disparity" in metrics
        assert "disparate_impact_ratio" in metrics

        # With bias factor 0.4, there should be measurable disparity
        assert metrics["approval_disparity"] > 0.05

    def test_no_bias_scenario(self):
        """With no bias, disparity should be minimal."""
        results = biased_lending_simulation(n_samples=5000, bias_factor=0.0)
        metrics = analyze_bias(results)

        # Without bias, disparity should be smaller
        assert metrics["approval_disparity"] < metrics["approval_rate_group0"]

    def test_fairness_test(self):
        """Fairness test should compute valid metrics."""
        np.random.seed(42)
        n = 1000
        predictions = np.random.randint(0, 2, n)
        labels = np.random.randint(0, 2, n)
        sensitive = np.random.choice([0, 1], n, p=[0.6, 0.4])

        results = fairness_test(predictions, labels, sensitive)

        assert "group_0" in results
        assert "group_1" in results
        assert "fairness_metrics" in results

        # Check metric ranges
        for group in ["group_0", "group_1"]:
            assert 0 <= results[group]["accuracy"] <= 1
            assert 0 <= results[group]["fpr"] <= 1
            assert 0 <= results[group]["fnr"] <= 1

    def test_fairness_test_perfect_classifier(self):
        """Perfect classifier should have zero error rates."""
        predictions = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        results = fairness_test(predictions, labels, sensitive)

        for group in ["group_0", "group_1"]:
            assert results[group]["accuracy"] == 1.0
            assert results[group]["fpr"] == 0.0
            assert results[group]["fnr"] == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_healthcare_pipeline(self):
        """Full pipeline from data generation to prediction."""
        # Generate data
        data = generate_patient_data(n_samples=200)

        # Create model
        model = MultiModalIntegrator(
            clinical_dim=5, xray_dim=3, text_dim=4, hidden_dim=32
        )

        # Prepare tensors
        clinical = torch.tensor(data["clinical_data"], dtype=torch.float32)
        xray = torch.tensor(data["xray_data"], dtype=torch.float32)
        text = torch.tensor(data["text_data"], dtype=torch.float32)

        # Get predictions
        mean_pred, uncertainty = model.predict_with_confidence(
            clinical, xray, text, n_samples=5
        )

        # Check output
        assert len(mean_pred) == 200
        assert len(uncertainty) == 200
        assert np.all(np.isfinite(mean_pred))

    def test_federated_with_neural_ode(self):
        """Federated learning should work with any local model."""
        hospitals = [FederatedHospital(i, "mixed", 50) for i in range(3)]
        integrator = FederatedIntegrator(hospitals)

        # Run multiple rounds
        for _ in range(3):
            risk, unc = integrator.aggregate()

        assert len(integrator.history) == 3
        # Results should converge
        assert abs(integrator.history[-1] - integrator.history[-2]) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
