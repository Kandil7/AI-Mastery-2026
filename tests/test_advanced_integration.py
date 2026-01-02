import pytest
import torch
import numpy as np
from src.core.advanced_integration import (
    NeuralODE,
    ODEFunc,
    MultiModalIntegrator,
    FederatedIntegrator,
    biased_lending_simulation,
)


class TestNeuralODE:
    def test_forward_pass(self):
        func = ODEFunc(dim=2)
        model = NeuralODE(func)
        x0 = torch.tensor([0.0, 1.0])
        t_span = torch.linspace(0, 1, 10)

        traj = model(x0, t_span)
        # Output shape should be [10, 2]
        assert traj.shape == (10, 2)
        # Initial state should match x0 (approximately, depending on solver step 0 handling)
        assert torch.allclose(traj[0], x0)

    def test_uncertainty_integration(self):
        func = ODEFunc(dim=2)
        model = NeuralODE(func)
        x0 = torch.tensor([0.0, 1.0])
        t_span = torch.linspace(0, 1, 5)

        mean, std, trajectories = model.integrate_with_uncertainty(
            x0, t_span, num_samples=5
        )

        assert mean.shape == (5, 2)
        assert std.shape == (5, 2)
        assert trajectories.shape == (5, 5, 2)


class TestMultiModalIntegrator:
    def test_fusion(self):
        # dims: clinical=5, xray=10, text=8
        model = MultiModalIntegrator(5, 10, 8)

        batch_size = 4
        c = torch.randn(batch_size, 5)
        x = torch.randn(batch_size, 10)
        t = torch.randn(batch_size, 8)

        mean, uncertainty = model(c, x, t)

        assert mean.shape == (batch_size,)
        assert uncertainty.shape == (batch_size,)
        assert torch.all(uncertainty > 0)  # Variance should be positive


class TestFederatedIntegrator:
    def test_bayesian_weighting(self):
        # Case 1: Equal uncertainty
        hospitals = [
            {"local_risk": 0.5, "local_uncertainty": 0.1, "sample_size": 100},
            {"local_risk": 0.5, "local_uncertainty": 0.1, "sample_size": 100},
        ]
        integrator = FederatedIntegrator(hospitals)
        g_risk, g_unc = integrator.bayesian_weighting(hospitals)

        assert np.isclose(g_risk, 0.5)

        # Case 2: One very noisy source
        hospitals_noisy = [
            {
                "local_risk": 0.1,
                "local_uncertainty": 0.01,
                "sample_size": 1000,
            },  # Very sure about 0.1
            {
                "local_risk": 0.9,
                "local_uncertainty": 10.0,
                "sample_size": 10,
            },  # Very unsure about 0.9
        ]
        g_risk_n, _ = integrator.bayesian_weighting(hospitals_noisy)
        # result should be close to 0.1
        assert 0.09 < g_risk_n < 0.15


class TestBiasSimulation:
    def test_simulation_runs(self):
        results = biased_lending_simulation(n_samples=100)
        assert "sensitive_attr" in results
        assert "approved" in results
        assert len(results["approved"]) == 100
