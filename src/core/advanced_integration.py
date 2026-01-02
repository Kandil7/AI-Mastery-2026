import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Callable, Any

# ==========================================
# 1. Neural ODEs & Deep Integration
# ==========================================


class NeuralODE(nn.Module):
    """
    Neural Ordinary Differential Equation model.
    Defines the hidden state derivative: dh(t)/dt = f(h(t), t, theta)
    """

    def __init__(self, func: nn.Module):
        super().__init__()
        self.func = func

    def forward(
        self, z0: torch.Tensor, t: torch.Tensor, method: str = "euler"
    ) -> torch.Tensor:
        """
        Solves the ODE starting from z0 over time steps t.

        Args:
            z0: Initial state (batch_size, dim)
            t: Time points to evaluate
            method: Integration method ('euler' or 'rk4')
        """
        dt = t[1] - t[0]
        z = z0
        trajectory = [z]

        for i in range(len(t) - 1):
            if method == "euler":
                dz = self.func(t[i], z)
                z = z + dz * dt
            elif method == "rk4":
                k1 = self.func(t[i], z)
                k2 = self.func(t[i] + dt / 2, z + dt / 2 * k1)
                k3 = self.func(t[i] + dt / 2, z + dt / 2 * k2)
                k4 = self.func(t[i] + dt, z + dt * k3)
                z = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            trajectory.append(z)

        return torch.stack(trajectory)

    def integrate_with_uncertainty(
        self, x0: torch.Tensor, t_span: torch.Tensor, num_samples: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates trajectory with uncertainty using Monte Carlo Dropout (if present) or input noise.
        """
        trajectories = []
        is_training = self.training
        self.train()  # Enable dropout if present

        for _ in range(num_samples):
            # Add small noise to initial state to simulate measurement uncertainty
            noisy_x0 = x0 + torch.randn_like(x0) * 0.01
            traj = self.forward(noisy_x0, t_span)
            trajectories.append(traj.detach().cpu().numpy())

        self.train(is_training)

        trajectories = np.array(trajectories)
        mean_path = np.mean(trajectories, axis=0)
        std_path = np.std(trajectories, axis=0)

        return mean_path, std_path, trajectories


class ODEFunc(nn.Module):
    """Simple dynamics function for Neural ODE demo."""

    def __init__(self, dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, dim))

    def forward(self, t, x):
        return self.net(x)


# ==========================================
# 2. Multi-Modal Integration
# ==========================================


class MultiModalIntegrator(nn.Module):
    """
    Integrates data from multiple sources (clinical, image, text) with uncertainty.
    """

    def __init__(
        self, clinical_dim: int, xray_dim: int, text_dim: int, hidden_dim: int = 64
    ):
        super().__init__()

        # Encoders for each modality
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.xray_net = nn.Sequential(
            nn.Linear(xray_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.text_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Fusion layer with uncertainty output
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [mean_logit, log_variance]
        )

    def forward(self, clinical, xray, text):
        clinical_emb = self.clinical_net(clinical)
        xray_emb = self.xray_net(xray)
        text_emb = self.text_net(text)

        combined = torch.cat([clinical_emb, xray_emb, text_emb], dim=1)
        output = self.fusion_layer(combined)

        mean = torch.sigmoid(output[:, 0])
        uncertainty = torch.exp(output[:, 1])  # Predict variance

        return mean, uncertainty


# ==========================================
# 3. Federated Learning Integration
# ==========================================


class FederatedIntegrator:
    """
    Simulates federated learning integration using Bayesian weighting.
    """

    def __init__(self, hospitals: List[Any], global_model: nn.Module = None):
        self.hospitals = hospitals
        # Placeholder for global model logic
        self.global_model = global_model
        self.history = []

    def bayesian_weighting(self, estimates: List[Dict]) -> Tuple[float, float]:
        """
        Combines local estimates using Bayesian inverse-variance weighting.
        """
        risks = np.array([est["local_risk"] for est in estimates])
        uncertainties = np.array([est["local_uncertainty"] for est in estimates])
        sample_sizes = np.array([est["sample_size"] for est in estimates])

        # Weight = sample_size / variance
        # (More samples & lower variance = higher weight)
        weights = sample_sizes / (uncertainties**2 + 1e-8)
        weights = weights / np.sum(weights)

        global_risk = np.sum(weights * risks)

        # Approximation of global uncertainty
        global_uncertainty = 1.0 / np.sqrt(
            np.sum(sample_sizes / (uncertainties**2 + 1e-8))
        )

        return global_risk, global_uncertainty


# ==========================================
# 4. Ethics & Bias Simulation
# ==========================================

from scipy import stats


def biased_lending_simulation(n_samples=1000, bias_factor=0.3):
    """
    Simulates a lending decision system with integrated bias.
    """
    np.random.seed(42)

    age = np.random.normal(45, 15, n_samples)
    credit_score = np.random.normal(700, 100, n_samples)

    # Sensitive attribute (0 or 1)
    sensitive_attr = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    # Induce bias: Group 1 has slightly lower credit score on average
    credit_score[sensitive_attr == 1] -= bias_factor * 50

    # True creditworthiness (latent)
    true_worth = (credit_score - 300) / 550.0
    true_worth = np.clip(true_worth, 0, 1)

    # Biased Integration: System underestimates Group 1
    perceived_worth = true_worth.copy()
    perceived_worth[sensitive_attr == 1] *= 1.0 - bias_factor * 0.5

    # Decision
    approval = (perceived_worth > 0.6).astype(int)

    return {
        "sensitive_attr": sensitive_attr,
        "approved": approval,
        "true_worth": true_worth,
        "perceived_worth": perceived_worth,
    }
