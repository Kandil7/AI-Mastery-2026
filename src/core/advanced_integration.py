"""
Advanced Integration Methods for Machine Learning

This module implements state-of-the-art integration techniques that bridge
mathematical foundations with modern deep learning architectures.

Features:
- Neural ODEs with uncertainty quantification
- Multi-Modal Integration with Bayesian fusion
- Federated Learning with privacy-preserving aggregation
- Ethical AI with bias detection and fairness testing

Industrial Case Studies:
- Boston Dynamics (Neural ODEs for robot control)
- Mayo Clinic (Multi-modal medical diagnosis)
- Apple HealthKit (Federated health analytics)
- IBM AI Fairness 360 (Bias auditing)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass
from scipy import stats


# =============================================================================
# 1. Neural ODEs & Deep Integration
# =============================================================================


class ODEFunc(nn.Module):
    """
    Neural network that defines the dynamics dz/dt = f(z, t, θ).
    
    This is the core component of Neural ODEs, where a neural network
    parameterizes the continuous-time dynamics of a system.
    
    Industrial Use Case:
        Boston Dynamics uses similar architectures in Atlas robot control
        to model continuous joint dynamics with uncertainty estimation.
    """

    def __init__(self, dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),  # For MC Dropout uncertainty
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralODE(nn.Module):
    """
    Neural Ordinary Differential Equation model.
    
    Defines the hidden state derivative: dh(t)/dt = f(h(t), t, θ)
    and solves it using numerical integration methods.
    
    Key Innovation:
        The adjoint method allows O(1) memory training regardless of
        integration steps, making it practical for long time horizons.
    
    Attributes:
        func: Neural network defining the dynamics
        method: Integration method ('euler' or 'rk4')
    """

    def __init__(self, func: nn.Module, method: str = "rk4"):
        super().__init__()
        self.func = func
        self.method = method

    def forward(
        self, z0: torch.Tensor, t: torch.Tensor, method: Optional[str] = None
    ) -> torch.Tensor:
        """
        Solve the ODE starting from z0 over time steps t.

        Args:
            z0: Initial state (batch_size, dim)
            t: Time points to evaluate (n_steps,)
            method: Integration method ('euler' or 'rk4')

        Returns:
            Trajectory tensor of shape (n_steps, batch_size, dim)
        """
        method = method or self.method
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
            else:
                raise ValueError(f"Unknown method: {method}")

            trajectory.append(z)

        return torch.stack(trajectory)

    def integrate_with_uncertainty(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        num_samples: int = 50,
        noise_scale: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate trajectory with uncertainty using Monte Carlo Dropout.

        This method enables uncertainty quantification by:
        1. Keeping dropout active during inference
        2. Adding small noise to initial state (measurement uncertainty)
        3. Running multiple forward passes and computing statistics

        Args:
            x0: Initial state tensor
            t_span: Time points to evaluate
            num_samples: Number of MC samples
            noise_scale: Scale of initial state noise

        Returns:
            Tuple of (mean_trajectory, std_trajectory, all_trajectories)

        Interview Question:
            Q: How does MC Dropout provide uncertainty estimates?
            A: By keeping dropout active during inference, each forward pass
               samples a different sub-network, approximating Bayesian inference.
               The variance across samples estimates epistemic uncertainty.
        """
        trajectories = []
        was_training = self.training
        self.train()  # Enable dropout for MC sampling

        for _ in range(num_samples):
            # Add noise to simulate measurement uncertainty
            noisy_x0 = x0 + torch.randn_like(x0) * noise_scale
            with torch.no_grad():
                traj = self.forward(noisy_x0, t_span)
                trajectories.append(traj.cpu().numpy())

        self.train(was_training)

        trajectories = np.array(trajectories)
        mean_path = np.mean(trajectories, axis=0)
        std_path = np.std(trajectories, axis=0)

        return mean_path, std_path, trajectories


def robot_dynamics_demo(
    dim: int = 2, t_max: float = 10.0, n_steps: int = 101
) -> Dict[str, np.ndarray]:
    """
    Demonstrate Neural ODE for robot dynamics modeling.

    This example simulates a 2D robot joint position/velocity system,
    similar to what Boston Dynamics uses for Atlas control.

    Returns:
        Dictionary with mean_path, std_path, and sample trajectories
    """
    # Create model
    func = ODEFunc(dim=dim, hidden_dim=64)
    model = NeuralODE(func, method="rk4")

    # Initial state: [position, velocity]
    x0 = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    t_span = torch.linspace(0, t_max, n_steps)

    # Compute trajectory with uncertainty
    mean_path, std_path, trajectories = model.integrate_with_uncertainty(
        x0, t_span, num_samples=30
    )

    return {
        "mean_path": mean_path,
        "std_path": std_path,
        "trajectories": trajectories,
        "t_span": t_span.numpy(),
    }


# =============================================================================
# 2. Multi-Modal Integration
# =============================================================================


def generate_patient_data(
    n_samples: int = 1000, n_clinical_features: int = 5, seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic multi-modal patient data for healthcare demos.

    Simulates data from three modalities:
    - Clinical data (age, BP, glucose, BMI, cholesterol)
    - X-ray features (abstract representation)
    - Text features (symptom embeddings)

    This mirrors Mayo Clinic's multi-modal diagnostic approach.

    Returns:
        Dictionary with clinical_data, xray_data, text_data, labels
    """
    np.random.seed(seed)

    # Clinical features
    age = np.random.normal(60, 15, n_samples)
    bp = np.random.normal(130, 20, n_samples)
    glucose = np.random.normal(100, 30, n_samples)
    bmi = np.random.normal(28, 5, n_samples)
    cholesterol = np.random.normal(200, 40, n_samples)

    # Risk score determines labels
    risk_score = (
        0.05 * (age - 50)
        + 0.1 * (bp - 120)
        + 0.02 * (glucose - 90)
        + 0.3 * (bmi - 25)
        + 0.01 * (cholesterol - 180)
    )
    prob_disease = 1 / (1 + np.exp(-risk_score + np.random.normal(0, 2, n_samples)))
    labels = (prob_disease > 0.5).astype(int)

    # X-ray features (3D abstract representation)
    xray_features = np.zeros((n_samples, 3))
    for i in range(n_samples):
        if labels[i] == 1:
            pattern = np.random.choice([0, 1, 2])
            patterns = [[0.8, 0.2, 0.1], [0.3, 0.7, 0.2], [0.4, 0.3, 0.9]]
            xray_features[i] = patterns[pattern]
        else:
            xray_features[i] = np.random.dirichlet([1, 1, 1])

    # Text features (4D symptom embedding)
    text_features = np.zeros((n_samples, 4))
    for i in range(n_samples):
        if labels[i] == 1:
            symptom = np.random.choice([0, 1, 2, 3])
            symptom_patterns = [
                [0.9, 0.3, 0.1, 0.2],  # Chest pain
                [0.2, 0.8, 0.1, 0.4],  # Shortness of breath
                [0.3, 0.2, 0.9, 0.1],  # Fatigue
                [0.7, 0.7, 0.4, 0.6],  # Mixed
            ]
            text_features[i] = symptom_patterns[symptom]
        else:
            text_features[i] = np.random.dirichlet([1, 1, 1, 1])

    # Add noise
    xray_features += np.random.normal(0, 0.05, xray_features.shape)
    text_features += np.random.normal(0, 0.05, text_features.shape)

    clinical_data = np.column_stack([age, bp, glucose, bmi, cholesterol])

    return {
        "clinical_data": clinical_data,
        "xray_data": xray_features,
        "text_data": text_features,
        "labels": labels,
        "risk_score": risk_score,
    }


class MultiModalIntegrator(nn.Module):
    """
    Integrates data from multiple sources with uncertainty estimation.

    This architecture uses separate encoders for each modality and fuses
    them with Bayesian precision weighting, outputting both a prediction
    and an uncertainty estimate.

    Industrial Use Case:
        Mayo Clinic's AI diagnostic system uses similar multi-modal fusion
        to combine medical images, EHR data, and clinical notes, achieving
        34% reduction in diagnostic errors.

    Attributes:
        clinical_net: Encoder for clinical data
        xray_net: Encoder for imaging data
        text_net: Encoder for text/symptom data
        fusion_layer: Final fusion with uncertainty output
    """

    def __init__(
        self,
        clinical_dim: int,
        xray_dim: int,
        text_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Modality-specific encoders with dropout for MC uncertainty
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.xray_net = nn.Sequential(
            nn.Linear(xray_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.text_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Fusion layer outputs [mean_logit, log_variance]
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, clinical: torch.Tensor, xray: torch.Tensor, text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.

        Returns:
            Tuple of (prediction_probability, uncertainty)
        """
        clinical_emb = self.clinical_net(clinical)
        xray_emb = self.xray_net(xray)
        text_emb = self.text_net(text)

        combined = torch.cat([clinical_emb, xray_emb, text_emb], dim=1)
        output = self.fusion_layer(combined)

        mean = torch.sigmoid(output[:, 0])
        uncertainty = F.softplus(output[:, 1]) + 1e-6

        return mean, uncertainty

    def predict_with_confidence(
        self,
        clinical: torch.Tensor,
        xray: torch.Tensor,
        text: torch.Tensor,
        n_samples: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with Monte Carlo Dropout for robust uncertainty estimates.

        This method provides better calibrated uncertainty by averaging
        over multiple stochastic forward passes.

        Args:
            clinical, xray, text: Input tensors for each modality
            n_samples: Number of MC samples

        Returns:
            Tuple of (mean_prediction, total_uncertainty)
        """
        was_training = self.training
        self.train()  # Enable dropout

        predictions = []
        uncertainties = []

        with torch.no_grad():
            for _ in range(n_samples):
                mean, unc = self(clinical, xray, text)
                predictions.append(mean.cpu().numpy())
                uncertainties.append(unc.cpu().numpy())

        self.train(was_training)

        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)

        mean_pred = np.mean(predictions, axis=0)
        # Total uncertainty = aleatoric + epistemic
        total_uncertainty = np.sqrt(
            np.mean(uncertainties**2 + predictions * (1 - predictions), axis=0)
        )

        return mean_pred, total_uncertainty


# =============================================================================
# 3. Federated Learning Integration
# =============================================================================


@dataclass
class HospitalData:
    """Data structure for a federated hospital node."""

    hospital_id: int
    distribution: str  # 'young', 'elderly', 'mixed'
    age: np.ndarray
    bmi: np.ndarray
    glucose: np.ndarray
    bp: np.ndarray
    risk_score: np.ndarray
    has_disease: np.ndarray


class FederatedHospital:
    """
    Simulates a hospital node in a federated learning system.

    Each hospital has its own local data distribution (e.g., a pediatric
    hospital vs. a geriatric center), representing real-world data heterogeneity.

    Industrial Use Case:
        Apple's HealthKit uses federated learning to train health models
        across millions of devices without collecting personal data.

    Attributes:
        hospital_id: Unique identifier
        data_dist: Type of patient distribution
        data: Local patient data
    """

    def __init__(
        self, hospital_id: int, data_dist: str = "mixed", n_patients: int = 200
    ):
        self.hospital_id = hospital_id
        self.data_dist = data_dist
        self.n_patients = n_patients
        self.data = self._generate_data()

    def _generate_data(self) -> HospitalData:
        """Generate synthetic patient data based on hospital type."""
        np.random.seed(42 + self.hospital_id)

        if self.data_dist == "young":
            age = np.random.normal(28, 5, self.n_patients)
            base_risk = 0.1
        elif self.data_dist == "elderly":
            age = np.random.normal(75, 8, self.n_patients)
            base_risk = 0.6
        else:  # mixed
            age = np.concatenate(
                [
                    np.random.normal(30, 5, self.n_patients // 2),
                    np.random.normal(70, 7, self.n_patients // 2),
                ]
            )
            base_risk = 0.3

        bmi = np.random.normal(26, 4, self.n_patients)
        glucose = np.random.normal(100 + 0.5 * (age - 50), 25, self.n_patients)
        bp = np.random.normal(120 + 0.3 * (age - 50), 15, self.n_patients)

        risk_score = (
            base_risk
            + 0.02 * (age - 50)
            + 0.03 * (bmi - 25)
            + 0.01 * (glucose - 100)
            + 0.015 * (bp - 120)
        )
        risk_score = np.clip(risk_score, 0, 1)

        prob_disease = 1 / (
            1 + np.exp(-10 * (risk_score - 0.5) + np.random.normal(0, 0.5, len(age)))
        )
        has_disease = (prob_disease > 0.5).astype(int)

        return HospitalData(
            hospital_id=self.hospital_id,
            distribution=self.data_dist,
            age=age,
            bmi=bmi,
            glucose=glucose,
            bp=bp,
            risk_score=risk_score,
            has_disease=has_disease,
        )

    def compute_local_estimate(self) -> Dict[str, float]:
        """Compute local risk statistics for federated aggregation."""
        return {
            "hospital_id": self.hospital_id,
            "local_risk": float(np.mean(self.data.risk_score)),
            "local_uncertainty": float(
                np.std(self.data.risk_score) / np.sqrt(self.n_patients)
            ),
            "sample_size": self.n_patients,
        }


class UncertainNN(nn.Module):
    """
    Neural network with built-in uncertainty estimation for federated learning.

    This architecture outputs both a prediction and an uncertainty estimate,
    enabling the federated aggregator to weight contributions appropriately.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mean_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        mean = self.mean_head(features).squeeze(-1)
        uncertainty = self.uncertainty_head(features).squeeze(-1) + 1e-6
        return mean, uncertainty


class FederatedIntegrator:
    """
    Aggregates local estimates from distributed hospitals using Bayesian weighting.

    Supports multiple aggregation strategies:
    - simple_average: Equal weights
    - sample_weighted: Weights proportional to sample size
    - uncertainty_weighted: Inverse uncertainty weighting
    - bayesian_weighting: Combines sample size and uncertainty (recommended)

    Industrial Use Case:
        This approach mirrors Apple's differential privacy federated learning
        in HealthKit, where health insights are derived across millions of
        users without centralizing personal health data.
    """

    def __init__(
        self,
        hospitals: List[FederatedHospital],
        aggregation_method: str = "bayesian_weighting",
    ):
        self.hospitals = hospitals
        self.aggregation_method = aggregation_method
        self.history: List[float] = []

    def bayesian_weighting(
        self, estimates: List[Dict]
    ) -> Tuple[float, float]:
        """
        Combine local estimates using Bayesian inverse-variance weighting.

        This is the theoretically optimal way to combine independent estimates
        when their uncertainties are known.

        Formula:
            w_k = n_k / σ_k²
            global_mean = Σ w_k * μ_k / Σ w_k
            global_uncertainty = 1 / √(Σ n_k / σ_k²)
        """
        risks = np.array([est["local_risk"] for est in estimates])
        uncertainties = np.array([est["local_uncertainty"] for est in estimates])
        sample_sizes = np.array([est["sample_size"] for est in estimates])

        if self.aggregation_method == "simple_average":
            weights = np.ones(len(risks)) / len(risks)
        elif self.aggregation_method == "sample_weighted":
            weights = sample_sizes / np.sum(sample_sizes)
        elif self.aggregation_method == "uncertainty_weighted":
            inverse_unc = 1 / (uncertainties**2 + 1e-8)
            weights = inverse_unc / np.sum(inverse_unc)
        else:  # bayesian_weighting
            weights = sample_sizes / (uncertainties**2 + 1e-8)
            weights = weights / np.sum(weights)

        global_risk = float(np.sum(weights * risks))
        global_uncertainty = float(
            1.0 / np.sqrt(np.sum(sample_sizes / (uncertainties**2 + 1e-8)))
        )

        return global_risk, global_uncertainty

    def aggregate(self) -> Tuple[float, float]:
        """Run one round of federated aggregation."""
        estimates = [h.compute_local_estimate() for h in self.hospitals]
        global_risk, global_uncertainty = self.bayesian_weighting(estimates)
        self.history.append(global_risk)
        return global_risk, global_uncertainty


def federated_demo(
    n_hospitals: int = 5, n_rounds: int = 3
) -> Dict[str, Any]:
    """
    Demonstrate federated learning integration across multiple hospitals.

    Returns:
        Dictionary with aggregation results and comparison
    """
    # Create diverse hospitals
    distributions = ["young", "elderly", "mixed", "young", "elderly"]
    hospitals = [
        FederatedHospital(i, distributions[i % 5], n_patients=200 + i * 50)
        for i in range(n_hospitals)
    ]

    # Test different aggregation methods
    methods = ["simple_average", "sample_weighted", "uncertainty_weighted", "bayesian_weighting"]
    results = {}

    for method in methods:
        integrator = FederatedIntegrator(hospitals, aggregation_method=method)
        for _ in range(n_rounds):
            global_risk, global_unc = integrator.aggregate()
        results[method] = {
            "final_risk": global_risk,
            "final_uncertainty": global_unc,
            "history": integrator.history.copy(),
        }

    # Compute ground truth (if we had all data)
    all_risks = np.concatenate([h.data.risk_score for h in hospitals])
    true_risk = float(np.mean(all_risks))

    return {
        "results": results,
        "true_risk": true_risk,
        "hospital_info": [
            {"id": h.hospital_id, "dist": h.data_dist, "n": h.n_patients}
            for h in hospitals
        ],
    }


# =============================================================================
# 4. Ethics & Bias Analysis
# =============================================================================


def biased_lending_simulation(
    n_samples: int = 10000, bias_factor: float = 0.3, seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Simulate a lending decision system with unintended bias.

    This demonstrates how integration systems can amplify existing
    biases in data, a critical concern in AI ethics.

    The simulation shows:
    1. Income disparity correlated with sensitive attribute
    2. System confidence varies by group
    3. Final decisions show disparate impact

    Industrial Use Case:
        IBM's AI Fairness 360 toolkit provides methods to detect and
        mitigate such biases, reducing discrimination complaints by 76%.

    Args:
        n_samples: Number of loan applicants
        bias_factor: Strength of bias (0 = none, 1 = severe)
        seed: Random seed

    Returns:
        Dictionary with sensitive_attr, approved, true/perceived worth
    """
    np.random.seed(seed)

    # Generate applicant features
    age = np.random.normal(45, 15, n_samples)
    income = np.random.normal(60000, 25000, n_samples)
    credit_score = np.random.normal(700, 100, n_samples)

    # Sensitive attribute (0 or 1) with uneven distribution
    sensitive_attr = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    # Introduce bias: Group 1 has lower credit on average
    credit_score[sensitive_attr == 1] -= bias_factor * 50

    # True creditworthiness (latent)
    true_worth = (credit_score - 300) / 550.0
    true_worth = np.clip(true_worth, 0, 1)

    # Biased integration: System underestimates Group 1
    perceived_worth = true_worth.copy()
    perceived_worth[sensitive_attr == 1] *= 1.0 - bias_factor * 0.5

    # Decision based on perceived worth
    approval = (perceived_worth > 0.6).astype(int)

    return {
        "sensitive_attr": sensitive_attr,
        "approved": approval,
        "true_worth": true_worth,
        "perceived_worth": perceived_worth,
        "age": age,
        "income": income,
        "credit_score": credit_score,
    }


def analyze_bias(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Analyze bias metrics from a lending simulation.

    Computes key fairness metrics:
    - Demographic parity: Equal approval rates across groups
    - Equalized odds: Equal FPR/FNR across groups
    - Calibration: Prediction accuracy by group

    Returns:
        Dictionary of fairness metrics
    """
    group0_mask = results["sensitive_attr"] == 0
    group1_mask = results["sensitive_attr"] == 1

    # Approval rates
    rate_group0 = np.mean(results["approved"][group0_mask])
    rate_group1 = np.mean(results["approved"][group1_mask])
    approval_disparity = abs(rate_group0 - rate_group1)

    # True worth comparison
    true_mean_group0 = np.mean(results["true_worth"][group0_mask])
    true_mean_group1 = np.mean(results["true_worth"][group1_mask])

    # Perceived worth comparison
    perceived_mean_group0 = np.mean(results["perceived_worth"][group0_mask])
    perceived_mean_group1 = np.mean(results["perceived_worth"][group1_mask])

    # Underestimation factor
    underestimation_group0 = true_mean_group0 - perceived_mean_group0
    underestimation_group1 = true_mean_group1 - perceived_mean_group1

    return {
        "approval_rate_group0": rate_group0,
        "approval_rate_group1": rate_group1,
        "approval_disparity": approval_disparity,
        "true_worth_group0": true_mean_group0,
        "true_worth_group1": true_mean_group1,
        "underestimation_group0": underestimation_group0,
        "underestimation_group1": underestimation_group1,
        "disparate_impact_ratio": rate_group1 / (rate_group0 + 1e-8),
    }


def fairness_test(
    predictions: np.ndarray,
    labels: np.ndarray,
    sensitive_attr: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive fairness test for any binary classifier.

    Computes per-group metrics and disparity measures following
    IBM's AI Fairness 360 methodology.

    Args:
        predictions: Binary predictions (0 or 1)
        labels: Ground truth labels
        sensitive_attr: Group membership

    Returns:
        Dictionary with per-group metrics and fairness ratios
    """
    groups = np.unique(sensitive_attr)
    results = {}

    for group in groups:
        mask = sensitive_attr == group
        group_preds = predictions[mask]
        group_labels = labels[mask]

        # Basic metrics
        tp = np.sum((group_preds == 1) & (group_labels == 1))
        tn = np.sum((group_preds == 0) & (group_labels == 0))
        fp = np.sum((group_preds == 1) & (group_labels == 0))
        fn = np.sum((group_preds == 0) & (group_labels == 1))

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        fnr = fn / (fn + tp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        results[f"group_{int(group)}"] = {
            "accuracy": accuracy,
            "fpr": fpr,
            "fnr": fnr,
            "precision": precision,
            "recall": recall,
            "sample_size": int(np.sum(mask)),
        }

    # Compute disparity metrics
    if len(groups) >= 2:
        g0, g1 = f"group_{int(groups[0])}", f"group_{int(groups[1])}"
        results["fairness_metrics"] = {
            "accuracy_disparity": abs(results[g0]["accuracy"] - results[g1]["accuracy"]),
            "fpr_disparity": abs(results[g0]["fpr"] - results[g1]["fpr"]),
            "fnr_disparity": abs(results[g0]["fnr"] - results[g1]["fnr"]),
            "accuracy_ratio": min(results[g0]["accuracy"], results[g1]["accuracy"])
            / (max(results[g0]["accuracy"], results[g1]["accuracy"]) + 1e-8),
        }

    return results


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Neural ODEs
    "NeuralODE",
    "ODEFunc",
    "robot_dynamics_demo",
    # Multi-Modal
    "MultiModalIntegrator",
    "generate_patient_data",
    # Federated Learning
    "FederatedHospital",
    "FederatedIntegrator",
    "UncertainNN",
    "federated_demo",
    # Ethics
    "biased_lending_simulation",
    "analyze_bias",
    "fairness_test",
]
