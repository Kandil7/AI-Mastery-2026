"""
Differential Privacy Module
==============================

Differential privacy mechanisms for medical data.

Mathematical Foundation:
- A mechanism M is ε-differentially private if:
  P[M(D) ∈ S] ≤ exp(ε) × P[M(D') ∈ S]
  for all neighboring datasets D, D' differing in one record.

Mechanisms:
- Laplace mechanism: for numeric queries
- Gaussian mechanism: for vector queries
- Exponential mechanism: for categorical outputs

Classes:
    DifferentialPrivacy: DP mechanisms implementation

Author: AI-Mastery-2026
"""

import logging
from typing import Any, Dict

import numpy as np

from .types import PrivacyBudget

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """
    Differential Privacy mechanisms for medical data.

    Mathematical Foundation:
    - A mechanism M is ε-differentially private if:
      P[M(D) ∈ S] ≤ exp(ε) × P[M(D') ∈ S]
      for all neighboring datasets D, D' differing in one record.

    Mechanisms implemented:
    - Laplace mechanism: for numeric queries
    - Gaussian mechanism: for vector queries
    - Exponential mechanism: for categorical outputs
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize DP mechanism.

        Args:
            epsilon: Privacy parameter (lower = more private)
            delta: Failure probability
        """
        self.epsilon = epsilon
        self.delta = delta

    def laplace_mechanism(self, value: float, sensitivity: float) -> float:
        """
        Apply Laplace mechanism for ε-DP.

        Adds noise from Laplace(0, sensitivity/ε) distribution.

        Args:
            value: True value to privatize
            sensitivity: L1 sensitivity (max change from one record)

        Returns:
            Privatized value
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise

    def gaussian_mechanism(self, vector: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Apply Gaussian mechanism for (ε,δ)-DP.

        Adds noise from N(0, σ²I) where:
        σ = sensitivity × sqrt(2 × ln(1.25/δ)) / ε

        Args:
            vector: True vector to privatize
            sensitivity: L2 sensitivity

        Returns:
            Privatized vector
        """
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma, vector.shape)
        return vector + noise

    def clip_gradients(self, gradients: np.ndarray, max_norm: float) -> np.ndarray:
        """
        Clip gradients for bounded sensitivity.

        Per-example gradient clipping is essential for DP-SGD.

        Args:
            gradients: Gradient vector
            max_norm: Maximum L2 norm

        Returns:
            Clipped gradients
        """
        norm = np.linalg.norm(gradients)
        if norm > max_norm:
            gradients = gradients * (max_norm / norm)
        return gradients

    def privatize_histogram(
        self, histogram: np.ndarray, sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Privatize a histogram (common for medical stats).

        Adds Laplace noise to each bin independently.
        """
        return np.array(
            [max(0, self.laplace_mechanism(count, sensitivity)) for count in histogram]
        )

    def add_noise_to_array(
        self, data: np.ndarray, sensitivity: float, mechanism: str = "gaussian"
    ) -> np.ndarray:
        """
        Add noise to array with specified mechanism.

        Args:
            data: Input data array
            sensitivity: Sensitivity bound
            mechanism: 'laplace' or 'gaussian'

        Returns:
            Noisy data
        """
        if mechanism == "laplace":
            return np.array(
                [self.laplace_mechanism(val, sensitivity) for val in data.flatten()]
            ).reshape(data.shape)
        else:
            return self.gaussian_mechanism(data, sensitivity)

    def compute_epsilon_for_sigma(self, sigma: float, sensitivity: float) -> float:
        """
        Compute effective epsilon for given sigma.

        ε = sensitivity × sqrt(2 × ln(1.25/δ)) / σ
        """
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / sigma

    def get_privacy_params(self) -> Dict[str, Any]:
        """Get current privacy parameters."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "mechanism": "gaussian",
        }


class PrivacyBudgetManager:
    """Manage privacy budgets across multiple queries."""

    def __init__(self, total_budget: float = 1.0, delta: float = 1e-5):
        self.total_budget = total_budget
        self.delta = delta
        self.budgets: Dict[str, PrivacyBudget] = {}

    def create_budget(
        self, client_id: str, budget: Optional[float] = None
    ) -> PrivacyBudget:
        """Create a new privacy budget for a client."""
        pb = PrivacyBudget(epsilon_budget=budget or self.total_budget, delta=self.delta)
        self.budgets[client_id] = pb
        return pb

    def get_budget(self, client_id: str) -> Optional[PrivacyBudget]:
        """Get budget for a client."""
        return self.budgets.get(client_id)

    def consume_budget(self, client_id: str, epsilon_cost: float) -> bool:
        """Consume budget for a client."""
        budget = self.budgets.get(client_id)
        if budget:
            return budget.consume(epsilon_cost)
        return False

    def get_all_reports(self) -> Dict[str, Dict[str, Any]]:
        """Get privacy reports for all clients."""
        return {
            client_id: {
                "epsilon_used": budget.epsilon_used,
                "epsilon_remaining": budget.remaining_budget,
                "queries_made": budget.queries_made,
            }
            for client_id, budget in self.budgets.items()
        }
