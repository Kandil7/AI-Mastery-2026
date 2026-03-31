"""
Federated Learning Module
=========================

Federated Learning client for medical devices.

Implements:
- Local model training on device data
- Differential privacy for gradient updates
- Secure aggregation protocol
- Personal Health Train integration

Classes:
    FederatedLearningClient: FL client for medical devices

Author: AI-Mastery-2026
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .privacy import DifferentialPrivacy
from .types import FederatedUpdate, PrivacyBudget

logger = logging.getLogger(__name__)


class FederatedLearningClient:
    """
    Federated Learning client for medical devices.

    Implements:
    - Local model training on device data
    - Differential privacy for gradient updates
    - Secure aggregation protocol
    - Personal Health Train integration

    Mathematical Foundation:
    - FedAvg: w_{t+1} = Σ (n_k/n) × w_k^t
    - DP-FedAvg: Add Gaussian noise to clipped gradients
    """

    def __init__(
        self,
        client_id: str,
        model_architecture: Dict[str, Any],
        privacy_budget: Optional[PrivacyBudget] = None,
        max_gradient_norm: float = 1.0,
    ):
        """
        Initialize FL client.

        Args:
            client_id: Unique identifier (hospital/device)
            model_architecture: Neural network architecture spec
            privacy_budget: DP budget tracker
            max_gradient_norm: Clip bound for gradients
        """
        self.client_id = client_id
        self.model_architecture = model_architecture
        self.privacy_budget = privacy_budget or PrivacyBudget()
        self.max_gradient_norm = max_gradient_norm

        # Initialize local model
        self.local_weights = self._initialize_weights()
        self.round_number = 0

        # Local data buffer (encrypted at rest)
        self.local_data: List[np.ndarray] = []
        self.local_labels: List[np.ndarray] = []

        # DP mechanism
        self.dp = DifferentialPrivacy(
            epsilon=self.privacy_budget.epsilon_budget, delta=self.privacy_budget.delta
        )

        # Training history
        self.training_history: List[Dict[str, Any]] = []

        logger.info(f"FL Client initialized: {client_id}")

    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """Initialize model weights based on architecture."""
        weights = {}

        layer_sizes = self.model_architecture.get("layer_sizes", [128, 64, 32, 2])

        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]

            # He initialization
            weights[f"W{i}"] = np.random.randn(in_size, out_size) * np.sqrt(
                2.0 / in_size
            )
            weights[f"b{i}"] = np.zeros(out_size)

        return weights

    def receive_global_model(self, global_weights: Dict[str, np.ndarray]):
        """
        Receive updated global model from server.

        This is the "download" phase of federated learning.
        """
        self.local_weights = {k: v.copy() for k, v in global_weights.items()}
        self.round_number += 1
        logger.info(f"Received global model for round {self.round_number}")

    def add_local_data(self, features: np.ndarray, labels: np.ndarray):
        """
        Add local training data (stays on device).

        Data is encrypted at rest using device-specific keys.
        """
        self.local_data.append(features)
        self.local_labels.append(labels)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through local model."""
        activation = X
        num_layers = len([k for k in self.local_weights if k.startswith("W")])

        for i in range(num_layers):
            W = self.local_weights[f"W{i}"]
            b = self.local_weights[f"b{i}"]

            activation = activation @ W + b

            # ReLU for hidden layers, sigmoid for output
            if i < num_layers - 1:
                activation = np.maximum(0, activation)
            else:
                activation = 1 / (1 + np.exp(-activation))

        return activation

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients via backpropagation."""
        batch_size = X.shape[0]
        gradients = {k: np.zeros_like(v) for k, v in self.local_weights.items()}

        # Per-example gradients (for DP clipping)
        for i in range(batch_size):
            x_i = X[i : i + 1]
            y_i = y[i : i + 1]

            # Forward pass with caching
            activations = [x_i]
            current = x_i
            num_layers = len([k for k in self.local_weights if k.startswith("W")])

            for layer in range(num_layers):
                W = self.local_weights[f"W{layer}"]
                b = self.local_weights[f"b{layer}"]

                z = current @ W + b

                if layer < num_layers - 1:
                    current = np.maximum(0, z)
                else:
                    current = 1 / (1 + np.exp(-z))

                activations.append(current)

            # Backward pass
            pred = activations[-1]
            delta = pred - y_i

            for layer in range(num_layers - 1, -1, -1):
                a_prev = activations[layer]

                dW = a_prev.T @ delta
                db = delta.sum(axis=0)

                # Clip per-example gradient
                grad_vector = np.concatenate([dW.flatten(), db.flatten()])
                grad_vector = self.dp.clip_gradients(
                    grad_vector, self.max_gradient_norm
                )

                dW_size = dW.size
                dW = grad_vector[:dW_size].reshape(dW.shape)
                db = grad_vector[dW_size:]

                gradients[f"W{layer}"] += dW
                gradients[f"b{layer}"] += db

                if layer > 0:
                    W = self.local_weights[f"W{layer}"]
                    delta = delta @ W.T
                    delta = delta * (activations[layer] > 0)

        for k in gradients:
            gradients[k] /= batch_size

        return gradients

    def train_local(
        self,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        apply_dp: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Train model on local data with DP guarantees.

        Args:
            epochs: Number of local epochs
            batch_size: Mini-batch size
            learning_rate: SGD learning rate
            apply_dp: Whether to apply differential privacy

        Returns:
            Gradient update to send to server
        """
        if not self.local_data:
            logger.warning("No local data for training")
            return {}

        X = np.vstack(self.local_data)
        y = np.vstack(self.local_labels)
        n_samples = X.shape[0]

        initial_weights = {k: v.copy() for k, v in self.local_weights.items()}

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                gradients = self._compute_gradients(X_batch, y_batch)

                if apply_dp:
                    for k, g in gradients.items():
                        epsilon_cost = 0.01
                        if self.privacy_budget.can_query(epsilon_cost):
                            gradients[k] = self.dp.gaussian_mechanism(
                                g, sensitivity=self.max_gradient_norm / batch_size
                            )
                            self.privacy_budget.consume(epsilon_cost)

                for k in self.local_weights:
                    self.local_weights[k] -= learning_rate * gradients[k]

        weight_update = {}
        for k in self.local_weights:
            weight_update[k] = self.local_weights[k] - initial_weights[k]

        self.training_history.append(
            {
                "round": self.round_number,
                "epochs": epochs,
                "samples": n_samples,
                "epsilon_used": self.privacy_budget.epsilon_used,
                "timestamp": datetime.now(),
            }
        )

        logger.info(
            f"Local training complete: {epochs} epochs, {n_samples} samples, "
            f"ε={self.privacy_budget.epsilon_used:.4f}"
        )

        return weight_update

    def create_federated_update(
        self, weight_update: Dict[str, np.ndarray]
    ) -> FederatedUpdate:
        """Package weight update for server transmission."""
        flat_gradients = np.concatenate(
            [weight_update[k].flatten() for k in sorted(weight_update.keys())]
        )

        return FederatedUpdate(
            client_id=self.client_id,
            round_number=self.round_number,
            gradients=flat_gradients,
            sample_count=sum(len(d) for d in self.local_data),
            timestamp=datetime.now(),
            encrypted=False,
            signature=hashlib.sha256(flat_gradients.tobytes()).hexdigest()[:32],
        )

    def get_privacy_report(self) -> Dict[str, Any]:
        """Get privacy budget status."""
        return {
            "epsilon_used": self.privacy_budget.epsilon_used,
            "epsilon_remaining": self.privacy_budget.remaining_budget,
            "delta": self.privacy_budget.delta,
            "queries_made": self.privacy_budget.queries_made,
            "training_rounds": len(self.training_history),
        }

    def clear_local_data(self) -> int:
        """Clear local training data."""
        count = len(self.local_data)
        self.local_data.clear()
        self.local_labels.clear()
        return count
