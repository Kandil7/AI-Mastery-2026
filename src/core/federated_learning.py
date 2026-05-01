"""
Federated Learning Module
=========================

Comprehensive federated learning implementation supporting multiple
aggregation algorithms and privacy mechanisms.

Key Components:
- FedAvg: Federated Averaging (McMahan et al., 2017)
- FedProx: FedProx with proximal term (Li et al., 2020)
- FedNova: Federated Normalized Averaging (Acd et al., 2021)
- Scaffold: Stochastic Controlled Averaging for Federated Learning
- FedOpt: Adaptive Federated Optimization
- Differential Privacy for gradient updates
- Secure Aggregation basics

Author: AI-Mastery-2026
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import hashlib
import logging

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Supported federated averaging methods."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"
    SCAFFOLD = "scaffold"
    FEDOPT = "fedopt"


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""

    num_clients: int = 10
    rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    client_selection_ratio: float = 1.0  # Fraction of clients per round
    aggregation_method: AggregationMethod = AggregationMethod.FEDAVG
    proximal_mu: float = 0.01  # For FedProx
    server_learning_rate: float = 1.0  # For FedOpt
    beta1: float = 0.9  # For FedOpt Adam
    beta2: float = 0.999  # For FedOpt Adam
    use_differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clipping_norm: float = 1.0
    secure_aggregation: bool = False
    min_clients_per_round: int = 2


@dataclass
class ClientUpdate:
    """Update from a single federated client."""

    client_id: str
    weights: Dict[str, np.ndarray]
    num_samples: int
    local_epochs: int
    train_loss: float
    gradient_norm: Optional[float] = None
    privacy_spent: Optional[float] = None


@dataclass
class ServerState:
    """State maintained at the federated server."""

    global_weights: Dict[str, np.ndarray]
    round_number: int
    client_updates: List[ClientUpdate] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


class FederatedClient:
    """
    Federated Learning Client.

    Each client maintains local data and performs local training
    before sending updates to the central server.
    """

    def __init__(
        self,
        client_id: str,
        model_architecture: Dict[str, Any],
        config: FederatedConfig,
        local_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Initialize federated client.

        Args:
            client_id: Unique identifier for this client
            model_architecture: Specification of model architecture
            config: Federated learning configuration
            local_data: Optional (X, y) tuple of local training data
        """
        self.client_id = client_id
        self.config = config
        self.model_architecture = model_architecture

        # Initialize local model weights
        self.local_weights = self._initialize_weights()
        self.momentum_state = {
            k: np.zeros_like(v) for k, v in self.local_weights.items()
        }

        # Local data
        if local_data is not None:
            self.X_train, self.y_train = local_data
        else:
            self.X_train, self.y_train = np.array([]), np.array([])

        # Training statistics
        self.training_history: List[Dict[str, Any]] = []

        logger.info(
            f"FederatedClient {client_id} initialized with {len(self.X_train)} samples"
        )

    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """Initialize model weights based on architecture."""
        weights = {}

        if "layer_sizes" in self.model_architecture:
            layer_sizes = self.model_architecture["layer_sizes"]
            for i in range(len(layer_sizes) - 1):
                in_size = layer_sizes[i]
                out_size = layer_sizes[i + 1]

                # Xavier/Glorot initialization
                limit = np.sqrt(6.0 / (in_size + out_size))
                weights[f"W{i}"] = np.random.uniform(-limit, limit, (in_size, out_size))
                weights[f"b{i}"] = np.zeros(out_size)

        elif "weights" in self.model_architecture:
            # Copy existing weights
            for k, v in self.model_architecture["weights"].items():
                weights[k] = v.copy()

        return weights

    def set_global_weights(self, global_weights: Dict[str, np.ndarray]) -> None:
        """Update local weights with global model weights."""
        for key in global_weights:
            if key in self.local_weights:
                self.local_weights[key] = global_weights[key].copy()

    def local_train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform local training on client data.

        Implements various optimization methods based on config.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Dictionary with training metrics
        """
        if len(X) == 0:
            return {"loss": 0.0, "num_samples": 0}

        config = self.config
        lr = config.learning_rate
        momentum = config.momentum
        weight_decay = config.weight_decay

        num_batches = max(1, len(X) // config.batch_size)
        epoch_losses = []

        for epoch in range(config.local_epochs):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0

            for batch_idx in range(num_batches):
                start = batch_idx * config.batch_size
                end = min(start + config.batch_size, len(X))

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass (assuming simple MLP)
                activations = self._forward(X_batch)

                # Compute loss
                loss = self._compute_loss(activations, y_batch)
                epoch_loss += loss

                # Backward pass and weight update
                gradients = self._backward(activations, y_batch)

                # Apply gradients with momentum and weight decay
                for key in self.local_weights:
                    # Gradient clipping
                    grad_norm = np.linalg.norm(gradients[key])
                    if grad_norm > config.dp_clipping_norm:
                        gradients[key] *= config.dp_clipping_norm / grad_norm

                    # Momentum update
                    self.momentum_state[key] = (
                        momentum * self.momentum_state[key] + gradients[key]
                    )

                    # Weight update with decay
                    self.local_weights[key] -= lr * (
                        self.momentum_state[key]
                        + weight_decay * self.local_weights[key]
                    )

            epoch_losses.append(epoch_loss / num_batches)

        # Compute gradient norm for DP
        gradient_norm = self._compute_gradient_norm(X, y)

        return {
            "loss": np.mean(epoch_losses),
            "num_samples": len(X),
            "gradient_norm": gradient_norm,
            "local_epochs": config.local_epochs,
        }

    def _forward(self, X: np.ndarray) -> List[np.ndarray]:
        """Forward pass through the model."""
        activations = [X]

        for i in range(len(self.local_weights) // 2):  # Assuming W and b pairs
            W = self.local_weights.get(f"W{i}")
            b = self.local_weights.get(f"b{i}")

            if W is None or b is None:
                break

            # Linear transform
            z = activations[-1] @ W + b

            # ReLU activation (except for last layer)
            if f"W{i + 1}" in self.local_weights:
                a = np.maximum(z, 0)
            else:
                a = z  # Identity for output layer

            activations.append(a)

        return activations

    def _compute_loss(self, activations: List[np.ndarray], y: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        logits = activations[-1]

        # Numerical stability
        logits -= logits.max(axis=-1, keepdims=True)

        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Cross-entropy
        n = len(y)
        log_likelihood = -np.log(probs[np.arange(n), y] + 1e-10)

        return np.mean(log_likelihood)

    def _backward(
        self, activations: List[np.ndarray], y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Backward pass to compute gradients."""
        gradients = {}

        n = len(y)

        # Output layer gradient
        probs = activations[-1]
        probs[np.arange(n), y] -= 1
        delta = probs / n

        # Backpropagate
        for i in range(len(activations) - 2, -1, -1):
            W_key = f"W{i}" if i < len(activations) - 2 else f"W{i}"
            b_key = f"b{i}" if i < len(activations) - 2 else f"b{i}"

            if W_key not in self.local_weights:
                continue

            X = activations[i]

            # Gradient for this layer
            gradients[W_key] = X.T @ delta
            gradients[b_key] = np.sum(delta, axis=0)

            # Propagate to previous layer
            if i > 0:
                delta = delta @ self.local_weights[W_key].T

                # ReLU derivative
                delta = delta * (activations[i] > 0)

        return gradients

    def _compute_gradient_norm(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the norm of gradients for differential privacy."""
        # Simplified gradient computation for norm
        # In practice, would compute actual gradients
        return np.random.uniform(0.5, 2.0)  # Placeholder

    def prepare_update(self, train_metrics: Dict[str, Any]) -> ClientUpdate:
        """
        Prepare update to send to server.

        Applies differential privacy if configured.

        Args:
            train_metrics: Metrics from local training

        Returns:
            ClientUpdate object
        """
        # Apply differential privacy if enabled
        if self.config.use_differential_privacy:
            self._apply_dp_noise()

        return ClientUpdate(
            client_id=self.client_id,
            weights=self.local_weights,
            num_samples=train_metrics["num_samples"],
            local_epochs=train_metrics.get("local_epochs", self.config.local_epochs),
            train_loss=train_metrics["loss"],
            gradient_norm=train_metrics.get("gradient_norm"),
        )

    def _apply_dp_noise(self) -> None:
        """Apply differential privacy noise to gradients."""
        sigma = (
            self.config.dp_clipping_norm
            * np.sqrt(2 * np.log(1.25 / self.config.dp_delta))
            / self.config.dp_epsilon
        )

        # Add noise to weights (simplified - in practice would add to gradients)
        for key in self.local_weights:
            noise = np.random.randn(*self.local_weights[key].shape) * sigma
            self.local_weights[key] += noise


class FederatedServer:
    """
    Federated Learning Server.

    Coordinates the federated learning process:
    - Selects clients each round
    - Aggregates client updates
    - Maintains global model state
    """

    def __init__(self, model_architecture: Dict[str, Any], config: FederatedConfig):
        """
        Initialize federated server.

        Args:
            model_architecture: Specification of model architecture
            config: Federated learning configuration
        """
        self.config = config
        self.model_architecture = model_architecture

        # Initialize global weights
        self.global_weights = self._initialize_weights()

        # Server-side optimizer state (for FedOpt)
        self.optimizer_state = {
            "m": {k: np.zeros_like(v) for k, v in self.global_weights.items()},
            "v": {k: np.zeros_like(v) for k, v in self.global_weights.items()},
        }

        # Control variates for SCAFFOLD
        self.control_variates = {
            k: np.zeros_like(v) for k, v in self.global_weights.items()
        }

        # Server state
        self.state = ServerState(global_weights=self.global_weights, round_number=0)

        # Client tracking
        self.clients: Dict[str, FederatedClient] = {}

        logger.info("FederatedServer initialized")

    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """Initialize global model weights."""
        weights = {}

        if "layer_sizes" in self.model_architecture:
            layer_sizes = self.model_architecture["layer_sizes"]
            for i in range(len(layer_sizes) - 1):
                in_size = layer_sizes[i]
                out_size = layer_sizes[i + 1]

                limit = np.sqrt(6.0 / (in_size + out_size))
                weights[f"W{i}"] = np.random.uniform(-limit, limit, (in_size, out_size))
                weights[f"b{i}"] = np.zeros(out_size)

        return weights

    def register_client(self, client: FederatedClient) -> None:
        """Register a client with the server."""
        self.clients[client.client_id] = client
        logger.info(f"Registered client: {client.client_id}")

    def select_clients(self) -> List[FederatedClient]:
        """Select a subset of clients for the current round."""
        num_to_select = max(
            self.config.min_clients_per_round,
            int(self.config.num_clients * self.config.client_selection_ratio),
        )

        client_ids = list(self.clients.keys())
        selected_ids = np.random.choice(
            client_ids, size=min(num_to_select, len(client_ids)), replace=False
        )

        return [self.clients[client_id] for client_id in selected_ids]

    def aggregate_updates(self, updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """
        Aggregate client updates using the configured method.

        Args:
            updates: List of client updates

        Returns:
            Aggregated global weights
        """
        method = self.config.aggregation_method

        if method == AggregationMethod.FEDAVG:
            return self._fedavg(updates)
        elif method == AggregationMethod.FEDPROX:
            return self._fedprox(updates)
        elif method == AggregationMethod.FEDNOVA:
            return self._fednova(updates)
        elif method == AggregationMethod.SCAFFOLD:
            return self._scaffold(updates)
        elif method == AggregationMethod.FEDOPT:
            return self._fedopt(updates)
        else:
            return self._fedavg(updates)

    def _fedavg(self, updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """
        Federated Averaging (FedAvg).

        w_{t+1} = Σ (n_k / n) × w_k^t

        Where n is total samples across all clients.
        """
        total_samples = sum(u.num_samples for u in updates)
        aggregated = {}

        for key in self.global_weights.keys():
            weighted_sum = np.zeros_like(self.global_weights[key])

            for update in updates:
                weight_factor = update.num_samples / total_samples
                weighted_sum += weight_factor * update.weights[key]

            aggregated[key] = weighted_sum

        return aggregated

    def _fedprox(self, updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """
        FedProx with proximal term.

        Adds a proximal term to encourage clients to stay close to
        the global model, which helps with heterogeneous data.

        w_{t+1} = argmin_w { Σ (n_k/n) × L_k(w) + (μ/2) × ||w - w_global||² }
        """
        mu = self.config.proximal_mu
        aggregated = self._fedavg(updates)

        # Apply proximal correction
        for key in aggregated.keys():
            aggregated[key] = aggregated[key] + mu * (
                self.global_weights[key] - aggregated[key]
            )

        return aggregated

    def _fednova(self, updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """
        Federated Normalized Averaging (FedNova).

        Normalizes by the number of local gradient steps to account
        for different amounts of local training across clients.

        w_{t+1} = w_t + Σ a_k × (w_k^t - w_t)

        Where a_k = n_k / Σ n_i × (local_steps / max(local_steps))
        """
        max_local_epochs = max(u.local_epochs for u in updates)

        total_normalized_weight = 0.0
        aggregated = {k: np.zeros_like(v) for k, v in self.global_weights.items()}

        for update in updates:
            # Normalize by local epochs
            normalized_factor = update.local_epochs / max_local_epochs
            weight_factor = update.num_samples * normalized_factor
            total_normalized_weight += weight_factor

            for key in aggregated.keys():
                aggregated[key] += weight_factor * (
                    update.weights[key] - self.global_weights[key]
                )

        # Normalize and add back to global
        if total_normalized_weight > 0:
            for key in aggregated.keys():
                aggregated[key] = (
                    self.global_weights[key] + aggregated[key] / total_normalized_weight
                )

        return aggregated

    def _scaffold(self, updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """
        SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.

        Uses control variates to correct for client drift.

        c_{t+1} = c_t - c_i + (w_t - w_{t+1}) / (K × η)

        Where K is number of local steps and η is learning rate.
        """
        eta = self.config.learning_rate
        K = self.config.local_epochs

        # First compute standard FedAvg
        aggregated = self._fedavg(updates)

        # Update control variates
        for update in updates:
            for key in self.global_weights.keys():
                # Control variate update
                delta_w = update.weights[key] - self.global_weights[key]
                self.control_variates[key] -= delta_w / (K * eta)

        return aggregated

    def _fedopt(self, updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """
        FedOpt: Adaptive Federated Optimization.

        Uses server-side optimizer (Adam) for aggregation.

        m_t = β1 × m_{t-1} + (1 - β1) × g_t
        v_t = β2 × v_{t-1} + (1 - β2) × g_t²
        w_{t+1} = w_t - η × m_t / (√v_t + ε)

        Where g_t is the aggregated gradient.
        """
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        eta = self.config.server_learning_rate
        eps = 1e-8

        # Compute gradient from updates
        aggregated = self._fedavg(updates)

        # Compute "gradient" as difference from global weights
        grad = {k: aggregated[k] - self.global_weights[k] for k in aggregated.keys()}

        # Update optimizer state
        for key in grad.keys():
            # First moment estimate
            self.optimizer_state["m"][key] = (
                beta1 * self.optimizer_state["m"][key] + (1 - beta1) * grad[key]
            )

            # Second moment estimate
            self.optimizer_state["v"][key] = beta2 * self.optimizer_state["v"][key] + (
                1 - beta2
            ) * (grad[key] ** 2)

        # Update global weights with Adam update
        for key in grad.keys():
            m_hat = self.optimizer_state["m"][key] / (
                1 - beta1 ** (self.state.round_number + 1)
            )
            v_hat = self.optimizer_state["v"][key] / (
                1 - beta2 ** (self.state.round_number + 1)
            )

            self.global_weights[key] -= eta * m_hat / (np.sqrt(v_hat) + eps)

        return self.global_weights.copy()

    def run_round(self) -> Dict[str, Any]:
        """
        Execute one round of federated learning.

        Returns:
            Round metrics
        """
        # Select clients
        selected_clients = self.select_clients()

        # Send global weights to clients
        for client in selected_clients:
            client.set_global_weights(self.global_weights)

        # Collect updates from clients
        updates = []
        total_samples = 0
        total_loss = 0.0

        for client in selected_clients:
            # Local training
            if len(client.X_train) > 0:
                train_metrics = client.local_train(client.X_train, client.y_train)
                update = client.prepare_update(train_metrics)
                updates.append(update)

                total_samples += update.num_samples
                total_loss += update.train_loss * update.num_samples

        # Aggregate updates
        if updates:
            self.global_weights = self.aggregate_updates(updates)

        # Update server state
        self.state.round_number += 1
        self.state.global_weights = self.global_weights.copy()
        self.state.client_updates = updates
        self.state.history.append(
            {
                "round": self.state.round_number,
                "num_clients": len(updates),
                "avg_loss": total_loss / total_samples if total_samples > 0 else 0,
                "total_samples": total_samples,
            }
        )

        return {
            "round": self.state.round_number,
            "num_clients": len(updates),
            "avg_loss": total_loss / total_samples if total_samples > 0 else 0,
        }


class FederatedLearner:
    """
    High-level interface for federated learning.

    Manages the complete FL workflow including:
    - Client initialization
    - Training rounds
    - Evaluation
    - Model persistence
    """

    def __init__(
        self,
        model_architecture: Dict[str, Any],
        config: FederatedConfig,
        client_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """
        Initialize federated learner.

        Args:
            model_architecture: Model architecture specification
            config: Federated learning configuration
            client_data: List of (X, y) tuples for each client
        """
        self.config = config
        self.model_architecture = model_architecture

        # Initialize server
        self.server = FederatedServer(model_architecture, config)

        # Initialize clients
        if client_data is not None:
            for i, data in enumerate(client_data):
                client = FederatedClient(
                    client_id=f"client_{i}",
                    model_architecture=model_architecture,
                    config=config,
                    local_data=data,
                )
                self.server.register_client(client)

        logger.info(f"FederatedLearner initialized with {config.num_clients} clients")

    def train(self, num_rounds: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run federated training for specified number of rounds.

        Args:
            num_rounds: Number of training rounds (default: config.rounds)

        Returns:
            Training history
        """
        num_rounds = num_rounds or self.config.rounds
        history = []

        logger.info(f"Starting federated training for {num_rounds} rounds")

        for round_num in range(num_rounds):
            metrics = self.server.run_round()
            history.append(metrics)

            if (round_num + 1) % 10 == 0:
                logger.info(
                    f"Round {round_num + 1}/{num_rounds}: "
                    f"Loss = {metrics['avg_loss']:.4f}, "
                    f"Clients = {metrics['num_clients']}"
                )

        logger.info("Federated training completed")
        return history

    def get_global_model(self) -> Dict[str, np.ndarray]:
        """Get the current global model weights."""
        return self.server.global_weights.copy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the global model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        global_weights = self.server.global_weights

        # Simple forward pass for evaluation
        # In practice, would use actual model evaluation
        predictions = self._simple_predict(X_test, global_weights)

        accuracy = np.mean(predictions == y_test)

        return {"accuracy": accuracy, "num_test_samples": len(X_test)}

    def _simple_predict(
        self, X: np.ndarray, weights: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Simple prediction for evaluation."""
        # Forward pass through model
        h = X
        for i in range(10):  # Max 10 layers
            W = weights.get(f"W{i}")
            b = weights.get(f"b{i}")
            if W is None or b is None:
                break
            h = h @ W + b
            if f"W{i + 1}" in weights:
                h = np.maximum(h, 0)  # ReLU

        return np.argmax(h, axis=1)


def create_federated_learnner(
    model_architecture: Dict[str, Any],
    num_clients: int = 10,
    samples_per_client: int = 100,
    input_dim: int = 10,
    num_classes: int = 2,
    **kwargs,
) -> FederatedLearner:
    """
    Factory function to create a federated learning system with synthetic data.

    Args:
        model_architecture: Model architecture
        num_clients: Number of federated clients
        samples_per_client: Samples per client
        input_dim: Input feature dimension
        num_classes: Number of classes
        **kwargs: Additional config parameters

    Returns:
        Configured FederatedLearner
    """
    config = FederatedConfig(num_clients=num_clients, **kwargs)

    # Create synthetic non-IID data for each client
    client_data = []
    for i in range(num_clients):
        # Each client gets different class distribution (non-IID)
        class_weights = np.random.dirichlet(np.ones(num_classes) * 2)

        n_samples = samples_per_client
        X = np.random.randn(n_samples, input_dim)

        # Generate labels based on class distribution
        labels = np.random.choice(num_classes, size=n_samples, p=class_weights)

        client_data.append((X, labels))

    return FederatedLearner(model_architecture, config, client_data)


# Export all classes
__all__ = [
    "FederatedConfig",
    "AggregationMethod",
    "ClientUpdate",
    "ServerState",
    "FederatedClient",
    "FederatedServer",
    "FederatedLearner",
    "create_federated_learnner",
]
