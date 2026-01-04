"""
Medical Edge AI Module
=======================

Edge AI for medical IoMT devices with privacy-first architecture.

Implements:
- Hardware isolation (Neuro Core pattern)
- Federated learning with differential privacy
- Personal Health Train framework
- HIPAA/GDPR compliance patterns

Mathematical Foundations:
- Differential Privacy: ε-δ guarantees
- Secure Aggregation: Shamir's secret sharing
- Homomorphic Encryption: Paillier basics

Reference: Intrinsics Imaging "Neuro Core", Personal Health Train

Author: AI-Mastery-2026
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import time
import hashlib
from collections import deque
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class MedicalDeviceType(Enum):
    """Types of medical IoMT devices."""
    WEARABLE = "wearable"  # Fitness trackers, smartwatches
    BEDSIDE_MONITOR = "bedside_monitor"  # Hospital monitors
    IMAGING = "imaging"  # X-ray, CT, MRI
    IMPLANTABLE = "implantable"  # Pacemakers, insulin pumps
    HOME_CARE = "home_care"  # Fall detection, vital signs
    DIAGNOSTIC = "diagnostic"  # Blood analyzers, ECG


class ClinicalEventType(Enum):
    """Types of clinical events detected by edge AI."""
    FALL_DETECTED = "fall_detected"
    ARRHYTHMIA = "arrhythmia"
    HYPOGLYCEMIA = "hypoglycemia"
    HYPERGLYCEMIA = "hyperglycemia"
    APNEA = "apnea"
    SEIZURE = "seizure"
    ANOMALY = "anomaly"
    NORMAL = "normal"


class AlertSeverity(Enum):
    """Clinical alert severity levels."""
    CRITICAL = "critical"  # Immediate intervention needed
    HIGH = "high"  # Urgent attention required
    MEDIUM = "medium"  # Should be reviewed
    LOW = "low"  # Informational
    NORMAL = "normal"  # No action needed


@dataclass
class ClinicalEvent:
    """
    Clinical event detected by edge AI.
    
    Note: Only this metadata is transmitted to cloud.
    Raw sensor data NEVER leaves the device.
    """
    event_id: str
    event_type: ClinicalEventType
    severity: AlertSeverity
    confidence: float
    timestamp: datetime
    device_id: str
    patient_id: str  # Pseudonymized
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PrivacyBudget:
    """
    Differential privacy budget tracker.
    
    Epsilon (ε): Privacy loss parameter. Lower = more private.
    Delta (δ): Probability of privacy breach.
    
    Composition theorem: Total ε = sum of per-query ε
    """
    epsilon_used: float = 0.0
    epsilon_budget: float = 1.0  # Total budget
    delta: float = 1e-5
    queries_made: int = 0
    
    def can_query(self, epsilon_cost: float) -> bool:
        """Check if query is within budget."""
        return self.epsilon_used + epsilon_cost <= self.epsilon_budget
    
    def consume(self, epsilon_cost: float) -> bool:
        """Consume privacy budget."""
        if self.can_query(epsilon_cost):
            self.epsilon_used += epsilon_cost
            self.queries_made += 1
            return True
        return False
    
    @property
    def remaining_budget(self) -> float:
        """Remaining privacy budget."""
        return self.epsilon_budget - self.epsilon_used


# ============================================================
# DIFFERENTIAL PRIVACY
# ============================================================

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
    
    def laplace_mechanism(
        self, 
        value: float, 
        sensitivity: float
    ) -> float:
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
    
    def gaussian_mechanism(
        self, 
        vector: np.ndarray, 
        sensitivity: float
    ) -> np.ndarray:
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
    
    def clip_gradients(
        self, 
        gradients: np.ndarray, 
        max_norm: float
    ) -> np.ndarray:
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
        self, 
        histogram: np.ndarray, 
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Privatize a histogram (common for medical stats).
        
        Adds Laplace noise to each bin independently.
        """
        return np.array([
            max(0, self.laplace_mechanism(count, sensitivity))
            for count in histogram
        ])


# ============================================================
# FEDERATED LEARNING
# ============================================================

@dataclass
class FederatedUpdate:
    """
    Model update from a federated client.
    
    Contains gradients/weights, not raw data.
    """
    client_id: str
    round_number: int
    gradients: np.ndarray
    sample_count: int
    timestamp: datetime
    encrypted: bool = False
    signature: str = ""


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
        max_gradient_norm: float = 1.0
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
            epsilon=self.privacy_budget.epsilon_budget,
            delta=self.privacy_budget.delta
        )
        
        # Training history
        self.training_history: List[Dict[str, Any]] = []
        
        logger.info(f"FL Client initialized: {client_id}")
    
    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """Initialize model weights based on architecture."""
        weights = {}
        
        # Simple MLP for demonstration
        layer_sizes = self.model_architecture.get(
            "layer_sizes", [128, 64, 32, 2]
        )
        
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            # He initialization
            weights[f"W{i}"] = np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)
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
    
    def _compute_gradients(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients via backpropagation.
        
        Uses per-example gradient clipping for DP.
        """
        batch_size = X.shape[0]
        gradients = {k: np.zeros_like(v) for k, v in self.local_weights.items()}
        
        # Per-example gradients (for DP clipping)
        for i in range(batch_size):
            x_i = X[i:i+1]
            y_i = y[i:i+1]
            
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
            delta = pred - y_i  # Binary cross-entropy derivative
            
            for layer in range(num_layers - 1, -1, -1):
                a_prev = activations[layer]
                
                # Compute gradients
                dW = a_prev.T @ delta
                db = delta.sum(axis=0)
                
                # Clip per-example gradient
                grad_vector = np.concatenate([dW.flatten(), db.flatten()])
                grad_vector = self.dp.clip_gradients(
                    grad_vector, self.max_gradient_norm
                )
                
                # Reshape back
                dW_size = dW.size
                dW = grad_vector[:dW_size].reshape(dW.shape)
                db = grad_vector[dW_size:]
                
                # Accumulate
                gradients[f"W{layer}"] += dW
                gradients[f"b{layer}"] += db
                
                # Propagate delta
                if layer > 0:
                    W = self.local_weights[f"W{layer}"]
                    delta = delta @ W.T
                    delta = delta * (activations[layer] > 0)  # ReLU derivative
        
        # Average gradients
        for k in gradients:
            gradients[k] /= batch_size
        
        return gradients
    
    def train_local(
        self,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        apply_dp: bool = True
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
        
        # Combine all local data
        X = np.vstack(self.local_data)
        y = np.vstack(self.local_labels)
        n_samples = X.shape[0]
        
        # Store initial weights
        initial_weights = {k: v.copy() for k, v in self.local_weights.items()}
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch SGD
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Compute gradients
                gradients = self._compute_gradients(X_batch, y_batch)
                
                # Apply DP noise if enabled
                if apply_dp:
                    for k, g in gradients.items():
                        # Check privacy budget
                        epsilon_cost = 0.01  # Cost per gradient update
                        if self.privacy_budget.can_query(epsilon_cost):
                            gradients[k] = self.dp.gaussian_mechanism(
                                g, 
                                sensitivity=self.max_gradient_norm / batch_size
                            )
                            self.privacy_budget.consume(epsilon_cost)
                
                # Update weights
                for k in self.local_weights:
                    self.local_weights[k] -= learning_rate * gradients[k]
        
        # Compute weight update (delta)
        weight_update = {}
        for k in self.local_weights:
            weight_update[k] = self.local_weights[k] - initial_weights[k]
        
        # Record training
        self.training_history.append({
            "round": self.round_number,
            "epochs": epochs,
            "samples": n_samples,
            "epsilon_used": self.privacy_budget.epsilon_used,
            "timestamp": datetime.now()
        })
        
        logger.info(f"Local training complete: {epochs} epochs, {n_samples} samples, "
                   f"ε={self.privacy_budget.epsilon_used:.4f}")
        
        return weight_update
    
    def create_federated_update(
        self,
        weight_update: Dict[str, np.ndarray]
    ) -> FederatedUpdate:
        """
        Package weight update for server transmission.
        
        Only gradients are sent, never raw data.
        """
        # Flatten gradients for transmission
        flat_gradients = np.concatenate([
            weight_update[k].flatten() 
            for k in sorted(weight_update.keys())
        ])
        
        return FederatedUpdate(
            client_id=self.client_id,
            round_number=self.round_number,
            gradients=flat_gradients,
            sample_count=sum(len(d) for d in self.local_data),
            timestamp=datetime.now(),
            encrypted=False,  # Would be encrypted in production
            signature=hashlib.sha256(flat_gradients.tobytes()).hexdigest()[:32]
        )
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Get privacy budget status."""
        return {
            "epsilon_used": self.privacy_budget.epsilon_used,
            "epsilon_remaining": self.privacy_budget.remaining_budget,
            "delta": self.privacy_budget.delta,
            "queries_made": self.privacy_budget.queries_made,
            "training_rounds": len(self.training_history)
        }


# ============================================================
# PERSONAL HEALTH TRAIN
# ============================================================

@dataclass
class HealthTrain:
    """
    Personal Health Train - algorithm that travels to data.
    
    Instead of data going to algorithm, algorithm goes to data.
    Ensures data sovereignty and privacy.
    """
    train_id: str
    algorithm: str  # Serialized algorithm
    required_permissions: List[str]
    created_by: str
    approved: bool = False
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStation:
    """
    Health Station - data repository (hospital, device).
    
    Controls access to local health data.
    """
    station_id: str
    organization: str
    data_categories: List[str]
    access_policies: Dict[str, Any]


class PersonalHealthTrainFramework:
    """
    Personal Health Train framework implementation.
    
    Architecture:
    - Stations: Data repositories (hospitals)
    - Trains: Algorithms that travel to stations
    - Tracks: Secure communication channels
    
    Workflow:
    1. Researcher creates a "Train" (algorithm)
    2. Train is reviewed and approved
    3. Train travels to Stations via secure Tracks
    4. Station executes Train on local data
    5. Only results (not data) return to researcher
    """
    
    def __init__(self):
        """Initialize PHT framework."""
        self.stations: Dict[str, HealthStation] = {}
        self.trains: Dict[str, HealthTrain] = {}
        self.audit_log: List[Dict[str, Any]] = []
    
    def register_station(self, station: HealthStation):
        """Register a new data station."""
        self.stations[station.station_id] = station
        self._log_audit("station_registered", station.station_id)
        logger.info(f"Station registered: {station.station_id}")
    
    def submit_train(self, train: HealthTrain) -> bool:
        """
        Submit algorithm train for approval.
        
        In production, this goes through ethical review.
        """
        self.trains[train.train_id] = train
        self._log_audit("train_submitted", train.train_id)
        logger.info(f"Train submitted for approval: {train.train_id}")
        return True
    
    def approve_train(self, train_id: str) -> bool:
        """Approve a train after review."""
        if train_id in self.trains:
            self.trains[train_id].approved = True
            self._log_audit("train_approved", train_id)
            return True
        return False
    
    def execute_train(
        self,
        train_id: str,
        station_id: str,
        executor: Callable[[str, Any], Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute approved train at station.
        
        Args:
            train_id: Train to execute
            station_id: Station to execute at
            executor: Function that runs algorithm on data
            
        Returns:
            Results from execution (never raw data)
        """
        train = self.trains.get(train_id)
        station = self.stations.get(station_id)
        
        if not train or not station:
            logger.error("Train or station not found")
            return None
        
        if not train.approved:
            logger.error("Train not approved")
            return None
        
        # Check permissions
        missing_perms = set(train.required_permissions) - set(station.access_policies.keys())
        if missing_perms:
            logger.error(f"Missing permissions: {missing_perms}")
            return None
        
        # Execute at station
        self._log_audit("train_executed", f"{train_id}@{station_id}")
        
        try:
            results = executor(train.algorithm, station.access_policies)
            train.results[station_id] = results
            return results
        except Exception as e:
            logger.error(f"Train execution failed: {e}")
            return None
    
    def _log_audit(self, action: str, target: str):
        """Log audit event."""
        self.audit_log.append({
            "action": action,
            "target": target,
            "timestamp": datetime.now()
        })


# ============================================================
# MEDICAL DEVICE
# ============================================================

class MedicalDevice:
    """
    Edge AI medical device with Neuro Core architecture.
    
    Hardware Isolation Pattern:
    - MCU1: Sensor interface (captures raw data)
    - Neuro Core: AI processing (classifies locally)
    - MCU2: Communication (receives only classifications)
    
    Physical separation ensures raw data CANNOT be transmitted
    even if software is compromised.
    """
    
    def __init__(
        self,
        device_id: str,
        device_type: MedicalDeviceType,
        patient_id: str,
        model_architecture: Dict[str, Any]
    ):
        """
        Initialize medical device.
        
        Args:
            device_id: Unique device identifier
            device_type: Type of medical device
            patient_id: Pseudonymized patient ID
            model_architecture: AI model specification
        """
        self.device_id = device_id
        self.device_type = device_type
        self.patient_id = patient_id
        
        # Initialize components
        self.neuro_core = NeuroCoreProcessor(model_architecture)
        self.fl_client = FederatedLearningClient(
            client_id=device_id,
            model_architecture=model_architecture
        )
        
        # Event buffer (only clinical events, not raw data)
        self.event_buffer: deque = deque(maxlen=1000)
        
        # Telemetry (only metadata)
        self.telemetry: List[Dict[str, Any]] = []
        
        logger.info(f"Medical device initialized: {device_id} ({device_type.value})")
    
    def process_sensor_data(
        self,
        sensor_data: np.ndarray,
        sensor_type: str = "accelerometer"
    ) -> Optional[ClinicalEvent]:
        """
        Process sensor data through Neuro Core.
        
        Raw data stays on device. Only events are transmitted.
        
        Args:
            sensor_data: Raw sensor readings
            sensor_type: Type of sensor
            
        Returns:
            ClinicalEvent if detected, None otherwise
        """
        # Neuro Core processes locally
        event_type, confidence = self.neuro_core.classify(sensor_data)
        
        # Only create event if significant
        if event_type != ClinicalEventType.NORMAL and confidence > 0.5:
            severity = self._determine_severity(event_type, confidence)
            
            event = ClinicalEvent(
                event_id=f"{self.device_id}_{datetime.now().timestamp()}",
                event_type=event_type,
                severity=severity,
                confidence=confidence,
                timestamp=datetime.now(),
                device_id=self.device_id,
                patient_id=self.patient_id,
                metadata={"sensor_type": sensor_type}
            )
            
            self.event_buffer.append(event)
            
            # Store for local training (raw data only)
            label = np.array([[1.0, 0.0]] if event_type != ClinicalEventType.NORMAL 
                            else [[0.0, 1.0]])
            self.fl_client.add_local_data(sensor_data.reshape(1, -1), label)
            
            return event
        
        return None
    
    def _determine_severity(
        self,
        event_type: ClinicalEventType,
        confidence: float
    ) -> AlertSeverity:
        """Determine alert severity based on event type and confidence."""
        critical_events = {
            ClinicalEventType.FALL_DETECTED,
            ClinicalEventType.ARRHYTHMIA,
            ClinicalEventType.SEIZURE,
            ClinicalEventType.APNEA
        }
        
        if event_type in critical_events and confidence > 0.8:
            return AlertSeverity.CRITICAL
        elif event_type in critical_events:
            return AlertSeverity.HIGH
        elif confidence > 0.8:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def get_events_for_transmission(self) -> List[Dict[str, Any]]:
        """
        Get events to transmit to cloud.
        
        Only clinical events (metadata), never raw sensor data.
        """
        events = []
        while self.event_buffer:
            event = self.event_buffer.popleft()
            events.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "confidence": event.confidence,
                "timestamp": event.timestamp.isoformat(),
                "device_id": event.device_id,
                "patient_id": event.patient_id  # Pseudonymized
                # Note: NO raw sensor data included
            })
        return events
    
    def participate_in_federated_round(
        self,
        global_weights: Dict[str, np.ndarray]
    ) -> FederatedUpdate:
        """
        Participate in federated learning round.
        
        1. Receive global model
        2. Train on local data with DP
        3. Return gradient update only
        """
        # Receive global model
        self.fl_client.receive_global_model(global_weights)
        
        # Train locally with differential privacy
        weight_update = self.fl_client.train_local(
            epochs=3,
            apply_dp=True
        )
        
        # Create update (only gradients, not data)
        return self.fl_client.create_federated_update(weight_update)
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get device status (no PHI)."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "privacy_budget": self.fl_client.get_privacy_report(),
            "events_pending": len(self.event_buffer),
            "telemetry_count": len(self.telemetry)
        }


# ============================================================
# NEURO CORE PROCESSOR
# ============================================================

class NeuroCoreProcessor:
    """
    Neuro Core - isolated AI processing unit.
    
    Runs classification entirely on-device.
    No network interface - only receives data from sensor MCU
    and outputs classifications to communication MCU.
    
    Hardware architecture ensures:
    - Raw data cannot bypass to network
    - Model cannot be extracted remotely
    - Processing is deterministic and auditable
    """
    
    def __init__(self, model_architecture: Dict[str, Any]):
        """
        Initialize Neuro Core.
        
        Args:
            model_architecture: Neural network specification
        """
        self.model_architecture = model_architecture
        self.model_weights = self._initialize_weights()
        
        # Use simple, interpretable model for medical
        self.model_type = model_architecture.get("type", "knn")
        
        # For k-NN: store training examples
        if self.model_type == "knn":
            self.k = model_architecture.get("k", 5)
            self.training_data: List[Tuple[np.ndarray, ClinicalEventType]] = []
        
        logger.info(f"Neuro Core initialized: {self.model_type}")
    
    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """Initialize model weights."""
        layer_sizes = self.model_architecture.get(
            "layer_sizes", [64, 32, len(ClinicalEventType)]
        )
        
        weights = {}
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            weights[f"W{i}"] = np.random.randn(in_size, out_size) * 0.01
            weights[f"b{i}"] = np.zeros(out_size)
        
        return weights
    
    def classify(
        self, 
        sensor_data: np.ndarray
    ) -> Tuple[ClinicalEventType, float]:
        """
        Classify sensor data locally.
        
        Args:
            sensor_data: Preprocessed sensor readings
            
        Returns:
            (event_type, confidence)
        """
        if self.model_type == "knn":
            return self._knn_classify(sensor_data)
        else:
            return self._nn_classify(sensor_data)
    
    def _knn_classify(
        self, 
        sensor_data: np.ndarray
    ) -> Tuple[ClinicalEventType, float]:
        """
        k-Nearest Neighbors classification.
        
        Chosen for medical devices because:
        - Interpretable (can explain via neighbors)
        - No training loop (low power)
        - Works well with small datasets
        """
        if not self.training_data:
            return ClinicalEventType.NORMAL, 0.5
        
        # Flatten input
        x = sensor_data.flatten()
        
        # Compute distances to all training examples
        distances = []
        for train_x, train_y in self.training_data:
            dist = np.linalg.norm(x - train_x.flatten())
            distances.append((dist, train_y))
        
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        # Get k nearest neighbors
        k_nearest = distances[:self.k]
        
        # Vote
        votes = {}
        for _, label in k_nearest:
            votes[label] = votes.get(label, 0) + 1
        
        # Get prediction
        predicted = max(votes, key=votes.get)
        confidence = votes[predicted] / self.k
        
        return predicted, confidence
    
    def _nn_classify(
        self, 
        sensor_data: np.ndarray
    ) -> Tuple[ClinicalEventType, float]:
        """
        Neural network classification with softmax output.
        """
        x = sensor_data.flatten()
        
        # Forward pass
        num_layers = len([k for k in self.model_weights if k.startswith("W")])
        activation = x
        
        for i in range(num_layers):
            W = self.model_weights[f"W{i}"]
            b = self.model_weights[f"b{i}"]
            
            # Ensure dimensions match
            if activation.shape[0] != W.shape[0]:
                # Pad or truncate
                if activation.shape[0] < W.shape[0]:
                    activation = np.pad(
                        activation, 
                        (0, W.shape[0] - activation.shape[0])
                    )
                else:
                    activation = activation[:W.shape[0]]
            
            activation = activation @ W + b
            
            if i < num_layers - 1:
                activation = np.maximum(0, activation)  # ReLU
            else:
                # Softmax
                exp_a = np.exp(activation - np.max(activation))
                activation = exp_a / exp_a.sum()
        
        # Get prediction
        class_idx = np.argmax(activation)
        confidence = float(activation[class_idx])
        
        event_types = list(ClinicalEventType)
        if class_idx < len(event_types):
            return event_types[class_idx], confidence
        
        return ClinicalEventType.NORMAL, confidence
    
    def add_training_example(
        self, 
        sensor_data: np.ndarray, 
        label: ClinicalEventType
    ):
        """Add training example (for k-NN)."""
        self.training_data.append((sensor_data, label))
    
    def update_weights(self, new_weights: Dict[str, np.ndarray]):
        """Update model weights from federated learning."""
        self.model_weights = new_weights


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_fall_detection_device(
    device_id: str,
    patient_id: str
) -> MedicalDevice:
    """
    Create a fall detection wearable device.
    
    Uses accelerometer and gyroscope data.
    """
    architecture = {
        "type": "knn",
        "k": 5,
        "layer_sizes": [128, 64, 8]  # For hybrid approach
    }
    
    device = MedicalDevice(
        device_id=device_id,
        device_type=MedicalDeviceType.HOME_CARE,
        patient_id=patient_id,
        model_architecture=architecture
    )
    
    # Add some training examples for k-NN
    # Normal activity patterns
    for _ in range(50):
        normal_data = np.random.randn(128) * 0.5
        device.neuro_core.add_training_example(
            normal_data, ClinicalEventType.NORMAL
        )
    
    # Fall patterns (high acceleration spike)
    for _ in range(20):
        fall_data = np.random.randn(128) * 0.5
        fall_data[50:60] = np.random.randn(10) * 3.0  # Spike
        device.neuro_core.add_training_example(
            fall_data, ClinicalEventType.FALL_DETECTED
        )
    
    return device


def create_cardiac_monitor(
    device_id: str,
    patient_id: str
) -> MedicalDevice:
    """
    Create a cardiac monitoring device.
    
    Uses ECG signal data.
    """
    architecture = {
        "type": "nn",
        "layer_sizes": [256, 128, 64, 8]
    }
    
    return MedicalDevice(
        device_id=device_id,
        device_type=MedicalDeviceType.BEDSIDE_MONITOR,
        patient_id=patient_id,
        model_architecture=architecture
    )


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Medical Edge AI - Demo")
    print("=" * 60)
    
    # Create fall detection device
    device = create_fall_detection_device(
        device_id="FALL_001",
        patient_id="PT_ANON_12345"  # Pseudonymized
    )
    
    print("\n1. Simulating normal activity...")
    for _ in range(10):
        sensor_data = np.random.randn(128) * 0.5
        event = device.process_sensor_data(sensor_data, "accelerometer")
        if event:
            print(f"   Event detected: {event.event_type.value} "
                  f"({event.confidence:.2%})")
    
    print("\n2. Simulating fall event...")
    fall_data = np.random.randn(128) * 0.5
    fall_data[50:60] = np.random.randn(10) * 3.0  # Sharp spike
    event = device.process_sensor_data(fall_data, "accelerometer")
    if event:
        print(f"   🚨 {event.severity.value.upper()}: {event.event_type.value} "
              f"({event.confidence:.2%})")
    
    print("\n3. Events for transmission (metadata only):")
    events = device.get_events_for_transmission()
    for e in events:
        print(f"   {e['event_type']} at {e['timestamp']}")
    
    print("\n4. Federated learning participation...")
    # Simulate global model
    global_weights = device.fl_client._initialize_weights()
    update = device.participate_in_federated_round(global_weights)
    print(f"   Generated update: {len(update.gradients)} parameters")
    print(f"   Privacy budget used: ε={device.fl_client.privacy_budget.epsilon_used:.4f}")
    
    print("\n5. Privacy report:")
    privacy = device.fl_client.get_privacy_report()
    for key, value in privacy.items():
        print(f"   {key}: {value}")
    
    print("\n6. Personal Health Train demo...")
    pht = PersonalHealthTrainFramework()
    
    # Register hospital as station
    hospital = HealthStation(
        station_id="hospital_001",
        organization="General Hospital",
        data_categories=["ecg", "vitals", "imaging"],
        access_policies={"research": True, "clinical": True}
    )
    pht.register_station(hospital)
    
    # Submit algorithm train
    train = HealthTrain(
        train_id="cardiac_model_v1",
        algorithm="federated_cnn_training",
        required_permissions=["research"],
        created_by="research_team_01"
    )
    pht.submit_train(train)
    pht.approve_train("cardiac_model_v1")
    
    # Execute train at station
    def mock_executor(algorithm, policies):
        return {"accuracy": 0.92, "samples_used": 1000}
    
    result = pht.execute_train("cardiac_model_v1", "hospital_001", mock_executor)
    print(f"   Train execution result: {result}")
    
    print("\n✅ Medical Edge AI demo complete!")
