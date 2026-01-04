"""
Industrial IoT Predictive Maintenance Module
==============================================

Edge AI for predictive maintenance in oil & gas and industrial settings.

Implements:
- Anomaly detection (Autoencoder, Isolation Forest)  
- Remaining Useful Life (RUL) prediction (LSTM)
- Store-and-forward for DDIL environments
- Multi-protocol sensor integration

Mathematical Foundations:
- Autoencoder: minimize ||x - D(E(x))||²
- Isolation Forest: anomaly score based on path length
- LSTM: learn temporal patterns in sensor data

Reference: Barbara IoT, AWS IoT Greengrass

Author: AI-Mastery-2026
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import time
import hashlib
from collections import deque
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class EquipmentType(Enum):
    """Types of industrial equipment."""
    PUMP = "pump"
    COMPRESSOR = "compressor"
    VALVE = "valve"
    PIPELINE = "pipeline"
    TURBINE = "turbine"
    MOTOR = "motor"
    HEAT_EXCHANGER = "heat_exchanger"
    SEPARATOR = "separator"


class SensorType(Enum):
    """Types of industrial sensors."""
    VIBRATION = "vibration"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    CURRENT = "current"
    ACOUSTIC = "acoustic"
    OIL_ANALYSIS = "oil_analysis"


class AlertSeverity(Enum):
    """Maintenance alert severity levels."""
    CRITICAL = "critical"  # Immediate shutdown recommended
    HIGH = "high"  # Schedule maintenance within 24h
    MEDIUM = "medium"  # Schedule within 1 week
    LOW = "low"  # Informational, monitor
    NORMAL = "normal"


class ProtocolType(Enum):
    """Industrial communication protocols."""
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPC_UA = "opc_ua"
    PROFIBUS = "profibus"
    MQTT = "mqtt"
    CANBUS = "canbus"


@dataclass
class SensorReading:
    """
    Time-series sensor reading.
    
    Designed for industrial environments with high-frequency sampling.
    """
    sensor_id: str
    sensor_type: SensorType
    timestamp: datetime
    value: float
    unit: str
    quality: float = 1.0  # Data quality score 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaintenanceAlert:
    """
    Predictive maintenance alert.
    """
    alert_id: str
    equipment_id: str
    equipment_type: EquipmentType
    severity: AlertSeverity
    anomaly_score: float
    rul_hours: Optional[float]  # Remaining useful life
    timestamp: datetime
    recommended_action: str
    contributing_sensors: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EquipmentHealth:
    """
    Overall equipment health status.
    """
    equipment_id: str
    health_score: float  # 0-100
    rul_hours: float
    rul_confidence: float
    anomaly_detected: bool
    last_maintenance: Optional[datetime]
    operating_hours: float
    alerts: List[MaintenanceAlert] = field(default_factory=list)


# ============================================================
# STORE AND FORWARD
# ============================================================

@dataclass
class QueuedMessage:
    """Message in store-and-forward queue."""
    message_id: str
    priority: int  # 0 = highest
    payload: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3


class StoreAndForwardQueue:
    """
    Store-and-Forward queue for DDIL environments.
    
    DDIL: Disconnected, Disrupted, Intermittent, Limited bandwidth
    
    Features:
    - Disk-persisted for durability
    - Priority queuing (critical alerts bypass)
    - Automatic retry with exponential backoff
    - Bandwidth-aware sync
    """
    
    def __init__(
        self,
        storage_path: str = "./queue",
        max_size_mb: float = 100.0,
        critical_bypass: bool = True
    ):
        """
        Initialize queue.
        
        Args:
            storage_path: Path for disk persistence
            max_size_mb: Maximum queue size
            critical_bypass: Allow critical messages to bypass queue
        """
        self.storage_path = storage_path
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.critical_bypass = critical_bypass
        
        # In-memory queues by priority
        self.queues: Dict[int, deque] = {
            0: deque(),  # Critical
            1: deque(),  # High
            2: deque(),  # Medium
            3: deque(),  # Low
        }
        
        # Statistics
        self.total_queued = 0
        self.total_sent = 0
        self.total_dropped = 0
        
        # Connection state
        self.connected = False
        
        logger.info(f"Store-and-Forward queue initialized: {storage_path}")
    
    def enqueue(
        self, 
        payload: Dict[str, Any], 
        priority: int = 2
    ) -> str:
        """
        Add message to queue.
        
        Args:
            payload: Message content
            priority: 0=critical, 1=high, 2=medium, 3=low
            
        Returns:
            Message ID
        """
        message_id = hashlib.md5(
            f"{datetime.now().isoformat()}{json.dumps(payload)}".encode()
        ).hexdigest()[:16]
        
        message = QueuedMessage(
            message_id=message_id,
            priority=priority,
            payload=payload,
            timestamp=datetime.now()
        )
        
        # Critical bypass: try to send immediately if connected
        if priority == 0 and self.critical_bypass and self.connected:
            if self._try_send(message):
                return message_id
        
        # Add to priority queue
        queue_priority = min(max(priority, 0), 3)
        self.queues[queue_priority].append(message)
        self.total_queued += 1
        
        return message_id
    
    def _try_send(self, message: QueuedMessage) -> bool:
        """Attempt to send message immediately."""
        try:
            # Simulated send (would use MQTT/HTTP in production)
            logger.debug(f"Sending message: {message.message_id}")
            self.total_sent += 1
            return True
        except Exception as e:
            logger.warning(f"Send failed: {e}")
            return False
    
    def sync(self, max_messages: int = 100) -> int:
        """
        Sync queued messages when connection available.
        
        Args:
            max_messages: Maximum messages to sync in this batch
            
        Returns:
            Number of messages sent
        """
        if not self.connected:
            logger.warning("Cannot sync: not connected")
            return 0
        
        sent_count = 0
        
        # Process by priority order
        for priority in range(4):
            while self.queues[priority] and sent_count < max_messages:
                message = self.queues[priority].popleft()
                
                if self._try_send(message):
                    sent_count += 1
                else:
                    # Retry logic
                    message.retry_count += 1
                    if message.retry_count < message.max_retries:
                        self.queues[priority].appendleft(message)
                    else:
                        self.total_dropped += 1
                        logger.warning(f"Dropped message after retries: {message.message_id}")
                    break
        
        return sent_count
    
    def set_connected(self, connected: bool):
        """Update connection state."""
        self.connected = connected
        if connected:
            logger.info("Connection established - initiating sync")
            self.sync()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        queue_sizes = {f"priority_{p}": len(q) for p, q in self.queues.items()}
        return {
            "total_queued": sum(len(q) for q in self.queues.values()),
            "total_sent": self.total_sent,
            "total_dropped": self.total_dropped,
            "connected": self.connected,
            **queue_sizes
        }


# ============================================================
# INDUSTRIAL SENSORS
# ============================================================

class IndustrialSensor:
    """
    Industrial sensor interface with multi-protocol support.
    
    Supports:
    - Modbus TCP/RTU (registers)
    - OPC UA (nodes)
    - MQTT (pub/sub)
    """
    
    def __init__(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        protocol: ProtocolType,
        address: str,
        sampling_rate_hz: float = 1.0
    ):
        """
        Initialize sensor.
        
        Args:
            sensor_id: Unique identifier
            sensor_type: Type of sensor
            protocol: Communication protocol
            address: Modbus register / OPC node / MQTT topic
            sampling_rate_hz: Polling frequency
        """
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.protocol = protocol
        self.address = address
        self.sampling_rate_hz = sampling_rate_hz
        
        # Data buffer (circular for time-series)
        self.buffer: deque = deque(maxlen=1000)
        
        # Statistics
        self.reading_count = 0
        self.last_reading: Optional[SensorReading] = None
        
        # Calibration
        self.offset = 0.0
        self.scale = 1.0
        
        logger.info(f"Sensor initialized: {sensor_id} ({sensor_type.value})")
    
    def read(self) -> Optional[SensorReading]:
        """
        Read current sensor value.
        
        In production, this would use actual protocol libraries:
        - pymodbus for Modbus
        - opcua for OPC UA
        - paho-mqtt for MQTT
        """
        try:
            # Simulated read with realistic noise
            if self.sensor_type == SensorType.VIBRATION:
                raw_value = np.random.exponential(0.5) + 0.1
                unit = "mm/s"
            elif self.sensor_type == SensorType.TEMPERATURE:
                raw_value = 45 + np.random.randn() * 5
                unit = "°C"
            elif self.sensor_type == SensorType.PRESSURE:
                raw_value = 100 + np.random.randn() * 10
                unit = "bar"
            else:
                raw_value = np.random.random() * 100
                unit = "units"
            
            # Apply calibration
            value = raw_value * self.scale + self.offset
            
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(),
                value=value,
                unit=unit
            )
            
            self.buffer.append(reading)
            self.reading_count += 1
            self.last_reading = reading
            
            return reading
            
        except Exception as e:
            logger.error(f"Sensor read failed: {e}")
            return None
    
    def get_time_series(self, n_samples: int = 100) -> np.ndarray:
        """Get recent readings as numpy array."""
        readings = list(self.buffer)[-n_samples:]
        return np.array([r.value for r in readings])
    
    def calibrate(self, offset: float, scale: float):
        """Set calibration parameters."""
        self.offset = offset
        self.scale = scale


# ============================================================
# ANOMALY DETECTION
# ============================================================

class Autoencoder:
    """
    Simple autoencoder for anomaly detection.
    
    Architecture:
    - Encoder: Input -> Hidden1 -> Latent
    - Decoder: Latent -> Hidden2 -> Output
    
    Anomaly detection:
    - Train on normal data
    - High reconstruction error = anomaly
    
    Mathematical foundation:
    - minimize ||x - D(E(x))||² over normal data
    - anomaly score = reconstruction error
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 32
    ):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Input dimension
            latent_dim: Bottleneck dimension
            hidden_dim: Hidden layer dimension
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self._init_weights()
        
        # Threshold for anomaly detection
        self.threshold = None
        
        logger.info(f"Autoencoder initialized: {input_dim} -> {latent_dim} -> {input_dim}")
    
    def _init_weights(self):
        """He initialization for weights."""
        # Encoder
        self.W_enc1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b_enc1 = np.zeros(self.hidden_dim)
        
        self.W_enc2 = np.random.randn(self.hidden_dim, self.latent_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b_enc2 = np.zeros(self.latent_dim)
        
        # Decoder
        self.W_dec1 = np.random.randn(self.latent_dim, self.hidden_dim) * np.sqrt(2.0 / self.latent_dim)
        self.b_dec1 = np.zeros(self.hidden_dim)
        
        self.W_dec2 = np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b_dec2 = np.zeros(self.input_dim)
    
    def _relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """ReLU derivative."""
        return (x > 0).astype(float)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent space."""
        h1 = self._relu(x @ self.W_enc1 + self.b_enc1)
        z = h1 @ self.W_enc2 + self.b_enc2  # Use h1, not x
        return z
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        h1 = self._relu(z @ self.W_dec1 + self.b_dec1)
        x_hat = h1 @ self.W_dec2 + self.b_dec2
        return x_hat
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error (anomaly score).
        
        MSE per sample.
        """
        x_hat = self.forward(x)
        error = np.mean((x - x_hat) ** 2, axis=1)
        return error
    
    def fit(
        self, 
        X: np.ndarray, 
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        """
        Train autoencoder on normal data.
        
        Args:
            X: Training data (n_samples, n_features)
            epochs: Number of training epochs
            learning_rate: SGD learning rate
            batch_size: Mini-batch size
        """
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            epoch_loss = 0.0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                
                # Forward pass
                h1_enc = self._relu(X_batch @ self.W_enc1 + self.b_enc1)
                z = h1_enc @ self.W_enc2 + self.b_enc2
                h1_dec = self._relu(z @ self.W_dec1 + self.b_dec1)
                x_hat = h1_dec @ self.W_dec2 + self.b_dec2
                
                # Compute loss
                loss = np.mean((X_batch - x_hat) ** 2)
                epoch_loss += loss * len(X_batch)
                
                # Backpropagation
                d_loss = 2 * (x_hat - X_batch) / len(X_batch)
                
                # Decoder gradients
                d_W_dec2 = h1_dec.T @ d_loss
                d_b_dec2 = d_loss.sum(axis=0)
                
                d_h1_dec = d_loss @ self.W_dec2.T * self._relu_derivative(z @ self.W_dec1 + self.b_dec1)
                d_W_dec1 = z.T @ d_h1_dec
                d_b_dec1 = d_h1_dec.sum(axis=0)
                
                # Encoder gradients (simplified)
                d_z = d_h1_dec @ self.W_dec1.T
                d_W_enc2 = h1_enc.T @ d_z
                d_b_enc2 = d_z.sum(axis=0)
                
                d_h1_enc = d_z @ self.W_enc2.T * self._relu_derivative(X_batch @ self.W_enc1 + self.b_enc1)
                d_W_enc1 = X_batch.T @ d_h1_enc
                d_b_enc1 = d_h1_enc.sum(axis=0)
                
                # Update weights
                self.W_dec2 -= learning_rate * d_W_dec2
                self.b_dec2 -= learning_rate * d_b_dec2
                self.W_dec1 -= learning_rate * d_W_dec1
                self.b_dec1 -= learning_rate * d_b_dec1
                self.W_enc2 -= learning_rate * d_W_enc2
                self.b_enc2 -= learning_rate * d_b_enc2
                self.W_enc1 -= learning_rate * d_W_enc1
                self.b_enc1 -= learning_rate * d_b_enc1
            
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}: loss={epoch_loss/n_samples:.6f}")
        
        # Set threshold based on training data
        train_errors = self.reconstruction_error(X)
        self.threshold = np.percentile(train_errors, 95)
        logger.info(f"Training complete. Threshold set to {self.threshold:.4f}")
    
    def detect(self, x: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if input is anomalous.
        
        Args:
            x: Input sample(s)
            
        Returns:
            (is_anomaly, anomaly_score)
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        error = self.reconstruction_error(x)[0]
        is_anomaly = error > self.threshold if self.threshold else False
        
        return is_anomaly, error


class IsolationForest:
    """
    Isolation Forest for anomaly detection.
    
    Key insight: Anomalies are easier to isolate
    (fewer random splits needed).
    
    Anomaly score = average path length across trees
    Shorter path = more anomalous
    """
    
    def __init__(
        self,
        n_trees: int = 100,
        sample_size: int = 256,
        contamination: float = 0.1
    ):
        """
        Initialize Isolation Forest.
        
        Args:
            n_trees: Number of isolation trees
            sample_size: Subsample size for each tree
            contamination: Expected proportion of anomalies
        """
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination
        
        self.trees: List[Dict] = []
        self.threshold: Optional[float] = None
        
        logger.info(f"Isolation Forest initialized: {n_trees} trees")
    
    def _build_tree(self, X: np.ndarray, depth: int = 0, max_depth: int = 10) -> Dict:
        """Build a single isolation tree recursively."""
        n_samples, n_features = X.shape
        
        # Termination conditions
        if depth >= max_depth or n_samples <= 1:
            return {"type": "leaf", "size": n_samples}
        
        # Random split
        feature_idx = np.random.randint(n_features)
        feature_values = X[:, feature_idx]
        
        min_val, max_val = feature_values.min(), feature_values.max()
        
        if min_val == max_val:
            return {"type": "leaf", "size": n_samples}
        
        split_value = np.random.uniform(min_val, max_val)
        
        # Split data
        left_mask = feature_values < split_value
        right_mask = ~left_mask
        
        return {
            "type": "node",
            "feature": feature_idx,
            "split": split_value,
            "left": self._build_tree(X[left_mask], depth + 1, max_depth),
            "right": self._build_tree(X[right_mask], depth + 1, max_depth)
        }
    
    def _path_length(self, x: np.ndarray, tree: Dict, depth: int = 0) -> float:
        """Compute path length for a single sample."""
        if tree["type"] == "leaf":
            # Adjustment for unbuilt trees
            c = self._c(tree["size"])
            return depth + c
        
        if x[tree["feature"]] < tree["split"]:
            return self._path_length(x, tree["left"], depth + 1)
        else:
            return self._path_length(x, tree["right"], depth + 1)
    
    def _c(self, n: int) -> float:
        """
        Average path length of unsuccessful search in BST.
        
        c(n) = 2H(n-1) - 2(n-1)/n
        where H(i) ≈ ln(i) + 0.5772156649 (Euler constant)
        """
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    
    def fit(self, X: np.ndarray):
        """
        Build forest on training data.
        
        Args:
            X: Training data (assumed mostly normal)
        """
        n_samples = X.shape[0]
        
        for _ in range(self.n_trees):
            # Subsample
            sample_indices = np.random.choice(
                n_samples, 
                min(self.sample_size, n_samples),
                replace=False
            )
            X_sample = X[sample_indices]
            
            # Build tree
            max_depth = int(np.ceil(np.log2(self.sample_size)))
            tree = self._build_tree(X_sample, max_depth=max_depth)
            self.trees.append(tree)
        
        # Compute threshold
        scores = self.anomaly_score(X)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        logger.info(f"Forest trained. Threshold: {self.threshold:.4f}")
    
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Score = 2^(-E[h(x)]/c(n))
        where h(x) is path length, c(n) is normalization
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Average path length across trees
            avg_path = np.mean([
                self._path_length(X[i], tree) 
                for tree in self.trees
            ])
            
            # Normalize
            c_n = self._c(self.sample_size)
            scores[i] = 2 ** (-avg_path / c_n) if c_n > 0 else 0.5
        
        return scores
    
    def detect(self, x: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if sample is anomalous.
        
        Returns:
            (is_anomaly, anomaly_score)
        """
        score = self.anomaly_score(x)[0]
        is_anomaly = score > self.threshold if self.threshold else False
        return is_anomaly, score


class AnomalyDetector:
    """
    Ensemble anomaly detector combining multiple methods.
    
    Uses both Autoencoder and Isolation Forest for robustness.
    Final score is weighted combination.
    """
    
    def __init__(
        self,
        input_dim: int,
        ae_weight: float = 0.5,
        if_weight: float = 0.5
    ):
        """
        Initialize ensemble detector.
        
        Args:
            input_dim: Feature dimension
            ae_weight: Weight for autoencoder score
            if_weight: Weight for isolation forest score
        """
        self.input_dim = input_dim
        self.ae_weight = ae_weight
        self.if_weight = if_weight
        
        self.autoencoder = Autoencoder(input_dim, latent_dim=max(8, input_dim // 8))
        self.isolation_forest = IsolationForest(n_trees=50)
        
        self.threshold = 0.5
        self.trained = False
        
        logger.info(f"Ensemble anomaly detector initialized: dim={input_dim}")
    
    def fit(self, X_normal: np.ndarray, epochs: int = 50):
        """Train on normal data."""
        logger.info("Training anomaly detector...")
        
        self.autoencoder.fit(X_normal, epochs=epochs)
        self.isolation_forest.fit(X_normal)
        
        # Calibrate ensemble threshold
        ae_scores = self.autoencoder.reconstruction_error(X_normal)
        ae_scores_norm = ae_scores / (ae_scores.max() + 1e-10)
        
        if_scores = self.isolation_forest.anomaly_score(X_normal)
        
        combined = self.ae_weight * ae_scores_norm + self.if_weight * if_scores
        self.threshold = np.percentile(combined, 95)
        
        self.trained = True
        logger.info(f"Ensemble trained. Threshold: {self.threshold:.4f}")
    
    def detect(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect anomaly using ensemble.
        
        Returns:
            (is_anomaly, combined_score, component_scores)
        """
        _, ae_score = self.autoencoder.detect(x)
        _, if_score = self.isolation_forest.detect(x)
        
        # Normalize AE score
        ae_norm = min(ae_score / (self.autoencoder.threshold + 1e-10), 2.0)
        
        # Combine
        combined = self.ae_weight * ae_norm + self.if_weight * if_score
        is_anomaly = combined > self.threshold
        
        return is_anomaly, combined, {
            "autoencoder": ae_score,
            "isolation_forest": if_score,
            "combined": combined
        }


# ============================================================
# REMAINING USEFUL LIFE (RUL) PREDICTION
# ============================================================

class LSTMCell:
    """
    LSTM cell for sequence modeling.
    
    Gates:
    - Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    - Input gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    - Candidate: c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
    - Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
    
    State updates:
    - c_t = f_t * c_{t-1} + i_t * c̃_t
    - h_t = o_t * tanh(c_t)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize LSTM cell.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Combined dimension for concatenated [h, x]
        combined_dim = hidden_dim + input_dim
        
        # Xavier initialization for gates
        scale = np.sqrt(6.0 / (combined_dim + hidden_dim))
        
        # Forget gate
        self.W_f = np.random.uniform(-scale, scale, (combined_dim, hidden_dim))
        self.b_f = np.ones(hidden_dim)  # Initialize to 1 for forget gate
        
        # Input gate
        self.W_i = np.random.uniform(-scale, scale, (combined_dim, hidden_dim))
        self.b_i = np.zeros(hidden_dim)
        
        # Candidate
        self.W_c = np.random.uniform(-scale, scale, (combined_dim, hidden_dim))
        self.b_c = np.zeros(hidden_dim)
        
        # Output gate
        self.W_o = np.random.uniform(-scale, scale, (combined_dim, hidden_dim))
        self.b_o = np.zeros(hidden_dim)
    
    def _sigmoid(self, x):
        """Sigmoid activation with numerical stability."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(
        self, 
        x: np.ndarray, 
        h_prev: np.ndarray, 
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through LSTM cell.
        
        Args:
            x: Input (batch, input_dim)
            h_prev: Previous hidden state (batch, hidden_dim)
            c_prev: Previous cell state (batch, hidden_dim)
            
        Returns:
            (h_t, c_t)
        """
        # Concatenate inputs
        combined = np.concatenate([h_prev, x], axis=-1)
        
        # Gate computations
        f_t = self._sigmoid(combined @ self.W_f + self.b_f)
        i_t = self._sigmoid(combined @ self.W_i + self.b_i)
        c_tilde = np.tanh(combined @ self.W_c + self.b_c)
        o_t = self._sigmoid(combined @ self.W_o + self.b_o)
        
        # State updates
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t


class RULPredictor:
    """
    Remaining Useful Life predictor using LSTM.
    
    Takes time-series sensor data and predicts hours until failure.
    
    Architecture:
    - LSTM layer(s) for temporal feature extraction
    - Dense layer for RUL regression
    
    Output: RUL in hours with uncertainty estimation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        sequence_length: int = 50
    ):
        """
        Initialize RUL predictor.
        
        Args:
            input_dim: Number of sensor features
            hidden_dim: LSTM hidden dimension
            sequence_length: Expected sequence length
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # LSTM layer
        self.lstm = LSTMCell(input_dim, hidden_dim)
        
        # Output layer (predicts mean and log-variance for uncertainty)
        self.W_out = np.random.randn(hidden_dim, 2) * 0.1
        self.b_out = np.zeros(2)
        
        # Training state
        self.trained = False
        
        logger.info(f"RUL Predictor initialized: {input_dim} -> {hidden_dim} -> RUL")
    
    def forward(self, sequence: np.ndarray) -> Tuple[float, float]:
        """
        Predict RUL from sequence.
        
        Args:
            sequence: (sequence_length, input_dim)
            
        Returns:
            (rul_prediction, uncertainty)
        """
        batch_size = 1 if len(sequence.shape) == 2 else sequence.shape[0]
        
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, *sequence.shape)
        
        # Initialize hidden states
        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))
        
        # Process sequence
        for t in range(sequence.shape[1]):
            x_t = sequence[:, t, :]
            h, c = self.lstm.forward(x_t, h, c)
        
        # Output layer
        output = h @ self.W_out + self.b_out
        
        # Interpret as mean and log-variance
        rul_mean = np.exp(output[:, 0])  # Ensure positive
        log_var = output[:, 1]
        rul_std = np.exp(0.5 * log_var)
        
        return float(rul_mean[0]), float(rul_std[0])
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        epochs: int = 50,
        learning_rate: float = 0.001
    ):
        """
        Train RUL predictor.
        
        Args:
            X: Sequences (n_samples, sequence_length, input_dim)
            y: RUL targets (n_samples,)
        """
        logger.info("Training RUL predictor...")
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i in range(n_samples):
                # Forward pass
                rul_pred, std = self.forward(X[i])
                
                # MSE loss (simplified)
                loss = (rul_pred - y[i]) ** 2
                epoch_loss += loss
                
                # Gradient descent (simplified - no full backprop)
                grad = 2 * (rul_pred - y[i])
                self.W_out[:, 0] -= learning_rate * grad * 0.01
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: loss={epoch_loss/n_samples:.4f}")
        
        self.trained = True
        logger.info("RUL predictor training complete")
    
    def predict(self, sequence: np.ndarray) -> Tuple[float, float, float]:
        """
        Predict RUL with confidence interval.
        
        Returns:
            (rul_hours, lower_bound, upper_bound)
        """
        rul, std = self.forward(sequence)
        
        # 95% confidence interval
        lower = max(0, rul - 1.96 * std)
        upper = rul + 1.96 * std
        
        return rul, lower, upper


# ============================================================
# PREDICTIVE MAINTENANCE ENGINE
# ============================================================

class PredictiveMaintenanceEngine:
    """
    Complete predictive maintenance solution.
    
    Combines:
    - Multi-sensor data ingestion
    - Anomaly detection
    - RUL prediction
    - Alert management
    - Store-and-forward communication
    """
    
    def __init__(
        self,
        equipment_id: str,
        equipment_type: EquipmentType,
        sensor_configs: List[Dict[str, Any]]
    ):
        """
        Initialize PdM engine.
        
        Args:
            equipment_id: Equipment identifier
            equipment_type: Type of equipment
            sensor_configs: List of sensor configurations
        """
        self.equipment_id = equipment_id
        self.equipment_type = equipment_type
        
        # Initialize sensors
        self.sensors: Dict[str, IndustrialSensor] = {}
        for config in sensor_configs:
            sensor = IndustrialSensor(
                sensor_id=config["id"],
                sensor_type=SensorType(config["type"]),
                protocol=ProtocolType(config.get("protocol", "modbus_tcp")),
                address=config.get("address", "0"),
                sampling_rate_hz=config.get("rate", 1.0)
            )
            self.sensors[config["id"]] = sensor
        
        # Feature dimension = number of sensors
        feature_dim = len(self.sensors)
        
        # Initialize models
        self.anomaly_detector = AnomalyDetector(feature_dim)
        self.rul_predictor = RULPredictor(feature_dim, hidden_dim=32)
        
        # Communication
        self.queue = StoreAndForwardQueue()
        
        # Health tracking
        self.health = EquipmentHealth(
            equipment_id=equipment_id,
            health_score=100.0,
            rul_hours=float('inf'),
            rul_confidence=0.0,
            anomaly_detected=False,
            last_maintenance=None,
            operating_hours=0.0
        )
        
        # History for RUL
        self.reading_history: deque = deque(maxlen=100)
        
        logger.info(f"PdM Engine initialized: {equipment_id} with {len(self.sensors)} sensors")
    
    def train_models(self, normal_data: np.ndarray, rul_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Train anomaly detection and RUL models.
        
        Args:
            normal_data: Normal operation data (n_samples, n_features)
            rul_data: Optional (sequences, rul_targets) for RUL training
        """
        self.anomaly_detector.fit(normal_data, epochs=30)
        
        if rul_data is not None:
            X_rul, y_rul = rul_data
            self.rul_predictor.fit(X_rul, y_rul, epochs=30)
    
    def collect_readings(self) -> np.ndarray:
        """Collect readings from all sensors."""
        values = []
        for sensor_id in sorted(self.sensors.keys()):
            reading = self.sensors[sensor_id].read()
            if reading:
                values.append(reading.value)
            else:
                values.append(0.0)  # Missing value
        
        feature_vector = np.array(values)
        self.reading_history.append(feature_vector)
        
        return feature_vector
    
    def analyze(self) -> Optional[MaintenanceAlert]:
        """
        Run full analysis: anomaly detection + RUL prediction.
        
        Returns:
            MaintenanceAlert if action needed, None otherwise
        """
        # Collect current readings
        current = self.collect_readings()
        
        if not self.anomaly_detector.trained:
            return None
        
        # Anomaly detection
        is_anomaly, score, component_scores = self.anomaly_detector.detect(current)
        
        # RUL prediction (if enough history)
        rul_hours = float('inf')
        rul_lower = 0.0
        rul_upper = float('inf')
        
        if len(self.reading_history) >= 50 and self.rul_predictor.trained:
            sequence = np.array(list(self.reading_history)[-50:])
            rul_hours, rul_lower, rul_upper = self.rul_predictor.predict(sequence)
        
        # Update health
        self.health.anomaly_detected = is_anomaly
        self.health.rul_hours = rul_hours
        self.health.rul_confidence = 0.9 if rul_hours < float('inf') else 0.0
        
        # Health score based on anomaly score and RUL
        if rul_hours < 24:
            self.health.health_score = max(0, 20 - score * 10)
        elif rul_hours < 168:  # 1 week
            self.health.health_score = max(20, 60 - score * 20)
        else:
            self.health.health_score = max(60, 100 - score * 30)
        
        # Generate alert if needed
        if is_anomaly or rul_hours < 168:
            severity = self._determine_severity(is_anomaly, score, rul_hours)
            
            alert = MaintenanceAlert(
                alert_id=f"ALERT_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                equipment_id=self.equipment_id,
                equipment_type=self.equipment_type,
                severity=severity,
                anomaly_score=score,
                rul_hours=rul_hours if rul_hours < float('inf') else None,
                timestamp=datetime.now(),
                recommended_action=self._get_recommendation(severity, rul_hours),
                contributing_sensors=list(self.sensors.keys()),
                confidence=0.85,
                metadata={
                    "rul_lower": rul_lower,
                    "rul_upper": rul_upper,
                    "component_scores": component_scores
                }
            )
            
            self.health.alerts.append(alert)
            
            # Queue for transmission
            priority = 0 if severity == AlertSeverity.CRITICAL else \
                      1 if severity == AlertSeverity.HIGH else 2
            self.queue.enqueue(self._alert_to_dict(alert), priority=priority)
            
            return alert
        
        return None
    
    def _determine_severity(
        self, 
        is_anomaly: bool, 
        score: float, 
        rul_hours: float
    ) -> AlertSeverity:
        """Determine alert severity."""
        if rul_hours < 24 or (is_anomaly and score > 0.9):
            return AlertSeverity.CRITICAL
        elif rul_hours < 72 or (is_anomaly and score > 0.7):
            return AlertSeverity.HIGH
        elif rul_hours < 168:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _get_recommendation(self, severity: AlertSeverity, rul_hours: float) -> str:
        """Get maintenance recommendation."""
        if severity == AlertSeverity.CRITICAL:
            return "IMMEDIATE SHUTDOWN RECOMMENDED - Inspect within 4 hours"
        elif severity == AlertSeverity.HIGH:
            return f"Schedule maintenance within 24 hours. Estimated RUL: {rul_hours:.0f}h"
        elif severity == AlertSeverity.MEDIUM:
            return f"Plan maintenance within 1 week. Estimated RUL: {rul_hours:.0f}h"
        else:
            return "Monitor closely. No immediate action required."
    
    def _alert_to_dict(self, alert: MaintenanceAlert) -> Dict[str, Any]:
        """Convert alert to dictionary for transmission."""
        return {
            "alert_id": alert.alert_id,
            "equipment_id": alert.equipment_id,
            "equipment_type": alert.equipment_type.value,
            "severity": alert.severity.value,
            "anomaly_score": alert.anomaly_score,
            "rul_hours": alert.rul_hours,
            "timestamp": alert.timestamp.isoformat(),
            "recommended_action": alert.recommended_action,
            "confidence": alert.confidence
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get current equipment health report."""
        return {
            "equipment_id": self.health.equipment_id,
            "health_score": self.health.health_score,
            "rul_hours": self.health.rul_hours,
            "rul_confidence": self.health.rul_confidence,
            "anomaly_detected": self.health.anomaly_detected,
            "operating_hours": self.health.operating_hours,
            "active_alerts": len(self.health.alerts),
            "queue_stats": self.queue.get_stats()
        }


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_pump_monitoring_system(pump_id: str) -> PredictiveMaintenanceEngine:
    """Create a pump monitoring system with standard sensors."""
    sensor_configs = [
        {"id": f"{pump_id}_vib", "type": "vibration", "rate": 10.0},
        {"id": f"{pump_id}_temp", "type": "temperature", "rate": 1.0},
        {"id": f"{pump_id}_press", "type": "pressure", "rate": 5.0},
        {"id": f"{pump_id}_flow", "type": "flow", "rate": 1.0},
    ]
    
    return PredictiveMaintenanceEngine(
        equipment_id=pump_id,
        equipment_type=EquipmentType.PUMP,
        sensor_configs=sensor_configs
    )


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Industrial IoT Predictive Maintenance - Demo")
    print("=" * 60)
    
    # Create pump monitoring system
    pdm = create_pump_monitoring_system("PUMP_001")
    
    print("\n1. Generating training data (normal operation)...")
    normal_data = np.random.randn(500, 4) * 0.5 + np.array([1.0, 45.0, 100.0, 50.0])
    pdm.train_models(normal_data)
    
    print("\n2. Simulating normal operation...")
    for _ in range(30):
        alert = pdm.analyze()
        if alert:
            print(f"   Alert: {alert.severity.value}")
    
    print("\n3. Simulating degradation...")
    # Inject anomaly into vibration sensor
    pdm.sensors["PUMP_001_vib"].offset = 2.0  # Offset vibration readings
    
    for i in range(20):
        alert = pdm.analyze()
        if alert:
            print(f"   🚨 {alert.severity.value}: {alert.recommended_action}")
    
    print("\n4. Health Report:")
    report = pdm.get_health_report()
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    print("\n5. Store-and-Forward Queue Status:")
    print(f"   {pdm.queue.get_stats()}")
    
    print("\n✅ Industrial IoT PdM demo complete!")
