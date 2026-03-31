"""
Industrial IoT Predictive Maintenance Submodule
================================================

Edge AI for predictive maintenance in oil & gas and industrial settings.

Implements:
- Anomaly detection (Autoencoder, Isolation Forest)
- Remaining Useful Life (RUL) prediction (LSTM)
- Store-and-forward for DDIL environments
- Multi-protocol sensor integration

Usage:
    from src.production.industrial_iot import (
        PredictiveMaintenanceEngine,
        EquipmentType,
        SensorType,
        AlertSeverity,
    )

Example:
    >>> engine = PredictiveMaintenanceEngine(
    ...     equipment_id="PUMP_001",
    ...     equipment_type=EquipmentType.PUMP,
    ...     sensor_configs=[
    ...         {"id": "vib_1", "type": "vibration", "protocol": "modbus_tcp"}
    ...     ]
    ... )
    >>> engine.train_models(normal_data)
    >>> alert = engine.analyze()

Author: AI-Mastery-2026
Version: 2.0.0
"""

# Types
from .types import (
    AlertSeverity,
    EquipmentHealth,
    EquipmentType,
    MaintenanceAlert,
    ProtocolType,
    QueuedMessage,
    SensorReading,
    SensorType,
    AlertHistory,
    EquipmentConfig,
    SensorData,
)

# Store and forward
from .store_forward import StoreAndForwardQueue

# Sensors
from .sensors import IndustrialSensor, SensorArray

# Anomaly detection
from .anomaly import AnomalyDetector, Autoencoder, IsolationForest

# RUL prediction
from .rul import LSTMCell, RULPredictor, RULPredictorWithDegradation

# Main engine
from .engine import PredictiveMaintenanceEngine, example_usage

__all__ = [
    # Types
    "EquipmentType",
    "SensorType",
    "AlertSeverity",
    "ProtocolType",
    "SensorReading",
    "MaintenanceAlert",
    "EquipmentHealth",
    "QueuedMessage",
    # Type aliases
    "SensorData",
    "AlertHistory",
    "EquipmentConfig",
    # Store and forward
    "StoreAndForwardQueue",
    # Sensors
    "IndustrialSensor",
    "SensorArray",
    # Anomaly detection
    "Autoencoder",
    "IsolationForest",
    "AnomalyDetector",
    # RUL prediction
    "LSTMCell",
    "RULPredictor",
    "RULPredictorWithDegradation",
    # Main engine
    "PredictiveMaintenanceEngine",
    "example_usage",
]

__version__ = "2.0.0"
