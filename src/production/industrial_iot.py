"""
Industrial IoT Predictive Maintenance Module
=============================================

Edge AI for predictive maintenance in oil & gas and industrial settings.

NOTE: This module is now a compatibility wrapper.
The implementation has been moved to the `industrial_iot` submodule.

New import style:
    from src.production.industrial_iot import (
        PredictiveMaintenanceEngine,
        EquipmentType,
        SensorType,
        AlertSeverity,
    )

This module maintains backward compatibility by re-exporting
from the submodule.

Author: AI-Mastery-2026
Version: 2.0.0 (modular)
"""

# Re-export all from submodule
from src.production.industrial_iot import (
    AlertHistory,
    AlertSeverity,
    AnomalyDetector,
    Autoencoder,
    EquipmentConfig,
    EquipmentHealth,
    EquipmentType,
    IsolationForest,
    LSTMCell,
    MaintenanceAlert,
    PredictiveMaintenanceEngine,
    ProtocolType,
    QueuedMessage,
    RULPredictor,
    RULPredictorWithDegradation,
    SensorArray,
    SensorData,
    SensorReading,
    SensorType,
    StoreAndForwardQueue,
    example_usage,
)

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
