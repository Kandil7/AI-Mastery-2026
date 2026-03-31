"""
Predictive Maintenance Engine Module
=====================================

Complete predictive maintenance solution.

Combines:
- Multi-sensor data ingestion
- Anomaly detection
- RUL prediction
- Alert management
- Store-and-forward communication

Classes:
    PredictiveMaintenanceEngine: Complete PdM solution

Author: AI-Mastery-2026
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .anomaly import AnomalyDetector
from .rul import RULPredictor
from .sensors import IndustrialSensor
from .store_forward import StoreAndForwardQueue
from .types import (
    AlertSeverity,
    EquipmentHealth,
    EquipmentType,
    MaintenanceAlert,
)

logger = logging.getLogger(__name__)


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
        sensor_configs: List[Dict[str, Any]],
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
                sensor_type=config["type"],
                protocol=config.get("protocol", "modbus_tcp"),
                address=config.get("address", "0"),
                sampling_rate_hz=config.get("rate", 1.0),
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
            rul_hours=float("inf"),
            rul_confidence=0.0,
            anomaly_detected=False,
            last_maintenance=None,
            operating_hours=0.0,
        )

        # History for RUL
        self.reading_history: deque = deque(maxlen=100)

        logger.info(
            f"PdM Engine initialized: {equipment_id} with {len(self.sensors)} sensors"
        )

    def train_models(
        self,
        normal_data: np.ndarray,
        rul_data: Optional[tuple] = None,
    ):
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
                values.append(0.0)

        feature_vector = np.array(values)
        self.reading_history.append(feature_vector)

        return feature_vector

    def analyze(self) -> Optional[MaintenanceAlert]:
        """Run full analysis: anomaly detection + RUL prediction."""
        current = self.collect_readings()

        if not self.anomaly_detector.trained:
            return None

        is_anomaly, score, component_scores = self.anomaly_detector.detect(current)

        rul_hours = float("inf")
        rul_lower = 0.0
        rul_upper = float("inf")

        if len(self.reading_history) >= 50 and self.rul_predictor.trained:
            sequence = np.array(list(self.reading_history)[-50:])
            rul_hours, rul_lower, rul_upper = self.rul_predictor.predict(sequence)

        self.health.anomaly_detected = is_anomaly
        self.health.rul_hours = rul_hours
        self.health.rul_confidence = 0.9 if rul_hours < float("inf") else 0.0

        if rul_hours < 24:
            self.health.health_score = max(0, 20 - score * 10)
        elif rul_hours < 168:
            self.health.health_score = max(20, 60 - score * 20)
        else:
            self.health.health_score = max(60, 100 - score * 30)

        if is_anomaly or rul_hours < 168:
            severity = self._determine_severity(is_anomaly, score, rul_hours)

            alert = MaintenanceAlert(
                alert_id=f"ALERT_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                equipment_id=self.equipment_id,
                equipment_type=self.equipment_type,
                severity=severity,
                anomaly_score=score,
                rul_hours=rul_hours if rul_hours < float("inf") else None,
                timestamp=datetime.now(),
                recommended_action=self._get_recommendation(severity, rul_hours),
                contributing_sensors=list(self.sensors.keys()),
                confidence=0.85,
                metadata={
                    "rul_lower": rul_lower,
                    "rul_upper": rul_upper,
                    "component_scores": component_scores,
                },
            )

            self.health.alerts.append(alert)

            priority = (
                0
                if severity == AlertSeverity.CRITICAL
                else 1
                if severity == AlertSeverity.HIGH
                else 2
            )
            self.queue.enqueue(self._alert_to_dict(alert), priority=priority)

            return alert

        return None

    def _determine_severity(
        self, is_anomaly: bool, score: float, rul_hours: float
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
            return (
                f"Schedule maintenance within 24 hours. Estimated RUL: {rul_hours:.0f}h"
            )
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
            "confidence": alert.confidence,
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
            "queue_stats": self.queue.get_stats(),
        }

    def get_alerts(
        self, severity: Optional[AlertSeverity] = None, limit: int = 100
    ) -> List[MaintenanceAlert]:
        """Get alerts, optionally filtered by severity."""
        alerts = self.health.alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:]

    def clear_alerts(self) -> int:
        """Clear all alerts and return count."""
        count = len(self.health.alerts)
        self.health.alerts.clear()
        return count


def example_usage():
    """Demonstrate Predictive Maintenance Engine usage."""
    # Create engine
    engine = PredictiveMaintenanceEngine(
        equipment_id="PUMP_001",
        equipment_type=EquipmentType.PUMP,
        sensor_configs=[
            {"id": "vib_1", "type": "vibration", "protocol": "modbus_tcp", "rate": 10},
            {
                "id": "temp_1",
                "type": "temperature",
                "protocol": "modbus_tcp",
                "rate": 1,
            },
            {"id": "press_1", "type": "pressure", "protocol": "modbus_tcp", "rate": 1},
        ],
    )

    # Generate synthetic training data
    normal_data = np.random.randn(100, 3) * 0.1 + np.array([0.5, 50, 100])
    engine.train_models(normal_data)

    # Run analysis
    for _ in range(10):
        alert = engine.analyze()
        if alert:
            print(f"Alert: {alert.severity.value} - {alert.recommended_action}")

    # Get health report
    report = engine.get_health_report()
    print(f"Health Score: {report['health_score']:.1f}")

    return engine


if __name__ == "__main__":
    example_usage()
