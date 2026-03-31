"""
Industrial IoT Type Definitions
================================

Core data structures and enums for industrial IoT systems.

Enums:
    EquipmentType: Types of industrial equipment
    SensorType: Types of industrial sensors
    AlertSeverity: Maintenance alert severity levels
    ProtocolType: Industrial communication protocols

Dataclasses:
    SensorReading: Time-series sensor reading
    MaintenanceAlert: Predictive maintenance alert
    EquipmentHealth: Overall equipment health status
    QueuedMessage: Message in store-and-forward queue

Author: AI-Mastery-2026
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


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


@dataclass
class QueuedMessage:
    """Message in store-and-forward queue."""

    message_id: str
    priority: int  # 0 = highest
    payload: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3


# Type aliases
SensorData = Dict[str, List[SensorReading]]
AlertHistory = List[MaintenanceAlert]
EquipmentConfig = Dict[str, Any]
