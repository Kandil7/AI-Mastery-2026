"""
Medical Edge AI Type Definitions
================================

Core data structures and enums for medical IoMT systems.

Enums:
    MedicalDeviceType: Types of medical IoMT devices
    ClinicalEventType: Types of clinical events detected by edge AI
    AlertSeverity: Clinical alert severity levels

Dataclasses:
    ClinicalEvent: Clinical event detected by edge AI
    PrivacyBudget: Differential privacy budget tracker
    FederatedUpdate: Model update from federated client
    HealthTrain: Personal Health Train - algorithm that travels to data
    HealthStation: Data repository (hospital, device)

Author: AI-Mastery-2026
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


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


@dataclass
class FederatedUpdate:
    """
    Model update from a federated client.

    Contains gradients/weights, not raw data.
    """

    client_id: str
    round_number: int
    gradients: Any  # np.ndarray
    sample_count: int
    timestamp: datetime
    encrypted: bool = False
    signature: str = ""


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


# Type aliases
EventHistory = List[ClinicalEvent]
PrivacyReport = Dict[str, Any]
DeviceConfig = Dict[str, Any]
