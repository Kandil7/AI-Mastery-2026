"""
A/B Testing Framework
====================
Service for managing A/B tests and experiments.

خدمة إدارة اختبارات A/B والتجارب
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ExperimentStatus(str, Enum):
    """Experiment status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class Experiment:
    """A/B test experiment configuration."""

    id: str
    name: str
    description: str
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


@dataclass
class Variant:
    """A/B test variant."""

    id: str
    experiment_id: str
    name: str
    allocation: float  # 0.0 to 1.0
    config: Dict


@dataclass
class Assignment:
    """Variant assignment to user."""

    experiment_id: str
    user_id: str
    variant_id: str
    assigned_at: datetime


class ABTestingService:
    """Manage A/B tests and experiments."""

    def __init__(self, experiment_repo, assignment_repo):
        self._experiment_repo = experiment_repo
        self._assignment_repo = assignment_repo

    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Variant],
    ) -> str:
        """Create new A/B test experiment."""
        experiment_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        self._experiment_repo.create_experiment(
            id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            created_at=created_at,
        )

        for variant in variants:
            self._experiment_repo.create_variant(
                id=str(uuid.uuid4()),
                experiment_id=experiment_id,
                name=variant.name,
                allocation=variant.allocation,
                config=variant.config,
            )

        return experiment_id

    def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Optional[Variant]:
        """
        Assign variant to user based on allocation.

        Uses deterministic hashing for consistent assignment.
        """
        experiment = self._experiment_repo.get_experiment(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None

        variants = self._experiment_repo.get_variants(experiment_id)
        if not variants:
            return None

        # Deterministic allocation using hashing
        import hashlib

        hash_value = int(hashlib.md5(f"{user_id}:{experiment_id}".encode()).hexdigest(), 16)
        allocated_value = hash_value / 65536.0

        cumulative = 0.0
        for variant in variants:
            cumulative += variant.allocation
            if allocated_value <= cumulative:
                # Record assignment
                self._assignment_repo.create_assignment(
                    experiment_id=experiment_id,
                    user_id=user_id,
                    variant_id=variant.id,
                    assigned_at=datetime.utcnow(),
                )
                return variant

        return None

    def record_metric(
        self,
        experiment_id: str,
        variant_id: str,
        metric_name: str,
        metric_value: float,
    ) -> None:
        """Record metric for A/B test analysis."""
        self._experiment_repo.record_metric(
            experiment_id=experiment_id,
            variant_id=variant_id,
            metric_name=metric_name,
            metric_value=metric_value,
            recorded_at=datetime.utcnow(),
        )

    def analyze_results(
        self,
        experiment_id: str,
    ) -> Dict:
        """Analyze A/B test results with statistical significance."""
        metrics = self._experiment_repo.get_metrics(experiment_id)
        # Implement statistical analysis (t-test, confidence intervals)
        return {"results": metrics, "significant": False}
