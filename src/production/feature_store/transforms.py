"""
Feature Transforms Module
==========================

Abstract base class and implementations for feature transformations.

Classes:
    FeatureTransform: Abstract base class for feature transformations
    AvgOrderValueTransform: Example batch transform
    SessionCountTransform: Example streaming transform

Author: AI-Mastery-2026
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from .types import FeatureDefinition


class FeatureTransform(ABC):
    """
    Abstract base class for feature transformations.

    Implement this to define how a feature is computed.
    """

    @property
    @abstractmethod
    def feature_definition(self) -> FeatureDefinition:
        """Return the feature definition."""
        pass

    @abstractmethod
    def compute(
        self, entity_id: str, raw_data: Dict[str, Any], dependencies: Dict[str, Any]
    ) -> Any:
        """
        Compute the feature value.

        Args:
            entity_id: ID of the entity
            raw_data: Raw input data
            dependencies: Values of dependent features

        Returns:
            Computed feature value
        """
        pass


# ============================================================
# EXAMPLE TRANSFORMS
# ============================================================


class AvgOrderValueTransform(FeatureTransform):
    """Example: Average order value for a user."""

    def __init__(self, window_days: int = 30):
        self.window_days = window_days

    @property
    def feature_definition(self) -> FeatureDefinition:
        return FeatureDefinition(
            name="user_avg_order_value",
            feature_type="numeric",
            computation_type="batch",
            description=f"Average order value in last {self.window_days} days",
            entity_key="user",
            ttl_seconds=86400,
        )

    def compute(
        self, entity_id: str, raw_data: Dict[str, Any], dependencies: Dict[str, Any]
    ) -> float:
        orders = raw_data.get("orders", [])
        if not orders:
            return 0.0
        return float(np.mean([o.get("value", 0) for o in orders]))


class SessionCountTransform(FeatureTransform):
    """Example: Session count in last 7 days."""

    @property
    def feature_definition(self) -> FeatureDefinition:
        return FeatureDefinition(
            name="user_session_count_7d",
            feature_type="numeric",
            computation_type="streaming",
            description="Number of sessions in last 7 days",
            entity_key="user",
            ttl_seconds=3600,
        )

    def compute(
        self, entity_id: str, raw_data: Dict[str, Any], dependencies: Dict[str, Any]
    ) -> int:
        sessions = raw_data.get("sessions", [])
        return len(sessions)


class UserEngagementScoreTransform(FeatureTransform):
    """Example: Composite engagement score."""

    @property
    def feature_definition(self) -> FeatureDefinition:
        return FeatureDefinition(
            name="user_engagement_score",
            feature_type="numeric",
            computation_type="batch",
            description="Composite engagement score from multiple signals",
            entity_key="user",
            dependencies=["user_avg_order_value", "user_session_count_7d"],
            ttl_seconds=86400,
        )

    def compute(
        self, entity_id: str, raw_data: Dict[str, Any], dependencies: Dict[str, Any]
    ) -> float:
        # Use dependent feature values
        avg_order = dependencies.get("user_avg_order_value", 0)
        session_count = dependencies.get("user_session_count_7d", 0)

        # Composite score (example formula)
        score = (avg_order * 0.6) + (session_count * 10 * 0.4)
        return float(score)


class ItemPopularityTransform(FeatureTransform):
    """Example: Item popularity score."""

    def __init__(self, decay_factor: float = 0.9):
        self.decay_factor = decay_factor

    @property
    def feature_definition(self) -> FeatureDefinition:
        return FeatureDefinition(
            name="item_popularity_score",
            feature_type="numeric",
            computation_type="streaming",
            description="Time-decayed popularity score",
            entity_key="item",
            ttl_seconds=3600,
        )

    def compute(
        self, entity_id: str, raw_data: Dict[str, Any], dependencies: Dict[str, Any]
    ) -> float:
        views = raw_data.get("views", [])
        purchases = raw_data.get("purchases", 0)

        # Simple popularity: weighted views + purchases
        view_score = len(views) * self.decay_factor
        purchase_score = purchases * 2.0

        return float(view_score + purchase_score)
