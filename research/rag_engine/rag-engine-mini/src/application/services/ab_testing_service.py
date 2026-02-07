"""
A/B Testing Service

This module implements A/B testing functionality for the RAG Engine,
allowing experimentation with different models, prompts, and algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid
import random
import hashlib
from scipy import stats
import numpy as np


class ExperimentStatus(str, Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class VariantType(str, Enum):
    """Types of variants in an A/B test."""
    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class ExperimentVariant:
    """Represents a variant in an A/B test."""
    name: str
    description: str
    traffic_split: float  # Percentage of traffic (0.0 to 1.0)
    config: Dict[str, Any]  # Configuration for this variant
    variant_type: VariantType


@dataclass
class ABExperiment:
    """Represents an A/B test experiment."""
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    variants: List[ExperimentVariant]
    metrics: List[str]  # Metrics to track
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    hypothesis: Optional[str] = None
    owner: Optional[str] = None
    sample_size: Optional[int] = None
    statistical_power: Optional[float] = None  # Typically 0.8
    significance_level: Optional[float] = None  # Typically 0.05


@dataclass
class ExperimentAssignment:
    """Result of assigning a user/session to an experiment variant."""
    experiment_id: str
    user_id: str
    variant_name: str
    assigned_at: datetime
    context: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentResult:
    """Result of an A/B test experiment."""
    experiment_id: str
    variant_results: Dict[str, Dict[str, Any]]  # variant_name -> metrics
    statistical_significance: Dict[str, Dict[str, float]]  # metric -> {p_value, confidence_interval}
    winner: Optional[str] = None
    conclusion: Optional[str] = None
    is_significant: Optional[bool] = None


class ABTestingServicePort(ABC):
    """Abstract port for A/B testing services."""

    @abstractmethod
    async def create_experiment(self, experiment: ABExperiment) -> ABExperiment:
        """Create a new A/B test experiment."""
        pass

    @abstractmethod
    async def get_experiment(self, experiment_id: str) -> Optional[ABExperiment]:
        """Get details of a specific experiment."""
        pass

    @abstractmethod
    async def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[ABExperiment]:
        """List A/B test experiments."""
        pass

    @abstractmethod
    async def assign_variant(self, experiment_id: str, user_id: str, context: Optional[Dict[str, Any]] = None) -> ExperimentAssignment:
        """Assign a user to a variant in an experiment."""
        pass

    @abstractmethod
    async def track_event(self, experiment_id: str, user_id: str, variant_name: str, event_type: str, value: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """Track an event for an experiment."""
        pass

    @abstractmethod
    async def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get results of an A/B test experiment."""
        pass

    @abstractmethod
    async def update_experiment_status(self, experiment_id: str, status: ExperimentStatus) -> ABExperiment:
        """Update the status of an experiment."""
        pass

    @abstractmethod
    async def calculate_sample_size(self, baseline_conversion_rate: float, minimum_detectable_effect: float, significance_level: float = 0.05, power: float = 0.8) -> int:
        """Calculate required sample size for an experiment."""
        pass


class ABTestingService(ABTestingServicePort):
    """Concrete implementation of the A/B testing service."""

    def __init__(self):
        # In a real implementation, this would connect to a database
        # For now, we'll use in-memory storage for demonstration
        self._experiments: Dict[str, ABExperiment] = {}
        self._assignments: Dict[str, List[ExperimentAssignment]] = {}  # experiment_id -> assignments
        self._events: Dict[str, List[Dict[str, Any]]] = {}  # experiment_id -> events
        self._variant_assignments: Dict[str, str] = {}  # hash(user_id + experiment_id) -> variant_name

    async def create_experiment(self, experiment: ABExperiment) -> ABExperiment:
        """Create a new A/B test experiment."""
        # Validate experiment
        if not experiment.experiment_id:
            experiment.experiment_id = str(uuid.uuid4())
        
        if not experiment.created_at:
            experiment.created_at = datetime.now()
        
        experiment.updated_at = datetime.now()
        
        # Validate traffic splits sum to 1.0
        total_traffic = sum(variant.traffic_split for variant in experiment.variants)
        if abs(total_traffic - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Traffic splits must sum to 1.0, got {total_traffic}")
        
        # Set experiment status to draft initially
        if not experiment.status:
            experiment.status = ExperimentStatus.DRAFT
        
        self._experiments[experiment.experiment_id] = experiment
        self._assignments[experiment.experiment_id] = []
        self._events[experiment.experiment_id] = []
        
        return experiment

    async def get_experiment(self, experiment_id: str) -> Optional[ABExperiment]:
        """Get details of a specific experiment."""
        return self._experiments.get(experiment_id)

    async def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[ABExperiment]:
        """List A/B test experiments."""
        experiments = list(self._experiments.values())
        
        if status:
            experiments = [exp for exp in experiments if exp.status == status]
        
        # Sort by creation date, newest first
        experiments.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        
        return experiments

    async def assign_variant(self, experiment_id: str, user_id: str, context: Optional[Dict[str, Any]] = None) -> ExperimentAssignment:
        """Assign a user to a variant in an experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if experiment.status != ExperimentStatus.ACTIVE:
            raise ValueError(f"Experiment {experiment_id} is not active")
        
        # Create a deterministic hash to ensure consistent assignment
        hash_input = f"{user_id}:{experiment_id}".encode('utf-8')
        hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16) % 1000000
        percentage = hash_value / 10000.0  # Convert to 0-100 percentage
        
        # Find which variant this user should be assigned to based on traffic splits
        cumulative_percentage = 0.0
        selected_variant = None
        
        for variant in experiment.variants:
            cumulative_percentage += variant.traffic_split
            if percentage <= cumulative_percentage * 100:  # Convert back to percentage
                selected_variant = variant
                break
        
        if not selected_variant:
            # Fallback to last variant in case of rounding errors
            selected_variant = experiment.variants[-1]
        
        assignment = ExperimentAssignment(
            experiment_id=experiment_id,
            user_id=user_id,
            variant_name=selected_variant.name,
            assigned_at=datetime.now(),
            context=context
        )
        
        self._assignments[experiment_id].append(assignment)
        
        return assignment

    async def track_event(self, experiment_id: str, user_id: str, variant_name: str, event_type: str, value: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """Track an event for an experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Verify the variant exists in this experiment
        variant_exists = any(v.name == variant_name for v in experiment.variants)
        if not variant_exists:
            raise ValueError(f"Variant {variant_name} not found in experiment {experiment_id}")
        
        event = {
            "experiment_id": experiment_id,
            "user_id": user_id,
            "variant_name": variant_name,
            "event_type": event_type,
            "value": value,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        
        self._events[experiment_id].append(event)

    async def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get results of an A/B test experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return None
        
        # Group events by variant
        variant_events = {}
        for event in self._events.get(experiment_id, []):
            variant_name = event["variant_name"]
            if variant_name not in variant_events:
                variant_events[variant_name] = []
            variant_events[variant_name].append(event)
        
        # Calculate metrics for each variant
        variant_results = {}
        for variant_name, events in variant_events.items():
            # Calculate common metrics
            total_events = len(events)
            
            # Calculate conversion rate if we have binary events (success/failure)
            success_events = [e for e in events if e.get("event_type") in ["conversion", "success", "positive"]]
            conversion_rate = len(success_events) / total_events if total_events > 0 else 0.0
            
            # Calculate average value if numeric values are present
            numeric_values = [e["value"] for e in events if e.get("value") is not None]
            avg_value = sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
            
            variant_results[variant_name] = {
                "total_events": total_events,
                "conversion_rate": conversion_rate,
                "average_value": avg_value,
                "numeric_samples": numeric_values
            }
        
        # Perform statistical analysis if we have enough data
        statistical_significance = {}
        winner = None
        is_significant = False
        
        if len(variant_results) >= 2:
            # Perform t-test between variants for numeric metrics
            for metric in ["conversion_rate", "average_value"]:
                if all(variant_results[v].get("numeric_samples") is not None for v in variant_results.keys()):
                    # Get samples for each variant
                    samples = {}
                    for variant_name, results in variant_results.items():
                        samples[variant_name] = results.get("numeric_samples", [])
                    
                    # Perform t-test between first variant (control) and others
                    control_variant = experiment.variants[0].name
                    control_samples = samples.get(control_variant, [])
                    
                    for treatment_variant, treatment_samples in samples.items():
                        if treatment_variant == control_variant:
                            continue
                        
                        if len(control_samples) > 1 and len(treatment_samples) > 1:
                            try:
                                _, p_value = stats.ttest_ind(control_samples, treatment_samples)
                                # Calculate confidence intervals
                                control_mean = np.mean(control_samples) if control_samples else 0
                                treatment_mean = np.mean(treatment_samples) if treatment_samples else 0
                                
                                # Calculate effect size
                                pooled_std = np.sqrt(((len(control_samples)-1)*np.var(control_samples, ddof=1) + 
                                                     (len(treatment_samples)-1)*np.var(treatment_samples, ddof=1)) / 
                                                    (len(control_samples) + len(treatment_samples) - 2))
                                effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std != 0 else 0
                                
                                if f"{metric}_vs_{control_variant}" not in statistical_significance:
                                    statistical_significance[f"{metric}_vs_{control_variant}"] = {}
                                
                                statistical_significance[f"{metric}_vs_{control_variant}"][treatment_variant] = {
                                    "p_value": float(p_value),
                                    "significant": float(p_value) < 0.05,  # 95% confidence
                                    "effect_size": float(effect_size),
                                    "control_mean": float(control_mean),
                                    "treatment_mean": float(treatment_mean)
                                }
                                
                                # Check if this difference is significant
                                if float(p_value) < 0.05:
                                    is_significant = True
                                    
                                    # Determine winner based on higher mean
                                    if treatment_mean > control_mean:
                                        winner = treatment_variant
                                    else:
                                        winner = control_variant
                            except Exception:
                                # If statistical test fails, skip it
                                continue
        
        return ExperimentResult(
            experiment_id=experiment_id,
            variant_results=variant_results,
            statistical_significance=statistical_significance,
            winner=winner,
            is_significant=is_significant,
            conclusion=self._generate_conclusion(winner, is_significant, statistical_significance)
        )

    def _generate_conclusion(self, winner: Optional[str], is_significant: bool, stats: Dict[str, Any]) -> Optional[str]:
        """Generate a human-readable conclusion for the experiment results."""
        if not is_significant:
            return "No statistically significant difference found between variants."
        
        if winner:
            # Find the largest effect size to determine the strength of the result
            max_effect_size = 0
            for metric_stats in stats.values():
                for variant_stats in metric_stats.values():
                    if isinstance(variant_stats, dict) and "effect_size" in variant_stats:
                        max_effect_size = max(max_effect_size, abs(variant_stats["effect_size"]))
            
            strength = "small" if max_effect_size < 0.2 else "medium" if max_effect_size < 0.8 else "large"
            return f"Variant '{winner}' is the winner with {strength} effect size and statistical significance."
        
        return "Statistically significant result found but no clear winner identified."

    async def update_experiment_status(self, experiment_id: str, status: ExperimentStatus) -> ABExperiment:
        """Update the status of an experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        old_status = experiment.status
        experiment.status = status
        experiment.updated_at = datetime.now()
        
        # Set end date if experiment is completed
        if status == ExperimentStatus.COMPLETED and old_status != ExperimentStatus.COMPLETED:
            experiment.end_date = datetime.now()
        
        return experiment

    async def calculate_sample_size(self, baseline_conversion_rate: float, minimum_detectable_effect: float, significance_level: float = 0.05, power: float = 0.8) -> int:
        """Calculate required sample size for an experiment using power analysis."""
        # This is a simplified calculation
        # In a real implementation, we would use more sophisticated power analysis
        
        # Effect size calculation (Cohen's h for proportions)
        p1 = baseline_conversion_rate
        p2 = baseline_conversion_rate + minimum_detectable_effect
        effect_size = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        
        # Approximate sample size using Lehr's rule (for 2-sample t-test)
        # n = 16 * (std_dev)^2 / (mean_difference)^2
        # For proportions, we can approximate:
        z_alpha = stats.norm.ppf(1 - significance_level/2)  # Two-tailed test
        z_beta = stats.norm.ppf(power)
        
        # Calculate sample size per group
        numerator = (z_alpha + z_beta) ** 2
        denominator = effect_size ** 2
        sample_size_per_group = int(np.ceil(numerator / denominator))
        
        # Total sample size is doubled since we have 2 groups
        total_sample_size = sample_size_per_group * 2
        
        return total_sample_size