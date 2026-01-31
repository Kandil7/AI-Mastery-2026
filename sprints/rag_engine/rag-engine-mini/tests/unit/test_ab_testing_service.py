"""
Tests for A/B Testing Service

This module tests the A/B testing service functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.application.services.ab_testing_service import (
    ABTestingService,
    ABExperiment,
    ExperimentVariant,
    VariantType,
    ExperimentStatus
)


@pytest.fixture
def ab_testing_service():
    """Create an A/B testing service instance for testing."""
    return ABTestingService()


@pytest.fixture
def sample_experiment():
    """Create a sample experiment for testing."""
    return ABExperiment(
        experiment_id="test-exp-1",
        name="Test Experiment",
        description="A test experiment for A/B testing",
        status=ExperimentStatus.DRAFT,
        variants=[
            ExperimentVariant(
                name="control",
                description="Control group",
                traffic_split=0.5,
                config={"model": "gpt-3.5-turbo"},
                variant_type=VariantType.CONTROL
            ),
            ExperimentVariant(
                name="treatment",
                description="Treatment group",
                traffic_split=0.5,
                config={"model": "gpt-4"},
                variant_type=VariantType.TREATMENT
            )
        ],
        metrics=["conversion_rate", "response_time"],
        created_by="test-user"
    )


@pytest.mark.asyncio
async def test_create_experiment(ab_testing_service, sample_experiment):
    """Test creating a new A/B test experiment."""
    created_exp = await ab_testing_service.create_experiment(sample_experiment)
    
    assert created_exp.experiment_id == sample_experiment.experiment_id
    assert created_exp.name == sample_experiment.name
    assert created_exp.status == ExperimentStatus.DRAFT
    assert len(created_exp.variants) == 2
    assert created_exp.created_at is not None
    assert created_exp.updated_at is not None


@pytest.mark.asyncio
async def test_get_experiment(ab_testing_service, sample_experiment):
    """Test getting a specific experiment."""
    await ab_testing_service.create_experiment(sample_experiment)
    
    retrieved_exp = await ab_testing_service.get_experiment(sample_experiment.experiment_id)
    
    assert retrieved_exp is not None
    assert retrieved_exp.experiment_id == sample_experiment.experiment_id
    assert retrieved_exp.name == sample_experiment.name


@pytest.mark.asyncio
async def test_list_experiments(ab_testing_service, sample_experiment):
    """Test listing experiments."""
    # Create multiple experiments
    exp1 = sample_experiment
    exp2 = ABExperiment(
        experiment_id="test-exp-2",
        name="Test Experiment 2",
        description="Second test experiment",
        status=ExperimentStatus.ACTIVE,
        variants=sample_experiment.variants,
        metrics=sample_experiment.metrics,
        created_by="test-user"
    )
    
    await ab_testing_service.create_experiment(exp1)
    await ab_testing_service.create_experiment(exp2)
    
    # Test listing all experiments
    all_experiments = await ab_testing_service.list_experiments()
    assert len(all_experiments) == 2
    
    # Test filtering by status
    active_experiments = await ab_testing_service.list_experiments(status=ExperimentStatus.ACTIVE)
    assert len(active_experiments) == 1
    assert active_experiments[0].experiment_id == "test-exp-2"


@pytest.mark.asyncio
async def test_assign_variant(ab_testing_service, sample_experiment):
    """Test assigning users to variants."""
    await ab_testing_service.create_experiment(sample_experiment)
    
    # Update experiment to active status
    await ab_testing_service.update_experiment_status(sample_experiment.experiment_id, ExperimentStatus.ACTIVE)
    
    # Assign a user to a variant
    assignment = await ab_testing_service.assign_variant(
        experiment_id=sample_experiment.experiment_id,
        user_id="test-user-123"
    )
    
    assert assignment.experiment_id == sample_experiment.experiment_id
    assert assignment.user_id == "test-user-123"
    assert assignment.variant_name in ["control", "treatment"]
    assert assignment.assigned_at is not None


@pytest.mark.asyncio
async def test_track_event(ab_testing_service, sample_experiment):
    """Test tracking events for an experiment."""
    await ab_testing_service.create_experiment(sample_experiment)
    
    # Update experiment to active status
    await ab_testing_service.update_experiment_status(sample_experiment.experiment_id, ExperimentStatus.ACTIVE)
    
    # Assign a user to a variant first
    assignment = await ab_testing_service.assign_variant(
        experiment_id=sample_experiment.experiment_id,
        user_id="test-user-456"
    )
    
    # Track an event
    await ab_testing_service.track_event(
        experiment_id=sample_experiment.experiment_id,
        user_id="test-user-456",
        variant_name=assignment.variant_name,
        event_type="conversion",
        value=1.0,
        metadata={"page": "homepage"}
    )
    
    # In a real implementation, we would verify the event was tracked
    # For now, just ensure no exceptions are thrown


@pytest.mark.asyncio
async def test_get_experiment_results(ab_testing_service, sample_experiment):
    """Test getting experiment results."""
    await ab_testing_service.create_experiment(sample_experiment)
    
    # Update experiment to active status
    await ab_testing_service.update_experiment_status(sample_experiment.experiment_id, ExperimentStatus.ACTIVE)
    
    # Assign users and track events
    for i in range(10):
        user_id = f"test-user-{i}"
        assignment = await ab_testing_service.assign_variant(
            experiment_id=sample_experiment.experiment_id,
            user_id=user_id
        )
        
        # Track conversion events with different values
        event_type = "conversion" if i % 3 == 0 else "view"
        value = 1.0 if i % 3 == 0 else None
        await ab_testing_service.track_event(
            experiment_id=sample_experiment.experiment_id,
            user_id=user_id,
            variant_name=assignment.variant_name,
            event_type=event_type,
            value=value,
            metadata={"test": True}
        )
    
    # Get results
    results = await ab_testing_service.get_experiment_results(sample_experiment.experiment_id)
    
    assert results is not None
    assert results.experiment_id == sample_experiment.experiment_id
    assert len(results.variant_results) >= 1  # At least one variant should have results


@pytest.mark.asyncio
async def test_update_experiment_status(ab_testing_service, sample_experiment):
    """Test updating experiment status."""
    await ab_testing_service.create_experiment(sample_experiment)
    
    # Update status
    updated_exp = await ab_testing_service.update_experiment_status(
        sample_experiment.experiment_id,
        ExperimentStatus.ACTIVE
    )
    
    assert updated_exp.status == ExperimentStatus.ACTIVE
    assert updated_exp.updated_at > sample_experiment.created_at


@pytest.mark.asyncio
async def test_calculate_sample_size(ab_testing_service):
    """Test calculating required sample size."""
    # Test with typical values
    sample_size = await ab_testing_service.calculate_sample_size(
        baseline_conversion_rate=0.10,  # 10% baseline
        minimum_detectable_effect=0.02,  # Detect 2% absolute improvement (10% -> 12%)
        significance_level=0.05,
        power=0.8
    )
    
    assert sample_size > 0
    assert isinstance(sample_size, int)


@pytest.mark.asyncio
async def test_invalid_experiment_creation(ab_testing_service):
    """Test creating an experiment with invalid traffic splits."""
    # Create an experiment with invalid traffic splits (don't sum to 1.0)
    invalid_experiment = ABExperiment(
        experiment_id="invalid-exp",
        name="Invalid Experiment",
        description="Experiment with invalid traffic splits",
        status=ExperimentStatus.DRAFT,
        variants=[
            ExperimentVariant(
                name="control",
                description="Control group",
                traffic_split=0.4,  # This makes the sum 0.4 + 0.3 = 0.7 != 1.0
                config={"model": "gpt-3.5-turbo"},
                variant_type=VariantType.CONTROL
            ),
            ExperimentVariant(
                name="treatment",
                description="Treatment group",
                traffic_split=0.3,  # Sum is 0.7, not 1.0
                config={"model": "gpt-4"},
                variant_type=VariantType.TREATMENT
            )
        ],
        metrics=["conversion_rate"],
        created_by="test-user"
    )
    
    with pytest.raises(ValueError, match="Traffic splits must sum to 1.0"):
        await ab_testing_service.create_experiment(invalid_experiment)


@pytest.mark.asyncio
async def test_assign_variant_nonexistent_experiment(ab_testing_service):
    """Test assigning variant to a non-existent experiment."""
    with pytest.raises(ValueError, match="not found"):
        await ab_testing_service.assign_variant(
            experiment_id="nonexistent-exp",
            user_id="test-user"
        )


@pytest.mark.asyncio
async def test_track_event_nonexistent_experiment(ab_testing_service):
    """Test tracking event for a non-existent experiment."""
    with pytest.raises(ValueError, match="not found"):
        await ab_testing_service.track_event(
            experiment_id="nonexistent-exp",
            user_id="test-user",
            variant_name="control",
            event_type="conversion"
        )