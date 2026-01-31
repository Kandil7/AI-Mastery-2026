"""
A/B Testing Routes
==================
API endpoints for A/B testing experiments.

نقاط نهاية API لتجارب A/B
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.v1.deps import get_tenant_id
from src.core.bootstrap import get_container
from src.application.services.ab_testing import (
    ABExperiment,
    ABVariant,
    ABExperimentConfig,
)

router = APIRouter(tags=["ab_testing"])


class CreateExperimentRequest(BaseModel):
    """Request model for creating experiments."""

    name: str
    description: Optional[str] = None
    variants: List[dict]
    traffic_split: Optional[List[float]] = None
    target_metric: str = "engagement"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class UpdateExperimentRequest(BaseModel):
    """Request model for updating experiments."""

    status: Optional[str] = None  # active, paused, completed
    traffic_split: Optional[List[float]] = None
    end_date: Optional[datetime] = None


class RecordConversionRequest(BaseModel):
    """Request model for recording conversions."""

    experiment_id: str
    variant_id: str
    success: bool
    value: Optional[float] = None


@router.get("/experiments")
async def list_experiments(
    tenant_id: str = Depends(get_tenant_id),
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    List A/B experiments.

    Args:
        tenant_id: Tenant/user ID from auth
        status: Filter by status (optional)
        limit: Max results
        offset: Pagination offset

    Returns:
        List of experiments with pagination metadata

    قائمة تجارب A/B
    """
    # Get service
    container = get_container()
    ab_service = container.get("ab_testing_service")

    # Get experiments
    experiments = ab_service.list_experiments(
        tenant_id=tenant_id,
        status=status,
        limit=limit,
        offset=offset,
    )

    return {
        "experiments": experiments,
        "count": len(experiments),
        "limit": limit,
        "offset": offset,
    }


@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Get A/B experiment details.

    Args:
        experiment_id: Experiment ID
        tenant_id: Tenant/user ID from auth

    Returns:
        Experiment with variants and metrics

    الحصول على تفاصيل تجربة A/B
    """
    # Get service
    container = get_container()
    ab_service = container.get("ab_testing_service")

    # Get experiment
    experiment = ab_service.get_experiment(
        experiment_id=experiment_id,
        tenant_id=tenant_id,
    )

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return experiment


@router.post("/experiments")
async def create_experiment(
    request: CreateExperimentRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Create a new A/B experiment.

    Args:
        request: Experiment creation request
        tenant_id: Tenant/user ID from auth

    Returns:
        Created experiment

    إنشاء تجربة A/B جديدة
    """
    # Get service
    container = get_container()
    ab_service = container.get("ab_testing_service")

    # Build experiment config
    variants = [
        ABVariant(
            id=v.get("id"),
            name=v.get("name"),
            config=v.get("config", {}),
        )
        for v in request.variants
    ]

    config = ABExperimentConfig(
        variants=variants,
        traffic_split=request.traffic_split,
        target_metric=request.target_metric,
    )

    # Create experiment
    experiment = ab_service.create_experiment(
        tenant_id=tenant_id,
        name=request.name,
        description=request.description,
        config=config,
        start_date=request.start_date,
        end_date=request.end_date,
    )

    return experiment


@router.put("/experiments/{experiment_id}")
async def update_experiment(
    experiment_id: str,
    request: UpdateExperimentRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Update an A/B experiment.

    Args:
        experiment_id: Experiment ID
        request: Update request
        tenant_id: Tenant/user ID from auth

    Returns:
        Updated experiment

    تحديث تجربة A/B
    """
    # Get service
    container = get_container()
    ab_service = container.get("ab_testing_service")

    # Update experiment
    updated = ab_service.update_experiment(
        experiment_id=experiment_id,
        tenant_id=tenant_id,
        status=request.status,
        traffic_split=request.traffic_split,
        end_date=request.end_date,
    )

    if not updated:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return updated


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Delete an A/B experiment.

    Args:
        experiment_id: Experiment ID
        tenant_id: Tenant/user ID from auth

    Returns:
        Deletion confirmation

    حذف تجربة A/B
    """
    # Get service
    container = get_container()
    ab_service = container.get("ab_testing_service")

    # Delete experiment
    deleted = ab_service.delete_experiment(
        experiment_id=experiment_id,
        tenant_id=tenant_id,
    )

    if not deleted:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {"experiment_id": experiment_id, "deleted": True}


@router.post("/experiments/{experiment_id}/assign")
async def assign_variant(
    experiment_id: str,
    user_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Assign user to a variant (deterministic).

    Args:
        experiment_id: Experiment ID
        user_id: User ID to assign
        tenant_id: Tenant/user ID from auth

    Returns:
        Assigned variant

    تعيين مستخدم لنسخة
    """
    # Get service
    container = get_container()
    ab_service = container.get("ab_testing_service")

    # Get experiment
    experiment = ab_service.get_experiment(
        experiment_id=experiment_id,
        tenant_id=tenant_id,
    )

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Assign variant
    variant_id = ab_service.assign_variant(
        experiment_id=experiment_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )

    if not variant_id:
        raise HTTPException(status_code=400, detail="Failed to assign variant")

    # Get variant details
    variant = next(
        (v for v in experiment.get("variants", []) if v["id"] == variant_id),
        None,
    )

    if not variant:
        raise HTTPException(status_code=404, detail="Variant not found")

    return {
        "experiment_id": experiment_id,
        "variant_id": variant_id,
        "variant_name": variant.get("name"),
        "variant_config": variant.get("config", {}),
    }


@router.post("/experiments/{experiment_id}/record")
async def record_conversion(
    experiment_id: str,
    request: RecordConversionRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Record a conversion event.

    Args:
        experiment_id: Experiment ID
        request: Conversion recording request
        tenant_id: Tenant/user ID from auth

    Returns:
        Recording confirmation

    تسجيل حدث تحويل
    """
    # Get service
    container = get_container()
    ab_service = container.get("ab_testing_service")

    # Record conversion
    recorded = ab_service.record_conversion(
        experiment_id=experiment_id,
        tenant_id=tenant_id,
        variant_id=request.variant_id,
        success=request.success,
        value=request.value,
    )

    if not recorded:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "experiment_id": experiment_id,
        "variant_id": request.variant_id,
        "recorded": True,
    }


@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Get A/B experiment results with statistical analysis.

    Args:
        experiment_id: Experiment ID
        tenant_id: Tenant/user ID from auth

    Returns:
        Experiment results with metrics and statistical significance

    الحصول على نتائج تجربة A/B
    """
    # Get service
    container = get_container()
    ab_service = container.get("ab_testing_service")

    # Get experiment
    experiment = ab_service.get_experiment(
        experiment_id=experiment_id,
        tenant_id=tenant_id,
    )

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Get results
    results = ab_service.get_results(
        experiment_id=experiment_id,
        tenant_id=tenant_id,
    )

    return results
