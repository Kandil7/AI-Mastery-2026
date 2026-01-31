"""
A/B Testing API Routes

This module implements A/B testing functionality API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from ...domain.entities import TenantId
from ...application.services.ab_testing_service import (
    ABTestingService,
    ABExperiment,
    ExperimentVariant,
    ExperimentStatus,
    VariantType,
    ExperimentAssignment,
    ExperimentResult
)
from ..dependencies import get_container, get_tenant_id

router = APIRouter(prefix="/ab-testing", tags=["ab-testing"])


@router.post("/experiments")
async def create_experiment(
    experiment_data: Dict[str, Any] = Body(...),
    container: Dict = Depends(get_container),
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Create a new A/B test experiment.
    
    This endpoint creates an A/B test experiment with the specified parameters.
    """
    try:
        ab_test_service: ABTestingService = container.get("ab_test_service")
        if not ab_test_service:
            # If service not in container, create a temporary one for this demo
            ab_test_service = ABTestingService()
        
        # Construct the experiment object from the request data
        experiment = ABExperiment(
            experiment_id=experiment_data.get("experiment_id", str(uuid.uuid4())),
            name=experiment_data["name"],
            description=experiment_data["description"],
            status=ExperimentStatus(experiment_data.get("status", "draft")),
            variants=[
                ExperimentVariant(
                    name=v["name"],
                    description=v["description"],
                    traffic_split=v["traffic_split"],
                    config=v.get("config", {}),
                    variant_type=VariantType(v["variant_type"])
                ) for v in experiment_data["variants"]
            ],
            metrics=experiment_data.get("metrics", []),
            created_by=experiment_data.get("created_by"),
            hypothesis=experiment_data.get("hypothesis"),
            owner=experiment_data.get("owner")
        )
        
        created_experiment = await ab_test_service.create_experiment(experiment)
        
        return {
            "experiment_id": created_experiment.experiment_id,
            "name": created_experiment.name,
            "description": created_experiment.description,
            "status": created_experiment.status,
            "variants": [
                {
                    "name": v.name,
                    "description": v.description,
                    "traffic_split": v.traffic_split,
                    "config": v.config,
                    "variant_type": v.variant_type
                } for v in created_experiment.variants
            ],
            "metrics": created_experiment.metrics,
            "created_at": created_experiment.created_at.isoformat(),
            "updated_at": created_experiment.updated_at.isoformat(),
            "tenant_id": tenant_id
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")


@router.get("/experiments")
async def list_experiments(
    status: Optional[str] = None,
    tenant_id: str = Depends(get_tenant_id),
    limit: int = 10,
    offset: int = 0,
    container: Dict = Depends(get_container)
):
    """
    List A/B test experiments for the tenant.
    
    Returns paginated list of experiments, optionally filtered by status.
    """
    try:
        ab_test_service: ABTestingService = container.get("ab_test_service")
        if not ab_test_service:
            # If service not in container, create a temporary one for this demo
            ab_test_service = ABTestingService()
        
        status_enum = ExperimentStatus(status) if status else None
        experiments = await ab_test_service.list_experiments(status=status_enum)
        
        # Apply pagination
        paginated_experiments = experiments[offset:offset+limit]
        
        return {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "description": exp.description,
                    "status": exp.status,
                    "variants": [
                        {
                            "name": v.name,
                            "description": v.description,
                            "traffic_split": v.traffic_split,
                            "variant_type": v.variant_type
                        } for v in exp.variants
                    ],
                    "metrics": exp.metrics,
                    "start_date": exp.start_date.isoformat() if exp.start_date else None,
                    "end_date": exp.end_date.isoformat() if exp.end_date else None,
                    "created_at": exp.created_at.isoformat() if exp.created_at else None,
                    "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
                    "tenant_id": tenant_id
                } for exp in paginated_experiments
            ],
            "total": len(experiments),
            "limit": limit,
            "offset": offset
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid status value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")


@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    tenant_id: str = Depends(get_tenant_id),
    container: Dict = Depends(get_container)
):
    """
    Get details of a specific A/B test experiment.
    """
    try:
        ab_test_service: ABTestingService = container.get("ab_test_service")
        if not ab_test_service:
            # If service not in container, create a temporary one for this demo
            ab_test_service = ABTestingService()
        
        experiment = await ab_test_service.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "status": experiment.status,
            "variants": [
                {
                    "name": v.name,
                    "description": v.description,
                    "traffic_split": v.traffic_split,
                    "config": v.config,
                    "variant_type": v.variant_type
                } for v in experiment.variants
            ],
            "metrics": experiment.metrics,
            "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
            "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
            "hypothesis": experiment.hypothesis,
            "owner": experiment.owner,
            "created_by": experiment.created_by,
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "updated_at": experiment.updated_at.isoformat() if experiment.updated_at else None,
            "tenant_id": tenant_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment: {str(e)}")


@router.post("/experiments/{experiment_id}/assign")
async def assign_variant(
    experiment_id: str,
    user_data: Dict[str, Any] = Body(...),
    tenant_id: str = Depends(get_tenant_id),
    container: Dict = Depends(get_container)
):
    """
    Assign a user to a variant in an A/B test.
    
    This endpoint determines which variant to assign to a user based on
    the experiment configuration and user characteristics.
    """
    try:
        ab_test_service: ABTestingService = container.get("ab_test_service")
        if not ab_test_service:
            # If service not in container, create a temporary one for this demo
            ab_test_service = ABTestingService()
        
        user_id = user_data.get("user_id", "anonymous")
        context = user_data.get("context", {})
        
        assignment = await ab_test_service.assign_variant(
            experiment_id=experiment_id,
            user_id=user_id,
            context=context
        )
        
        return {
            "experiment_id": assignment.experiment_id,
            "user_id": assignment.user_id,
            "variant_name": assignment.variant_name,
            "assigned_at": assignment.assigned_at.isoformat(),
            "context": assignment.context,
            "tenant_id": tenant_id
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to assign variant: {str(e)}")


@router.post("/experiments/{experiment_id}/track")
async def track_event(
    experiment_id: str,
    event_data: Dict[str, Any] = Body(...),
    tenant_id: str = Depends(get_tenant_id),
    container: Dict = Depends(get_container)
):
    """
    Track an event for an A/B test experiment.
    
    This endpoint records events related to an experiment for analysis.
    """
    try:
        ab_test_service: ABTestingService = container.get("ab_test_service")
        if not ab_test_service:
            # If service not in container, create a temporary one for this demo
            ab_test_service = ABTestingService()
        
        user_id = event_data.get("user_id", "anonymous")
        variant_name = event_data["variant_name"]
        event_type = event_data["event_type"]
        value = event_data.get("value")
        metadata = event_data.get("metadata", {})
        
        await ab_test_service.track_event(
            experiment_id=experiment_id,
            user_id=user_id,
            variant_name=variant_name,
            event_type=event_type,
            value=value,
            metadata=metadata
        )
        
        return {
            "experiment_id": experiment_id,
            "user_id": user_id,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "tenant_id": tenant_id
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track event: {str(e)}")


@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str,
    tenant_id: str = Depends(get_tenant_id),
    container: Dict = Depends(get_container)
):
    """
    Get results of an A/B test experiment.
    
    This endpoint returns statistical analysis of experiment results.
    """
    try:
        ab_test_service: ABTestingService = container.get("ab_test_service")
        if not ab_test_service:
            # If service not in container, create a temporary one for this demo
            ab_test_service = ABTestingService()
        
        results = await ab_test_service.get_experiment_results(experiment_id)
        if not results:
            raise HTTPException(status_code=404, detail=f"Results for experiment {experiment_id} not found")
        
        return {
            "experiment_id": results.experiment_id,
            "variant_results": results.variant_results,
            "statistical_significance": results.statistical_significance,
            "winner": results.winner,
            "is_significant": results.is_significant,
            "conclusion": results.conclusion,
            "tenant_id": tenant_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")


@router.patch("/experiments/{experiment_id}/status")
async def update_experiment_status(
    experiment_id: str,
    status_data: Dict[str, str] = Body(...),
    tenant_id: str = Depends(get_tenant_id),
    container: Dict = Depends(get_container)
):
    """
    Update the status of an A/B test experiment.
    
    This endpoint updates the status of an active experiment.
    """
    try:
        ab_test_service: ABTestingService = container.get("ab_test_service")
        if not ab_test_service:
            # If service not in container, create a temporary one for this demo
            ab_test_service = ABTestingService()
        
        new_status = ExperimentStatus(status_data["status"])
        updated_experiment = await ab_test_service.update_experiment_status(experiment_id, new_status)
        
        return {
            "experiment_id": updated_experiment.experiment_id,
            "status": updated_experiment.status,
            "updated_at": updated_experiment.updated_at.isoformat(),
            "tenant_id": tenant_id
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid status value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update status: {str(e)}")


@router.post("/calculate-sample-size")
async def calculate_sample_size(
    calculation_params: Dict[str, float] = Body(...),
    tenant_id: str = Depends(get_tenant_id),
    container: Dict = Depends(get_container)
):
    """
    Calculate required sample size for an A/B test.
    
    This endpoint calculates the required sample size based on statistical parameters.
    """
    try:
        ab_test_service: ABTestingService = container.get("ab_test_service")
        if not ab_test_service:
            # If service not in container, create a temporary one for this demo
            ab_test_service = ABTestingService()
        
        baseline_conversion_rate = calculation_params["baseline_conversion_rate"]
        minimum_detectable_effect = calculation_params["minimum_detectable_effect"]
        significance_level = calculation_params.get("significance_level", 0.05)
        power = calculation_params.get("power", 0.8)
        
        sample_size = await ab_test_service.calculate_sample_size(
            baseline_conversion_rate=baseline_conversion_rate,
            minimum_detectable_effect=minimum_detectable_effect,
            significance_level=significance_level,
            power=power
        )
        
        return {
            "baseline_conversion_rate": baseline_conversion_rate,
            "minimum_detectable_effect": minimum_detectable_effect,
            "significance_level": significance_level,
            "power": power,
            "required_sample_size": sample_size,
            "tenant_id": tenant_id
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate sample size: {str(e)}")


__all__ = ["router"]