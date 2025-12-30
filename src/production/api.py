"""
API Module

This module implements a FastAPI application for serving ML models in production.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import pickle
import time
import logging
from contextlib import asynccontextmanager
import asyncio
from enum import Enum


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Enumeration of supported model types."""
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[List[float]]  # 2D array for batch predictions
    model_type: ModelType
    model_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float]
    model_type: ModelType
    model_id: Optional[str] = None
    processing_time: float
    timestamp: float


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: float


class ModelRegistry:
    """Simple model registry for managing multiple models."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_types: Dict[str, ModelType] = {}
    
    def register_model(self, model_id: str, model: Any, model_type: ModelType):
        """Register a model in the registry."""
        self.models[model_id] = model
        self.model_types[model_id] = model_type
        logger.info(f"Model {model_id} of type {model_type} registered")
    
    def get_model(self, model_id: str):
        """Get a model from the registry."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        return self.models[model_id]
    
    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self.models.keys())


# Global model registry
model_registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for the FastAPI app."""
    logger.info("Starting up the ML API service...")
    
    # Load default models here if needed
    # Example: model_registry.register_model("default_lr", load_model("path/to/model"), ModelType.LINEAR_REGRESSION)
    
    yield
    
    logger.info("Shutting down the ML API service...")


app = FastAPI(
    title="AI-Mastery-2026 ML API",
    description="Production-ready API for serving machine learning models",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to AI-Mastery-2026 ML API"}


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=time.time()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the specified model."""
    start_time = time.time()
    
    try:
        # Convert features to numpy array
        features = np.array(request.features)
        
        # Make prediction based on model type
        if request.model_type == ModelType.LINEAR_REGRESSION:
            # For now, we'll use a simple placeholder
            # In a real implementation, you'd load the actual model
            predictions = features.sum(axis=1).tolist()  # Placeholder calculation
        elif request.model_type == ModelType.LOGISTIC_REGRESSION:
            # Placeholder for logistic regression
            predictions = (features.sum(axis=1) > 0).astype(int).tolist()
        elif request.model_type == ModelType.DECISION_TREE:
            # Placeholder for decision tree
            predictions = (features.sum(axis=1) % 2).tolist()
        elif request.model_type == ModelType.NEURAL_NETWORK:
            # Placeholder for neural network
            predictions = np.tanh(features.sum(axis=1)).tolist()
        elif request.model_type == ModelType.TRANSFORMER:
            # Placeholder for transformer
            predictions = np.sin(features.sum(axis=1)).tolist()
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {request.model_type}")
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            predictions=predictions,
            model_type=request.model_type,
            model_id=request.model_id,
            processing_time=processing_time,
            timestamp=time.time()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/models/register")
async def register_model(model_id: str, model_type: ModelType, model_data: str = None):
    """Register a new model."""
    try:
        # In a real implementation, you would deserialize and validate the model
        # For now, we'll just register a placeholder
        model_registry.register_model(model_id, {"type": model_type}, model_type)
        return {"message": f"Model {model_id} registered successfully", "model_id": model_id}
    except Exception as e:
        logger.error(f"Model registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model registration failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List all registered models."""
    return {"models": model_registry.list_models()}


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model."""
    try:
        model = model_registry.get_model(model_id)
        model_type = model_registry.model_types[model_id]
        return {
            "model_id": model_id,
            "model_type": model_type,
            "info": "Model registered and ready for inference"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model from the registry."""
    try:
        if model_id in model_registry.models:
            del model_registry.models[model_id]
            del model_registry.model_types[model_id]
            return {"message": f"Model {model_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        logger.error(f"Model deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model deletion failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    return {
        "model_count": len(model_registry.models),
        "registered_model_types": list(set(model_registry.model_types.values())),
        "timestamp": time.time()
    }


# Additional endpoints for model management and monitoring
@app.post("/batch_predict")
async def batch_predict(request: PredictionRequest):
    """Make batch predictions with additional processing options."""
    start_time = time.time()
    
    try:
        features = np.array(request.features)
        
        # Perform prediction (same as single predict for now)
        if request.model_type == ModelType.LINEAR_REGRESSION:
            predictions = features.sum(axis=1).tolist()
        elif request.model_type == ModelType.LOGISTIC_REGRESSION:
            predictions = (features.sum(axis=1) > 0).astype(int).tolist()
        elif request.model_type == ModelType.DECISION_TREE:
            predictions = (features.sum(axis=1) % 2).tolist()
        elif request.model_type == ModelType.NEURAL_NETWORK:
            predictions = np.tanh(features.sum(axis=1)).tolist()
        elif request.model_type == ModelType.TRANSFORMER:
            predictions = np.sin(features.sum(axis=1)).tolist()
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {request.model_type}")
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            predictions=predictions,
            model_type=request.model_type,
            model_id=request.model_id,
            processing_time=processing_time,
            timestamp=time.time()
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)