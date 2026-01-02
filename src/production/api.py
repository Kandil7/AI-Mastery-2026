"""
API Module

This module implements a FastAPI application for serving ML models in production.
Supports real scikit-learn model loading and inference.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import numpy as np
import time
import logging
import os
import json
from contextlib import asynccontextmanager
import asyncio
from enum import Enum
from pathlib import Path

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Enumeration of supported model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    LOGISTIC = "logistic"
    # Legacy types for backwards compatibility
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[List[float]]  # 2D array for batch predictions
    model_type: Optional[ModelType] = None
    model_id: Optional[str] = None


class SimplePredictionRequest(BaseModel):
    """Simple request model for single predictions."""
    features: List[float]  # 1D array for single prediction
    model_name: Optional[str] = "classification_model"


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[Union[float, int]]
    probabilities: Optional[List[List[float]]] = None
    model_type: Optional[str] = None
    model_id: Optional[str] = None
    processing_time: float
    timestamp: float


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str
    models_loaded: int
    timestamp: float


class ModelInfo(BaseModel):
    """Model information response."""
    model_id: str
    model_type: str
    n_features: int
    metadata: Dict[str, Any]


class ModelCache:
    """
    Model cache for efficient model loading and serving.
    
    Loads models from disk and caches them in memory for fast inference.
    """
    
    def __init__(self, models_dir: str = None):
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.models_dir = models_dir or self._find_models_dir()
        
    def _find_models_dir(self) -> str:
        """Find the models directory."""
        # Try common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'models'),
            os.path.join(os.path.dirname(__file__), 'models'),
            './models',
            '../models',
        ]
        
        for candidate in candidates:
            abs_path = os.path.abspath(candidate)
            if os.path.exists(abs_path):
                return abs_path
        
        # Create models directory if it doesn't exist
        default_path = os.path.abspath('./models')
        os.makedirs(default_path, exist_ok=True)
        return default_path
    
    def load_all_models(self):
        """Load all models from the models directory."""
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        # Check for metadata file
        metadata_path = os.path.join(self.models_dir, 'models_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(all_metadata)} models")
        
        # Load all .joblib files
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.joblib'):
                model_name = filename.replace('.joblib', '')
                self.load_model(model_name)
            elif filename.endswith('.pkl'):
                model_name = filename.replace('.pkl', '')
                self.load_model(model_name)
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model."""
        # Try .joblib first, then .pkl
        for ext in ['.joblib', '.pkl']:
            model_path = os.path.join(self.models_dir, f"{model_name}{ext}")
            if os.path.exists(model_path):
                try:
                    if JOBLIB_AVAILABLE and ext == '.joblib':
                        data = joblib.load(model_path)
                    else:
                        with open(model_path, 'rb') as f:
                            data = pickle.load(f)
                    
                    # Handle both old format (just model) and new format (dict with model and metadata)
                    if isinstance(data, dict) and 'model' in data:
                        self.models[model_name] = data['model']
                        self.model_metadata[model_name] = data.get('metadata', {})
                    else:
                        self.models[model_name] = data
                        self.model_metadata[model_name] = {'model_type': 'unknown'}
                    
                    logger.info(f"Loaded model: {model_name} from {model_path}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    return False
        
        logger.warning(f"Model file not found: {model_name}")
        return False
    
    def get_model(self, model_name: str):
        """Get a loaded model."""
        if model_name not in self.models:
            # Try to load it
            if not self.load_model(model_name):
                raise ValueError(f"Model {model_name} not found")
        return self.models[model_name]
    
    def get_metadata(self, model_name: str) -> Dict:
        """Get model metadata."""
        return self.model_metadata.get(model_name, {})
    
    def list_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())
    
    def predict(self, model_name: str, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction using a model."""
        model = self.get_model(model_name)
        metadata = self.get_metadata(model_name)
        
        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        result = {'predictions': None, 'probabilities': None}
        
        # Make prediction
        predictions = model.predict(features)
        result['predictions'] = predictions.tolist()
        
        # Try to get probabilities for classifiers
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(features)
                result['probabilities'] = probabilities.tolist()
            except Exception:
                pass
        
        return result


# Global model cache
model_cache = ModelCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for the FastAPI app."""
    logger.info("Starting up the ML API service...")
    logger.info(f"Looking for models in: {model_cache.models_dir}")
    
    # Load all available models
    model_cache.load_all_models()
    
    loaded_models = model_cache.list_models()
    if loaded_models:
        logger.info(f"Loaded {len(loaded_models)} models: {loaded_models}")
    else:
        logger.warning("No models loaded. Run 'python scripts/train_save_models.py' to create models.")
    
    yield
    
    logger.info("Shutting down the ML API service...")


app = FastAPI(
    title="AI-Mastery-2026 ML API",
    description="Production-ready API for serving machine learning models with real inference",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to AI-Mastery-2026 ML API",
        "version": "2.0.0",
        "models_loaded": len(model_cache.list_models()),
        "models": model_cache.list_models()
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        models_loaded=len(model_cache.list_models()),
        timestamp=time.time()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: SimplePredictionRequest):
    """
    Make predictions using a loaded model.
    
    Simple endpoint that accepts a feature array and model name.
    """
    start_time = time.time()
    
    try:
        # Convert features to numpy array
        features = np.array(request.features)
        
        # Determine model to use
        model_name = request.model_name or "classification_model"
        
        # Check if model is loaded
        if not model_cache.list_models():
            # Fall back to placeholder logic if no models loaded
            logger.warning("No models loaded, using placeholder prediction")
            predictions = [float(np.sum(features) > 0)]
            probabilities = None
        else:
            # Use real model
            result = model_cache.predict(model_name, features)
            predictions = result['predictions']
            probabilities = result['probabilities']
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_type=model_cache.get_metadata(model_name).get('model_type'),
            model_id=model_name,
            processing_time=processing_time,
            timestamp=time.time()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=PredictionResponse)
async def batch_predict(request: PredictionRequest):
    """Make batch predictions using the specified model."""
    start_time = time.time()
    
    try:
        features = np.array(request.features)
        model_name = request.model_id or "classification_model"
        
        if not model_cache.list_models():
            # Fallback to legacy placeholder logic
            if request.model_type == ModelType.LINEAR_REGRESSION:
                predictions = features.sum(axis=1).tolist()
            elif request.model_type == ModelType.LOGISTIC_REGRESSION:
                predictions = (features.sum(axis=1) > 0).astype(int).tolist()
            else:
                predictions = (features.sum(axis=1) > 0).astype(int).tolist()
            probabilities = None
        else:
            result = model_cache.predict(model_name, features)
            predictions = result['predictions']
            probabilities = result['probabilities']
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_type=str(request.model_type) if request.model_type else None,
            model_id=model_name,
            processing_time=processing_time,
            timestamp=time.time()
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List all loaded models with their metadata."""
    models = []
    for model_name in model_cache.list_models():
        metadata = model_cache.get_metadata(model_name)
        models.append({
            "model_id": model_name,
            "model_type": metadata.get('model_type', 'unknown'),
            "n_features": metadata.get('n_features', 'unknown'),
            "metadata": metadata
        })
    return {"models": models, "count": len(models)}


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model."""
    try:
        model = model_cache.get_model(model_id)
        metadata = model_cache.get_metadata(model_id)
        return {
            "model_id": model_id,
            "model_type": metadata.get('model_type', 'unknown'),
            "n_features": metadata.get('n_features', 'unknown'),
            "metadata": metadata,
            "status": "loaded"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/models/reload")
async def reload_models():
    """Reload all models from disk."""
    model_cache.models.clear()
    model_cache.model_metadata.clear()
    model_cache.load_all_models()
    return {
        "message": "Models reloaded",
        "models_loaded": len(model_cache.list_models()),
        "models": model_cache.list_models()
    }


@app.get("/metrics")
async def get_metrics():
    """Get service metrics in Prometheus-compatible format."""
    models = model_cache.list_models()
    metrics = {
        "ml_api_models_loaded": len(models),
        "ml_api_status": 1,
        "timestamp": time.time()
    }
    
    # Add per-model metrics
    for i, model_name in enumerate(models):
        metadata = model_cache.get_metadata(model_name)
        metrics[f"ml_api_model_{i}_features"] = metadata.get('n_features', 0)
    
    return metrics


# Legacy endpoint for backwards compatibility
@app.post("/v1/predict", response_model=PredictionResponse)
async def legacy_predict(request: PredictionRequest):
    """Legacy prediction endpoint for backwards compatibility."""
    start_time = time.time()
    
    try:
        features = np.array(request.features)
        
        # Map legacy model types to new model names
        model_mapping = {
            ModelType.CLASSIFICATION: "classification_model",
            ModelType.LOGISTIC_REGRESSION: "logistic_model",
            ModelType.LOGISTIC: "logistic_model",
            ModelType.REGRESSION: "regression_model",
            ModelType.LINEAR_REGRESSION: "regression_model",
        }
        
        model_name = request.model_id
        if not model_name and request.model_type:
            model_name = model_mapping.get(request.model_type, "classification_model")
        model_name = model_name or "classification_model"
        
        if model_cache.list_models():
            result = model_cache.predict(model_name, features)
            predictions = result['predictions']
            probabilities = result['probabilities']
        else:
            # Fallback placeholder
            predictions = features.sum(axis=1).tolist()
            probabilities = None
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_type=str(request.model_type) if request.model_type else None,
            model_id=model_name,
            processing_time=processing_time,
            timestamp=time.time()
        )
    
    except Exception as e:
        logger.error(f"Legacy prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


class ChatRequest(BaseModel):
    """Request model for chat/RAG."""
    query: str
    k: int = 3


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    Chat completion endpoint (RAG placeholder).
    
    Currently returns a mock response to ensure frontend connectivity.
    Real RAG integration will be added in the next phase.
    """
    return {
        "response": f"I received your query: '{request.query}'.\n\n(Note: The full RAG system is currently being integrated into the API. This is a confirmation that the endpoint is reachable.)",
        "sources": ["System Placeholder"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)