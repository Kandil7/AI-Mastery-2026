"""
GitHub Issue Classifier FastAPI Service
========================================

Production-ready API for serving the trained issue classifier.

Endpoints:
- POST /classify - Classify a GitHub issue
- GET /health - Health check
- GET /metrics - Prometheus metrics
- GET /model/info - Model metadata

Features:
- Model caching
- Request validation
- Prometheus monitoring
- Error handling
- Logging
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pickle
import numpy as np
import logging
from pathlib import Path
import time
from datetime import datetime

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GitHub Issue Classifier API",
    description="Classify GitHub issues into bug, feature, question, or documentation categories",
    version="1.0.0"
)

# Prometheus Metrics
PREDICTION_COUNT = Counter(
    'issue_classifier_predictions_total',
    'Total number of predictions made',
    ['label', 'model_version']
)

PREDICTION_LATENCY = Histogram(
    'issue_classifier_prediction_latency_seconds',
    'Time taken to make a prediction',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

REQUEST_COUNT = Counter(
    'issue_classifier_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status']
)

MODEL_LOAD_TIME = Gauge(
    'issue_classifier_model_load_time_seconds',
    'Time taken to load the model'
)

# Global variables for model components
classifier = None
vectorizer = None
labels = None
model_metadata = None


class IssueRequest(BaseModel):
    """Request model for issue classification."""
    text: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The GitHub issue text to classify",
        example="Bug: Application crashes when clicking submit button"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Error when login: AuthenticationError"
            }
        }


class IssueResponse(BaseModel):
    """Response model for issue classification."""
    label: str = Field(..., description="Predicted issue category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    all_scores: Dict[str, float] = Field(..., description="Confidence scores for all categories")
    processing_time_ms: float = Field(..., description="Time taken to process request in milliseconds")
    model_version: str = Field(..., description="Model version used for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "label": "bug",
                "confidence": 0.92,
                "all_scores": {
                    "bug": 0.92,
                    "feature": 0.04,
                    "question": 0.03,
                    "documentation": 0.01
                },
                "processing_time_ms": 8.5,
                "model_version": "1.0.0"
            }
        }


class ModelInfo(BaseModel):
    """Model metadata response."""
    model_version: str
    labels: List[str]
    test_accuracy: float
    n_features: int
    n_classes: int
    loaded_at: str


@app.on_event("startup")
async def load_model():
    """Load model and vectorizer on application startup."""
    global classifier, vectorizer, labels, model_metadata
    
    start_time = time.time()
    logger.info("Loading model...")
    
    try:
        model_path = Path("models/issue_classifier.pkl")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Rebuild vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=data['vectorizer_params']['max_features'],
            vocabulary=data['vectorizer_vocab'],
            ngram_range=tuple(data['vectorizer_params']['ngram_range'])
        )
        vectorizer.idf_ = np.array(data['vectorizer_idf'])
        
        # Rebuild model
        from src.ml.deep_learning import NeuralNetwork, Dense, Activation, Dropout
        
        classifier = NeuralNetwork()
        
        # Reconstruct architecture based on saved parameters
        # Layer 1: Dense + ReLU + Dropout
        input_dim = data['metadata']['n_features']
        classifier.add(Dense(input_dim, 128, weight_init='he'))
        classifier.add(Activation('relu'))
        classifier.add(Dropout(0.3))
        
        # Layer 2: Dense + ReLU + Dropout
        classifier.add(Dense(128, 64, weight_init='he'))
        classifier.add(Activation('relu'))
        classifier.add(Dropout(0.2))
        
        # Output layer
        classifier.add(Dense(64, data['metadata']['n_classes'], weight_init='xavier'))
        classifier.add(Activation('softmax'))
        
        # Set saved parameters
        for i, layer_params in enumerate(data['layer_params']):
            classifier.layers[i].set_params(layer_params)
        
        labels = data['labels']
        model_metadata = data['metadata']
        model_metadata['loaded_at'] = datetime.now().isoformat()
        model_metadata['model_version'] = "1.0.0"
        
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)
        
        logger.info(f"✓ Model loaded successfully in {load_time:.2f}s")
        logger.info(f"  Test Accuracy: {model_metadata['test_accuracy']:.4f}")
        logger.info(f"  Labels: {labels}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.post("/classify", response_model=IssueResponse, tags=["Prediction"])
async def classify_issue(request: IssueRequest):
    """
    Classify a GitHub issue into one of four categories.
    
    Categories:
    - bug: Error reports and defects
    - feature: New feature requests
    - question: User questions
    - documentation: Documentation updates
    """
    start_time = time.time()
    
    try:
        # Vectorize input text
        X = vectorizer.transform([request.text]).toarray()
        
        # Get prediction
        with PREDICTION_LATENCY.time():
            # Get probability scores
            classifier.training = False  # Ensure dropout is off
            proba = []
            for layer in classifier.layers:
                if isinstance(layer, Dropout):
                    layer.training = False
            
            # Forward pass
            output = X
            for layer in classifier.layers:
                output = layer.forward(output)
            
            proba = output[0]  # Get first sample predictions
        
        predicted_idx = np.argmax(proba)
        predicted_label = labels[predicted_idx]
        confidence = float(proba[predicted_idx])
        
        # Build response
        all_scores = {labels[i]: float(proba[i]) for i in range(len(labels))}
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        PREDICTION_COUNT.labels(
            label=predicted_label,
            model_version=model_metadata['model_version']
        ).inc()
        
        REQUEST_COUNT.labels(
            endpoint='/classify',
            method='POST',
            status='200'
        ).inc()
        
        logger.info(f"Classified: '{request.text[:50]}...' -> {predicted_label} ({confidence:.4f})")
        
        return IssueResponse(
            label=predicted_label,
            confidence=confidence,
            all_scores=all_scores,
            processing_time_ms=processing_time_ms,
            model_version=model_metadata['model_version']
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(
            endpoint='/classify',
            method='POST',
            status='500'
        ).inc()
        
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health", tags=["Operations"])
async def health_check():
    """Health check endpoint."""
    is_ready = all([classifier is not None, vectorizer is not None, labels is not None])
    
    status = "healthy" if is_ready else "not_ready"
    status_code = 200 if is_ready else 503
    
    REQUEST_COUNT.labels(
        endpoint='/health',
        method='GET',
        status=str(status_code)
    ).inc()
    
    return {
        "status": status,
        "model_loaded": classifier is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics", tags=["Operations"])
async def metrics():
    """Prometheus metrics endpoint."""
    REQUEST_COUNT.labels(
        endpoint='/metrics',
        method='GET',
        status='200'
    ).inc()
    
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model metadata and information."""
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    REQUEST_COUNT.labels(
        endpoint='/model/info',
        method='GET',
        status='200'
    ).inc()
    
    return ModelInfo(
        model_version=model_metadata['model_version'],
        labels=labels,
        test_accuracy=model_metadata['test_accuracy'],
        n_features=model_metadata['n_features'],
        n_classes=model_metadata['n_classes'],
        loaded_at=model_metadata['loaded_at']
    )


@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "GitHub Issue Classifier API",
        "version": "1.0.0",
        "description": "Classify GitHub issues into bug, feature, question, or documentation",
        "endpoints": {
            "classify": "POST /classify",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "model_info": "GET /model/info",
            "docs": "GET /docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "issue_classifier_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )
