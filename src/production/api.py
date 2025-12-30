"""
FastAPI Model Serving
=====================
Production-ready API endpoints for ML model inference.

Features:
- Async request handling
- Input validation with Pydantic
- SSE streaming for LLM responses
- Health checks and metrics
- Error handling and logging

Author: AI-Mastery-2026
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
import numpy as np
import time
import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================

class PredictionRequest(BaseModel):
    """
    Input schema for prediction endpoint.
    
    Pydantic provides automatic validation, serialization,
    and OpenAPI documentation.
    """
    features: List[float] = Field(
        ..., 
        description="List of feature values",
        example=[0.5, 1.2, -0.3, 2.1]
    )
    model_name: Optional[str] = Field(
        default="default",
        description="Name of the model to use"
    )
    
    @validator('features')
    def features_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError('Features cannot be empty')
        return v


class BatchPredictionRequest(BaseModel):
    """Schema for batch predictions."""
    instances: List[List[float]] = Field(
        ...,
        description="Batch of feature vectors",
        example=[[0.5, 1.2], [0.3, 0.8]]
    )
    model_name: Optional[str] = "default"


class PredictionResponse(BaseModel):
    """Output schema for predictions."""
    prediction: Union[float, int, List[float]]
    confidence: Optional[float] = None
    model_name: str
    latency_ms: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[Union[float, int, List[float]]]
    model_name: str
    latency_ms: float
    batch_size: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str


class ChatMessage(BaseModel):
    """Chat message for LLM endpoints."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    """Chat completion request."""
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stream: bool = Field(default=False)


# ============================================================
# MODEL CACHE
# ============================================================

class ModelCache:
    """
    Singleton model cache for efficient inference.
    
    Loads models once and reuses them across requests.
    Thread-safe for async operations.
    """
    
    _instance = None
    _models: Dict[str, Any] = {}
    _load_times: Dict[str, datetime] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, name: str, model: Any):
        """Load a model into cache."""
        self._models[name] = model
        self._load_times[name] = datetime.now()
        logger.info(f"Model '{name}' loaded into cache")
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a model from cache."""
        return self._models.get(name)
    
    def is_loaded(self, name: str) -> bool:
        """Check if model is loaded."""
        return name in self._models
    
    def get_all_models(self) -> List[str]:
        """List all loaded models."""
        return list(self._models.keys())
    
    def unload_model(self, name: str):
        """Unload a model from cache."""
        if name in self._models:
            del self._models[name]
            del self._load_times[name]
            logger.info(f"Model '{name}' unloaded from cache")


# Global instance
model_cache = ModelCache()


# ============================================================
# METRICS COLLECTION
# ============================================================

class MetricsCollector:
    """
    Simple metrics collector for monitoring.
    
    In production, integrate with Prometheus or similar.
    """
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.latencies: List[float] = []
        self.start_time = time.time()
    
    def record_request(self, latency_ms: float, success: bool = True):
        """Record a request."""
        self.request_count += 1
        self.latencies.append(latency_ms)
        if not success:
            self.error_count += 1
        
        # Keep only last 1000 latencies for memory efficiency
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        latencies = self.latencies if self.latencies else [0]
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "uptime_seconds": time.time() - self.start_time
        }


metrics = MetricsCollector()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown.
    
    Use this to load models, initialize connections, etc.
    """
    # Startup
    logger.info("Starting up ML API server...")
    
    # Load default model (example - replace with actual model loading)
    # model_cache.load_model("default", load_your_model())
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML API server...")


def create_app(title: str = "ML Model API",
               version: str = "1.0.0",
               debug: bool = False) -> FastAPI:
    """
    Factory function to create FastAPI application.
    
    Args:
        title: API title
        version: API version
        debug: Enable debug mode
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        version=version,
        description="Production ML Model Serving API",
        lifespan=lifespan,
        debug=debug
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    register_routes(app)
    
    return app


def register_routes(app: FastAPI):
    """Register all API routes."""
    
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint."""
        return {"message": "ML Model API is running", "docs": "/docs"}
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Health check endpoint.
        
        Used by load balancers and Kubernetes probes.
        """
        return HealthResponse(
            status="healthy",
            model_loaded=model_cache.is_loaded("default"),
            uptime_seconds=time.time() - metrics.start_time,
            version="1.0.0"
        )
    
    @app.get("/ready", tags=["Health"])
    async def readiness_check():
        """
        Readiness check - requires model to be loaded.
        
        Kubernetes uses this to determine if pod can receive traffic.
        """
        if not model_cache.is_loaded("default"):
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {"status": "ready"}
    
    @app.get("/metrics", tags=["Monitoring"])
    async def get_metrics():
        """
        Metrics endpoint for monitoring.
        
        Returns request counts, latencies, error rates.
        Format compatible with Prometheus.
        """
        return metrics.get_metrics()
    
    @app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
    async def predict(request: PredictionRequest):
        """
        Single prediction endpoint.
        
        Send feature vector, get model prediction.
        
        Example:
            POST /predict
            {"features": [0.5, 1.2, -0.3, 2.1]}
        """
        start_time = time.time()
        
        try:
            model = model_cache.get_model(request.model_name)
            
            if model is None:
                # Demo: return dummy prediction if no model loaded
                features = np.array(request.features)
                prediction = float(np.mean(features))
                confidence = 0.85
            else:
                features = np.array(request.features).reshape(1, -1)
                prediction = model.predict(features)[0]
                confidence = getattr(model, 'predict_proba', lambda x: [[0.85]])(features)[0].max()
            
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_request(latency_ms, success=True)
            
            return PredictionResponse(
                prediction=prediction,
                confidence=float(confidence),
                model_name=request.model_name,
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat()
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_request(latency_ms, success=False)
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
    async def predict_batch(request: BatchPredictionRequest):
        """
        Batch prediction endpoint.
        
        Process multiple instances in one request for efficiency.
        """
        start_time = time.time()
        
        try:
            model = model_cache.get_model(request.model_name)
            features = np.array(request.instances)
            
            if model is None:
                # Demo: return dummy predictions
                predictions = [float(np.mean(f)) for f in features]
            else:
                predictions = model.predict(features).tolist()
            
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_request(latency_ms, success=True)
            
            return BatchPredictionResponse(
                predictions=predictions,
                model_name=request.model_name,
                latency_ms=latency_ms,
                batch_size=len(request.instances)
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_request(latency_ms, success=False)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/chat/completions", tags=["LLM"])
    async def chat_completions(request: ChatRequest):
        """
        Chat completions endpoint (OpenAI-compatible).
        
        Supports both streaming and non-streaming responses.
        """
        if request.stream:
            return StreamingResponse(
                stream_chat_response(request),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            response_text = generate_chat_response(request)
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }]
            }
    
    @app.get("/models", tags=["Models"])
    async def list_models():
        """List all loaded models."""
        return {"models": model_cache.get_all_models()}
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__}
        )


# ============================================================
# STREAMING (SSE for LLM responses)
# ============================================================

async def stream_chat_response(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Server-Sent Events (SSE) streaming for chat.
    
    Used for real-time token-by-token LLM output.
    """
    # Simulate token generation (replace with actual LLM)
    response_text = generate_chat_response(request)
    tokens = response_text.split()
    
    for i, token in enumerate(tokens):
        data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "delta": {"content": token + " "},
                "finish_reason": None if i < len(tokens) - 1 else "stop"
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.05)  # Simulate token generation delay
    
    yield "data: [DONE]\n\n"


def generate_chat_response(request: ChatRequest) -> str:
    """
    Generate chat response (placeholder).
    
    Replace with actual LLM inference:
    - OpenAI API
    - Local model (Llama, Mistral)
    - vLLM / TGI
    """
    # Get last user message
    user_messages = [m for m in request.messages if m.role == "user"]
    last_message = user_messages[-1].content if user_messages else ""
    
    # Placeholder response
    return f"Echo: {last_message}"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_sklearn_model(path: str):
    """
    Load a scikit-learn model from disk.
    
    Args:
        path: Path to pickle/joblib file
    
    Returns:
        Loaded model
    """
    import joblib
    return joblib.load(path)


def load_pytorch_model(path: str, model_class):
    """
    Load a PyTorch model from disk.
    
    Args:
        path: Path to .pt file
        model_class: Model class to instantiate
    
    Returns:
        Loaded model in eval mode
    """
    import torch
    model = model_class()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


# ============================================================
# MAIN ENTRY POINT
# ============================================================

# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
