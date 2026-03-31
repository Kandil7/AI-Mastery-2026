"""
LLM Service with Prometheus Metrics

This FastAPI application simulates LLM API calls and exports
comprehensive token usage metrics to Prometheus.
"""

import asyncio
import time
import random
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
import tiktoken
import httpx

from metrics import (
    record_token_usage,
    record_error,
    update_rate_metrics,
    set_model_availability,
    LLM_ACTIVE_REQUESTS,
    LLM_MODEL_USAGE,
)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration."""
    ENVIRONMENT = "development"
    PROMETHEUS_PORT = 8000
    
    # Pricing per 1000 tokens (USD)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "gemini-pro": {"input": 0.00025, "output": 0.0005},
    }
    
    # Simulated latency ranges (ms)
    LATENCY_RANGES = {
        "gpt-4": (200, 800),
        "gpt-4-turbo": (150, 500),
        "gpt-3.5-turbo": (100, 300),
        "claude-3-opus": (300, 1000),
        "claude-3-sonnet": (150, 500),
        "claude-3-haiku": (50, 200),
        "gemini-pro": (100, 400),
    }

config = Config()

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Chat completion request model."""
    prompt: str = Field(..., description="Input prompt")
    model: str = Field(default="gpt-3.5-turbo", description="Model to use")
    max_tokens: int = Field(default=100, ge=1, le=4000, description="Max output tokens")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Temperature")
    stream: bool = Field(default=False, description="Stream response")


class ChatResponse(BaseModel):
    """Chat completion response model."""
    id: str
    model: str
    choices: list[dict]
    usage: dict
    cost_usd: float
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


# =============================================================================
# TOKENIZER
# =============================================================================

class TokenCounter:
    """Count tokens for various models."""
    
    _encoders: Dict[str, tiktoken.Encoding] = {}
    
    @classmethod
    def count_tokens(cls, text: str, model: str) -> int:
        """Count tokens for given text and model."""
        try:
            if model not in cls._encoders:
                cls._encoders[model] = tiktoken.encoding_for_model(model)
            return len(cls._encoders[model].encode(text))
        except KeyError:
            # Fallback to cl100k_base for unknown models
            if "cl100k_base" not in cls._encoders:
                cls._encoders["cl100k_base"] = tiktoken.get_encoding("cl100k_base")
            return len(cls._encoders["cl100k_base"].encode(text))
        except Exception:
            # Rough estimate: 4 characters ≈ 1 token
            return len(text) // 4


# =============================================================================
# COST CALCULATOR
# =============================================================================

class CostCalculator:
    """Calculate LLM costs."""
    
    @staticmethod
    def calculate(model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for a request.
        
        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        pricing = config.PRICING.get(model, {"input": 0.001, "output": 0.002})
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return round(input_cost + output_cost, 6)


# =============================================================================
# SIMULATED LLM CLIENT
# =============================================================================

class SimulatedLLMClient:
    """Simulate LLM API responses for testing."""
    
    SAMPLE_RESPONSES = [
        "The capital of France is Paris. It is known for its rich history, culture, and iconic landmarks like the Eiffel Tower.",
        "Quantum computing leverages quantum mechanics principles like superposition and entanglement to process information in ways classical computers cannot.",
        "Code flows like water,\nVariables dance in the light,\nBugs hide in the shadows.",
        "Regular exercise improves cardiovascular health, boosts mood through endorphin release, and enhances cognitive function.",
        "Start with Python basics: variables, data types, control flow. Practice with small projects, then gradually tackle more complex challenges."
    ]
    
    @classmethod
    async def generate(
        cls,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[str, int]:
        """
        Generate a simulated LLM response.
        
        Returns:
            Tuple of (response_text, output_tokens)
        """
        # Simulate API latency
        latency_range = config.LATENCY_RANGES.get(model, (100, 500))
        await asyncio.sleep(random.uniform(*latency_range) / 1000)
        
        # Generate response based on prompt
        response = random.choice(cls.SAMPLE_RESPONSES)
        
        # Truncate to max_tokens (approximate)
        estimated_tokens = TokenCounter.count_tokens(response, model)
        if estimated_tokens > max_tokens:
            ratio = max_tokens / estimated_tokens
            response = response[:int(len(response) * ratio)]
        
        output_tokens = TokenCounter.count_tokens(response, model)
        
        return response, output_tokens


# =============================================================================
# LIFESPAN & STARTUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting LLM Service with Prometheus metrics...")
    
    # Set initial model availability
    for model in config.PRICING.keys():
        provider = "openai" if "gpt" in model else ("anthropic" if "claude" in model else "google")
        set_model_availability(model, provider, True)
    
    # Start background metrics updater
    asyncio.create_task(update_metrics_periodically())
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Service...")


async def update_metrics_periodically():
    """Periodically update rate-based metrics."""
    while True:
        await asyncio.sleep(60)  # Update every minute
        # In production, calculate actual rates from recent data
        for model in config.PRICING.keys():
            update_rate_metrics(
                model=model,
                input_tokens_per_min=random.uniform(100, 1000),
                output_tokens_per_min=random.uniform(50, 500),
                cost_per_hour=random.uniform(0.1, 10.0),
                environment=config.ENVIRONMENT
            )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="LLM Service with Metrics",
    description="Simulated LLM service with comprehensive Prometheus metrics",
    version="1.0.0",
    lifespan=lifespan
)

# Track active requests
active_requests: Dict[str, int] = {}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        version="1.0.0"
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest, http_request: Request):
    """
    Chat completion endpoint with full metrics tracking.
    
    This endpoint simulates an LLM API call and records
    comprehensive metrics including tokens, cost, and latency.
    """
    start_time = time.time()
    
    # Validate model
    if request.model not in config.PRICING:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {request.model}. Available: {list(config.PRICING.keys())}"
        )
    
    # Track active requests
    LLM_ACTIVE_REQUESTS.labels(
        model=request.model,
        endpoint="/chat"
    ).inc()
    
    try:
        # Count input tokens
        input_tokens = TokenCounter.count_tokens(request.prompt, request.model)
        
        # Simulate LLM generation
        response_text, output_tokens = await SimulatedLLMClient.generate(
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Calculate cost
        cost_usd = CostCalculator.calculate(
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        record_token_usage(
            model=request.model,
            endpoint="/chat",
            environment=config.ENVIRONMENT,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            duration_seconds=latency_ms / 1000,
            status="success"
        )
        
        # Record model usage
        provider = "openai" if "gpt" in request.model else ("anthropic" if "claude" in request.model else "google")
        LLM_MODEL_USAGE.labels(
            model=request.model,
            provider=provider
        ).inc()
        
        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            cost_usd=cost_usd,
            latency_ms=round(latency_ms, 2)
        )
        
    except Exception as e:
        # Record error
        record_error(
            model=request.model,
            endpoint="/chat",
            error_type=type(e).__name__
        )
        raise
    
    finally:
        # Decrement active requests
        LLM_ACTIVE_REQUESTS.labels(
            model=request.model,
            endpoint="/chat"
        ).dec()


@app.post("/completion", response_model=ChatResponse)
async def text_completion(request: ChatRequest):
    """Text completion endpoint (similar to chat but for completions)."""
    # Reuse chat completion logic
    return await chat_completion(request, Request(scope={}))


@app.get("/models")
async def list_models():
    """List available models with pricing."""
    return {
        "models": [
            {
                "id": model,
                "pricing": pricing,
                "latency_range_ms": config.LATENCY_RANGES.get(model, (0, 0))
            }
            for model, pricing in config.PRICING.items()
        ]
    }


@app.get("/pricing")
async def get_pricing():
    """Get current pricing information."""
    return {
        "pricing": config.PRICING,
        "currency": "USD",
        "unit": "per 1000 tokens"
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
