"""
Prometheus Metrics Definitions for LLM Token Tracking

This module defines all Prometheus metrics used for tracking
token usage, costs, and related statistics in LLM applications.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary
from typing import Optional

# =============================================================================
# TOKEN COUNTERS
# =============================================================================

# Total input tokens processed (cumulative counter)
# Labels: model, endpoint, environment
LLM_TOKENS_INPUT_TOTAL = Counter(
    name='llm_tokens_input_total',
    documentation='Total number of input tokens processed',
    labelnames=['model', 'endpoint', 'environment']
)

# Total output tokens generated (cumulative counter)
# Labels: model, endpoint, environment
LLM_TOKENS_OUTPUT_TOTAL = Counter(
    name='llm_tokens_output_total',
    documentation='Total number of output tokens generated',
    labelnames=['model', 'endpoint', 'environment']
)

# =============================================================================
# TOKEN DISTRIBUTIONS
# =============================================================================

# Distribution of tokens per request (histogram)
# Labels: model, type (input/output)
LLM_TOKENS_PER_REQUEST = Histogram(
    name='llm_tokens_per_request',
    documentation='Distribution of tokens per request',
    labelnames=['model', 'type'],
    buckets=[
        10, 25, 50, 100, 250, 500, 750, 1000,
        2000, 3000, 4000, 5000, 7500, 10000,
        15000, 20000, 50000, 100000, float('inf')
    ]
)

# Token ratio (output/input) distribution
LLM_TOKEN_RATIO = Histogram(
    name='llm_token_ratio',
    documentation='Ratio of output tokens to input tokens',
    labelnames=['model', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
)

# =============================================================================
# RATE METRICS (GAUGES)
# =============================================================================

# Current tokens per minute rate (gauge, updated periodically)
# Labels: model, type
LLM_TOKENS_PER_MINUTE = Gauge(
    name='llm_tokens_per_minute',
    documentation='Current rate of tokens per minute',
    labelnames=['model', 'type']
)

# Active requests being processed
LLM_ACTIVE_REQUESTS = Gauge(
    name='llm_active_requests',
    documentation='Number of currently active LLM requests',
    labelnames=['model', 'endpoint']
)

# =============================================================================
# COST METRICS
# =============================================================================

# Total cost in USD (cumulative counter)
# Labels: model, endpoint, environment
LLM_COST_USD_TOTAL = Counter(
    name='llm_cost_usd_total',
    documentation='Total cost in USD for LLM usage',
    labelnames=['model', 'endpoint', 'environment']
)

# Cost per request distribution
LLM_COST_PER_REQUEST = Histogram(
    name='llm_cost_per_request',
    documentation='Cost distribution per request in USD',
    labelnames=['model', 'endpoint'],
    buckets=[
        0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01,
        0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0
    ]
)

# Current cost rate per hour (gauge)
LLM_COST_PER_HOUR = Gauge(
    name='llm_cost_per_hour',
    documentation='Current cost rate per hour in USD',
    labelnames=['environment']
)

# =============================================================================
# REQUEST METRICS
# =============================================================================

# Total requests processed (counter)
LLM_REQUESTS_TOTAL = Counter(
    name='llm_requests_total',
    documentation='Total number of LLM requests processed',
    labelnames=['model', 'endpoint', 'environment', 'status']
)

# Request duration histogram
LLM_REQUEST_DURATION = Histogram(
    name='llm_request_duration_seconds',
    documentation='LLM request duration in seconds',
    labelnames=['model', 'endpoint'],
    buckets=[
        0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0,
        2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0, float('inf')
    ]
)

# =============================================================================
# ERROR METRICS
# =============================================================================

# Total errors (counter)
LLM_ERRORS_TOTAL = Counter(
    name='llm_errors_total',
    documentation='Total number of LLM errors',
    labelnames=['model', 'endpoint', 'error_type']
)

# Error rate (gauge, calculated periodically)
LLM_ERROR_RATE = Gauge(
    name='llm_error_rate',
    documentation='Current error rate (errors/total requests)',
    labelnames=['model', 'endpoint']
)

# =============================================================================
# MODEL-SPECIFIC METRICS
# =============================================================================

# Model usage count
LLM_MODEL_USAGE = Counter(
    name='llm_model_usage_total',
    documentation='Total usage count per model',
    labelnames=['model', 'provider']
)

# Model availability (gauge: 1 = available, 0 = unavailable)
LLM_MODEL_AVAILABLE = Gauge(
    name='llm_model_available',
    documentation='Model availability status',
    labelnames=['model', 'provider']
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def record_token_usage(
    model: str,
    endpoint: str,
    environment: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    duration_seconds: float,
    status: str = "success"
) -> None:
    """
    Record complete token usage metrics for a request.
    
    Args:
        model: Model identifier (e.g., "gpt-4", "claude-3-opus")
        endpoint: API endpoint (e.g., "/chat", "/completion")
        environment: Environment name (e.g., "production", "development")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost_usd: Cost in USD
        duration_seconds: Request duration in seconds
        status: Request status ("success" or "error")
    """
    # Record token counters
    LLM_TOKENS_INPUT_TOTAL.labels(
        model=model,
        endpoint=endpoint,
        environment=environment
    ).inc(input_tokens)
    
    LLM_TOKENS_OUTPUT_TOTAL.labels(
        model=model,
        endpoint=endpoint,
        environment=environment
    ).inc(output_tokens)
    
    # Record token distributions
    LLM_TOKENS_PER_REQUEST.labels(
        model=model,
        type="input"
    ).observe(input_tokens)
    
    LLM_TOKENS_PER_REQUEST.labels(
        model=model,
        type="output"
    ).observe(output_tokens)
    
    # Record token ratio
    if input_tokens > 0:
        ratio = output_tokens / input_tokens
        LLM_TOKEN_RATIO.labels(
            model=model,
            endpoint=endpoint
        ).observe(ratio)
    
    # Record cost
    LLM_COST_USD_TOTAL.labels(
        model=model,
        endpoint=endpoint,
        environment=environment
    ).inc(cost_usd)
    
    LLM_COST_PER_REQUEST.labels(
        model=model,
        endpoint=endpoint
    ).observe(cost_usd)
    
    # Record request metrics
    LLM_REQUESTS_TOTAL.labels(
        model=model,
        endpoint=endpoint,
        environment=environment,
        status=status
    ).inc()
    
    LLM_REQUEST_DURATION.labels(
        model=model,
        endpoint=endpoint
    ).observe(duration_seconds)


def record_error(
    model: str,
    endpoint: str,
    error_type: str
) -> None:
    """
    Record an error metric.
    
    Args:
        model: Model identifier
        endpoint: API endpoint
        error_type: Type of error (e.g., "timeout", "rate_limit", "api_error")
    """
    LLM_ERRORS_TOTAL.labels(
        model=model,
        endpoint=endpoint,
        error_type=error_type
    ).inc()


def update_rate_metrics(
    model: str,
    input_tokens_per_min: float,
    output_tokens_per_min: float,
    cost_per_hour: float,
    environment: str
) -> None:
    """
    Update rate-based gauge metrics.
    
    Args:
        model: Model identifier
        input_tokens_per_min: Input tokens per minute
        output_tokens_per_min: Output tokens per minute
        cost_per_hour: Cost per hour in USD
        environment: Environment name
    """
    LLM_TOKENS_PER_MINUTE.labels(
        model=model,
        type="input"
    ).set(input_tokens_per_min)
    
    LLM_TOKENS_PER_MINUTE.labels(
        model=model,
        type="output"
    ).set(output_tokens_per_min)
    
    LLM_COST_PER_HOUR.labels(
        environment=environment
    ).set(cost_per_hour)


def set_model_availability(
    model: str,
    provider: str,
    available: bool
) -> None:
    """
    Set model availability status.
    
    Args:
        model: Model identifier
        provider: Provider name (e.g., "openai", "anthropic")
        available: Whether model is available
    """
    LLM_MODEL_AVAILABLE.labels(
        model=model,
        provider=provider
    ).set(1 if available else 0)
