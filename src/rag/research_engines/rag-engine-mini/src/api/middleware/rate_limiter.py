"""
Rate Limiting Middleware
=============================
Redis-based token bucket rate limiting.

تقييد معدل الطلب باستخدام الحاوية
"""

import time
import asyncio
from typing import Optional, Callable, Any
from fastapi import Request, Response, HTTPException, status

from redis import Redis
from src.core.config import settings

# ============================================================================
# Rate Limiting Configuration
# ============================================================================

# Token bucket parameters
DEFAULT_RATE_LIMIT = 100  # Requests per window
DEFAULT_WINDOW_SECONDS = 60  # Time window (1 minute)
DEFAULT_BURST_LIMIT = 10  # Short burst limit
DEFAULT_BURST_WINDOW_SECONDS = 10  # Burst window (10 seconds)

# Redis key prefix
RATE_LIMIT_PREFIX = "ratelimit:"


class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    detail = "Rate limit exceeded"


class TokenBucket:
    """
    Token bucket rate limiting algorithm.

    Concept:
    - Bucket starts with capacity tokens
    - Each request consumes 1 token
    - Tokens refill at constant rate
    - If bucket is empty, request is rate limited

    Benefits:
    - Smooth rate enforcement (no hard cutoffs)
    - Allows short bursts
    - Prevents sustained high rates

    حاوية الرمز: خوارزمية مرنة وفعالة
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
    ) -> None:
        """
        Initialize token bucket.

        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens per second
        """
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.time()

    def consume(self, tokens: int) -> bool:
        """
        Attempt to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume (usually 1 per request)

        Returns:
            True if tokens available, False otherwise

        Algorithm:
        1. Calculate time elapsed since last refill
        2. Refill tokens based on elapsed time
        3. Check if enough tokens available
        4. Consume if enough, return success/failure
        """
        now = time.time()
        elapsed = now - self._last_refill

        # Refill tokens
        refill_tokens = elapsed * self._refill_rate
        self._tokens = min(self._capacity, self._tokens + refill_tokens)
        self._last_refill = now

        # Check if enough tokens
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def get_available_tokens(self) -> int:
        """Get current number of available tokens."""
        return int(self._tokens)


class RedisRateLimiter:
    """
    Redis-backed distributed rate limiting.

    Features:
    - Distributed rate limiting across multiple API instances
    - Per-endpoint rate limits
    - Per-tenant rate limits
    - Token bucket algorithm
    - Sliding window for burst handling

    Redis Key Structure:
    - ratelimit:{tenant_id}:{endpoint}: Request count
    - ratelimit:{tenant_id}:{endpoint}:burst: Burst allowance
    - ratelimit:{tenant_id}:block: Temporary block list

    مقيّد معدل موزّع مع Redis
    """

    def __init__(
        self,
        redis_client: Redis,
        default_rate_limit: int = DEFAULT_RATE_LIMIT,
        default_window_seconds: int = DEFAULT_WINDOW_SECONDS,
        default_burst_limit: int = DEFAULT_BURST_LIMIT,
        default_burst_window_seconds: int = DEFAULT_BURST_WINDOW_SECONDS,
    ) -> None:
        """
        Initialize Redis rate limiter.

        Args:
            redis_client: Redis client instance
            default_rate_limit: Requests per window
            default_window_seconds: Window size in seconds
            default_burst_limit: Short-term burst limit
            default_burst_window_seconds: Burst window size
        """
        self._redis = redis_client
        self._default_rate = default_rate_limit
        self._default_window = default_window_seconds
        self._default_burst = default_burst_limit
        self._default_burst_window = default_burst_window_seconds

    def _get_key(self, tenant_id: str, endpoint: str) -> str:
        """Get Redis key for rate limiting."""
        return f"{RATE_LIMIT_PREFIX}{tenant_id}:{endpoint}"

    async def is_rate_limited(
        self,
        tenant_id: str,
        endpoint: str,
    ) -> tuple[bool, dict]:
        """
        Check if request is rate limited.

        Uses token bucket algorithm with Redis for persistence.

        Args:
            tenant_id: Tenant/user identifier
            endpoint: API endpoint path

        Returns:
            Tuple of (is_limited, headers) where headers include retry info
        """
        key = self._get_key(tenant_id, endpoint)

        # Check if blocked
        block_key = f"{key}:block"
        if await self._redis.exists(block_key):
            # Check block expiration
            ttl = await self._redis.ttl(block_key)
            return True, {
                "X-RateLimit-Limit": str(self._default_rate),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + ttl),
                "Retry-After": str(ttl),
            }

        # Get current count
        current = await self._redis.incr(key)
        if current == 1:
            # First request in window, set expiry
            await self._redis.expire(key, self._default_window)

        # Check burst limit
        burst_key = f"{key}:burst"
        current_burst = await self._redis.incr(burst_key)
        if current_burst == 1:
            await self._redis.expire(burst_key, self._default_burst_window)

        # Check if over limits
        if current > self._default_rate or current_burst > self._default_burst:
            # Calculate retry-after time
            ttl = await self._redis.ttl(key)
            reset_time = int(time.time()) + ttl

            return True, {
                "X-RateLimit-Limit": str(self._default_rate),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_time),
                "Retry-After": str(ttl),
            }

        remaining = self._default_rate - current
        return False, {
            "X-RateLimit-Limit": str(self._default_rate),
            "X-RateLimit-Remaining": str(max(0, remaining)),
            "X-RateLimit-Reset": str(int(time.time()) + self._default_window),
            "X-RateLimit-Window": str(self._default_window),
        }

    async def block_tenant(
        self,
        tenant_id: str,
        endpoint: str,
        duration: int = 3600,  # 1 hour default
    ) -> None:
        """
        Temporarily block a tenant from all endpoints.

        Use for:
        - Abusive behavior
        - Fraud prevention
        - Emergency measures

        Args:
            tenant_id: Tenant ID to block
            endpoint: Specific endpoint (optional, blocks all if None)
            duration: Block duration in seconds
        """
        # Block all endpoints if no specific endpoint
        if endpoint is None:
            endpoints = ["*"]
        else:
            endpoints = [endpoint]

        for ep in endpoints:
            key = f"{RATE_LIMIT_PREFIX}{tenant_id}:{ep}:block"
            await self._redis.set(key, "1", ex=duration)

    async def unblock_tenant(self, tenant_id: str, endpoint: str = None) -> None:
        """Remove tenant block."""
        endpoints = ["*"] if endpoint is None else [endpoint]
        for ep in endpoints:
            key = f"{RATE_LIMIT_PREFIX}{tenant_id}:{ep}:block"
            await self._redis.delete(key)

    async def increment_rate_limit_violation(
        self,
        tenant_id: str,
        endpoint: str,
    ) -> None:
        """
        Track rate limit violations for monitoring/alerting.

        Args:
            tenant_id: Tenant ID
            endpoint: API endpoint
        """
        violation_key = f"{RATE_LIMIT_PREFIX}{tenant_id}:{endpoint}:violations"
        await self._redis.incr(violation_key)
        await self._redis.expire(violation_key, 3600)  # Track for 1 hour


async def rate_limit_middleware(
    tenant_id: str,
    endpoint: str,
):
    """
    Factory function to create rate limiting middleware.

    Args:
        tenant_id: Tenant/user ID from authentication
        endpoint: API endpoint path (e.g., /api/v1/queries/ask)

    Returns:
        FastAPI middleware function
    """
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    limiter = RedisRateLimiter(redis_client=redis)

    async def middleware(request: Request, call_next: Callable[[Any], Any]) -> Response:
        """
        FastAPI middleware function.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response or passes to next handler
        """
        # Check rate limit
        is_limited, headers = await limiter.is_rate_limited(tenant_id, endpoint)

        if is_limited:
            # Track violation
            await limiter.increment_rate_limit_violation(tenant_id, endpoint)

            # Return 429 with rate limit headers
            response = Response(
                content={"detail": "Rate limit exceeded"},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers=headers,
            )
            return response

        # Not rate limited, proceed
        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response

    return middleware


# ============================================================================
# Global Rate Limiter Instance
# ============================================================================

_limiter: RedisRateLimiter | None = None


def get_rate_limiter() -> RedisRateLimiter:
    """Get global rate limiter instance."""
    global _limiter
    if _limiter is None:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        _limiter = RedisRateLimiter(redis_client=redis)
    return _limiter
