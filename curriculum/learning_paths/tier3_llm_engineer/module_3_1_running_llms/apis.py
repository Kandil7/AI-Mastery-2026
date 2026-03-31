"""
LLM API Clients with Production-Ready Features

Provides unified interfaces for OpenAI, Anthropic, and Google APIs with:
- Automatic retry with exponential backoff
- Rate limiting and token budgeting
- Streaming support (SSE)
- Comprehensive error handling
- Structured logging and metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"


@dataclass
class TokenUsage:
    """Token usage metrics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0  # For providers with caching

    def __post_init__(self) -> None:
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    """Standardized LLM response structure."""

    content: str
    model: str
    usage: TokenUsage
    finish_reason: str
    raw_response: Dict[str, Any]
    latency_ms: float
    stream: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
        }


@dataclass
class StreamChunk:
    """A chunk from streaming response."""

    content: str
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000
    burst_limit: int = 10


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_attempts: int = 5
    min_wait_seconds: float = 1.0
    max_wait_seconds: float = 60.0
    exponential_base: float = 2.0
    retryable_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    retryable_exceptions: List[type] = field(default_factory=lambda: [
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
    ])


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "default",
        timeout: float = 120.0,
        retry_config: Optional[RetryConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.api_key = api_key or self._get_default_api_key()
        self.base_url = base_url or self._get_default_base_url()
        self.model = model
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.extra_headers = headers or {}

        self._request_count = 0
        self._token_count = 0
        self._last_request_time = 0.0

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=self._build_headers(),
        )

        logger.info(f"Initialized {self.__class__.__name__} with model={model}")

    @abstractmethod
    def _get_default_api_key(self) -> str:
        """Get API key from environment."""
        pass

    @abstractmethod
    def _get_default_base_url(self) -> str:
        """Get default base URL."""
        pass

    @abstractmethod
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        pass

    @abstractmethod
    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        stop: Optional[List[str]],
        stream: bool,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make the actual API request."""
        pass

    @abstractmethod
    def _parse_response(self, response_data: Dict[str, Any], latency_ms: float) -> LLMResponse:
        """Parse API response into standardized format."""
        pass

    @abstractmethod
    async def _parse_stream(self, response: httpx.Response) -> AsyncGenerator[StreamChunk, None]:
        """Parse streaming response."""
        pass

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()
        time_elapsed = current_time - self._last_request_time

        # Reset counters if minute has passed
        if time_elapsed >= 60:
            self._request_count = 0
            self._token_count = 0

        if self._request_count >= self.rate_limit_config.requests_per_minute:
            wait_time = 60 - time_elapsed
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                self._request_count = 0
                self._token_count = 0

    async def _rate_limit_wait(self) -> None:
        """Async rate limit check."""
        current_time = time.time()
        time_elapsed = current_time - self._last_request_time

        if time_elapsed >= 60:
            self._request_count = 0
            self._token_count = 0

        if self._request_count >= self.rate_limit_config.requests_per_minute:
            wait_time = 60 - time_elapsed
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._token_count = 0

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        reraise=True,
    )
    async def _request_with_retry(self, **kwargs: Any) -> httpx.Response:
        """Make request with automatic retry."""
        return await self._make_request(**kwargs)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncGenerator[StreamChunk, None]]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stop: Stop sequences
            stream: Whether to stream the response
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse or AsyncGenerator[StreamChunk, None]
        """
        await self._rate_limit_wait()

        start_time = time.time()
        self._request_count += 1

        try:
            response = await self._request_with_retry(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=stream,
                **kwargs,
            )

            if stream:
                return self._parse_stream(response)

            latency_ms = (time.time() - start_time) * 1000
            response_data = response.json()
            llm_response = self._parse_response(response_data, latency_ms)

            # Update token counts for rate limiting
            self._token_count += llm_response.usage.total_tokens

            logger.info(
                f"Generated response: model={llm_response.model}, "
                f"tokens={llm_response.usage.total_tokens}, "
                f"latency={latency_ms:.2f}ms"
            )

            return llm_response

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Generation failed: {type(e).__name__}: {e}")
            raise

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Simplified async generation with system prompt support.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Passed to generate()

        Returns:
            LLMResponse
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.generate(messages=messages, **kwargs)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "BaseLLMClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class OpenAIClient(BaseLLMClient):
    """OpenAI API client with GPT-4, GPT-3.5 support."""

    def _get_default_api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return key

    def _get_default_base_url(self) -> str:
        return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        stop: Optional[List[str]],
        stream: bool,
        **kwargs: Any,
    ) -> httpx.Response:
        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream,
        }

        if stop:
            payload["stop"] = stop

        # Add OpenAI-specific parameters
        payload.update(kwargs)

        if stream:
            response = await self._client.post(endpoint, json=payload, timeout=None)
        else:
            response = await self._client.post(endpoint, json=payload)

        response.raise_for_status()
        return response

    def _parse_response(self, response_data: Dict[str, Any], latency_ms: float) -> LLMResponse:
        choice = response_data["choices"][0]
        usage = response_data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"] or "",
            model=response_data.get("model", self.model),
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason", "unknown"),
            raw_response=response_data,
            latency_ms=latency_ms,
        )

    async def _parse_stream(self, response: httpx.Response) -> AsyncGenerator[StreamChunk, None]:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk_data = json.loads(data)
                    choice = chunk_data["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")

                    yield StreamChunk(
                        content=content,
                        finish_reason=choice.get("finish_reason"),
                    )
                except json.JSONDecodeError:
                    continue


class AnthropicClient(BaseLLMClient):
    """Anthropic API client with Claude support."""

    def _get_default_api_key(self) -> str:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return key

    def _get_default_base_url(self) -> str:
        return os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        headers.update(self.extra_headers)
        return headers

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        stop: Optional[List[str]],
        stream: bool,
        **kwargs: Any,
    ) -> httpx.Response:
        endpoint = f"{self.base_url}/v1/messages"

        # Convert OpenAI format to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if system_message:
            payload["system"] = system_message

        if stop:
            payload["stop_sequences"] = stop

        # Anthropic doesn't support frequency/presence penalty
        # Add any Anthropic-specific parameters
        payload.update(kwargs)

        if stream:
            response = await self._client.post(endpoint, json=payload, timeout=None)
        else:
            response = await self._client.post(endpoint, json=payload)

        response.raise_for_status()
        return response

    def _parse_response(self, response_data: Dict[str, Any], latency_ms: float) -> LLMResponse:
        content = ""
        if response_data.get("content"):
            for block in response_data["content"]:
                if block.get("type") == "text":
                    content += block.get("text", "")

        usage = response_data.get("usage", {})

        return LLMResponse(
            content=content,
            model=response_data.get("model", self.model),
            usage=TokenUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            ),
            finish_reason=response_data.get("stop_reason", "unknown"),
            raw_response=response_data,
            latency_ms=latency_ms,
        )

    async def _parse_stream(self, response: httpx.Response) -> AsyncGenerator[StreamChunk, None]:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                try:
                    event = json.loads(data)
                    event_type = event.get("type")

                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        content = delta.get("text", "")
                        yield StreamChunk(content=content)
                    elif event_type == "message_stop":
                        yield StreamChunk(content="", finish_reason="end_turn")
                except json.JSONDecodeError:
                    continue


class GoogleClient(BaseLLMClient):
    """Google AI (Gemini) API client."""

    def _get_default_api_key(self) -> str:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        return key

    def _get_default_base_url(self) -> str:
        return os.getenv(
            "GOOGLE_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta"
        )

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        stop: Optional[List[str]],
        stream: bool,
        **kwargs: Any,
    ) -> httpx.Response:
        # Convert messages to Gemini format
        contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}],
                })

        endpoint = f"{self.base_url}/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        if stop:
            payload["generationConfig"]["stopSequences"] = stop

        payload.update(kwargs)

        if stream:
            endpoint = f"{self.base_url}/models/{self.model}:streamGenerateContent"
            params["alt"] = "sse"
            response = await self._client.post(endpoint, json=payload, params=params, timeout=None)
        else:
            response = await self._client.post(endpoint, json=payload, params=params)

        response.raise_for_status()
        return response

    def _parse_response(self, response_data: Dict[str, Any], latency_ms: float) -> LLMResponse:
        candidates = response_data.get("candidates", [])
        content = ""

        if candidates:
            candidate = candidates[0]
            content_parts = candidate.get("content", {}).get("parts", [])
            content = "".join(part.get("text", "") for part in content_parts)

        # Google doesn't provide detailed token usage in all cases
        usage_metadata = response_data.get("usageMetadata", {})

        return LLMResponse(
            content=content,
            model=self.model,
            usage=TokenUsage(
                prompt_tokens=usage_metadata.get("promptTokenCount", 0),
                completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
                total_tokens=usage_metadata.get("totalTokenCount", 0),
            ),
            finish_reason=candidates[0].get("finishReason", "unknown") if candidates else "unknown",
            raw_response=response_data,
            latency_ms=latency_ms,
        )

    async def _parse_stream(self, response: httpx.Response) -> AsyncGenerator[StreamChunk, None]:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                try:
                    # Google streams JSON arrays
                    chunk_data = json.loads(data)
                    if isinstance(chunk_data, list):
                        chunk_data = chunk_data[0]

                    candidates = chunk_data.get("candidates", [])
                    if candidates:
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        content = "".join(part.get("text", "") for part in content_parts)
                        finish_reason = candidates[0].get("finishReason")
                        yield StreamChunk(content=content, finish_reason=finish_reason)
                except json.JSONDecodeError:
                    continue


def create_client(
    provider: Union[LLMProvider, str],
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: LLM provider (openai, anthropic, google)
        model: Model name
        api_key: API key (optional, can use env vars)
        base_url: Custom base URL
        **kwargs: Additional client configuration

    Returns:
        Configured LLM client

    Example:
        >>> client = create_client("openai", "gpt-4-turbo")
        >>> response = await client.generate_async("Hello!")
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    client_classes = {
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.GOOGLE: GoogleClient,
    }

    if provider not in client_classes:
        raise ValueError(f"Unsupported provider: {provider}")

    return client_classes[provider](
        api_key=api_key,
        base_url=base_url,
        model=model,
        **kwargs,
    )


# Synchronous wrapper for non-async contexts
class SyncLLMClient:
    """Synchronous wrapper for async LLM clients."""

    def __init__(self, client: BaseLLMClient) -> None:
        self._client = client
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous generation."""
        return self._loop.run_until_complete(
            self._client.generate(messages=messages, **kwargs)
        )

    def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous simplified generation."""
        return self._loop.run_until_complete(
            self._client.generate_async(prompt=prompt, system_prompt=system_prompt, **kwargs)
        )

    def close(self) -> None:
        """Close the client."""
        self._loop.run_until_complete(self._client.close())
        self._loop.close()

    def __enter__(self) -> "SyncLLMClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
