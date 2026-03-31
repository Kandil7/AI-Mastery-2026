# LLM Adapter Testing Guide

## Overview

Testing LLM (Large Language Model) adapters requires a specialized approach due to the unique characteristics of LLM APIs: network latency, rate limits, streaming responses, and non-deterministic outputs.

This guide covers comprehensive testing strategies for LLM adapters including error handling, retries, timeouts, rate limiting, and edge cases.

## Table of Contents

1. [Why LLM Testing is Different](#why-llm-testing-is-different)
2. [Test Categories](#test-categories)
3. [Mocking Strategies](#mocking-strategies)
4. [Error Handling Tests](#error-handling-tests)
5. [Retry Logic Tests](#retry-logic-tests)
6. [Rate Limiting Tests](#rate-limiting-tests)
7. [Streaming Tests](#streaming-tests)
8. [Edge Cases](#edge-cases)
9. [Best Practices](#best-practices)
10. [CI/CD Integration](#cicd-integration)

## Why LLM Testing is Different

### 1. External API Dependencies

Unlike internal components, LLM adapters depend on third-party APIs:
- **OpenAI** (GPT-4, GPT-3.5)
- **Google** (Gemini)
- **HuggingFace** (Inference API)
- **Anthropic** (Claude)
- **Local models** (Ollama)

**Challenges:**
- API availability and reliability
- Rate limits and quotas
- Network latency and timeouts
- Costs for real API calls

### 2. Non-Deterministic Responses

Same input can produce different outputs:
```python
# First call
response1 = await llm.generate("What is 2+2?")
# Returns: "2+2 equals 4."

# Second call
response2 = await llm.generate("What is 2+2?")
# Returns: "The answer is 4."
```

**Solution:** Test behavior, not exact content.

### 3. Streaming Complexity

Modern LLMs support streaming responses:
```python
async for chunk in llm.generate_stream("Tell me a story"):
    print(chunk, end="")
    # Output: "Once... upon... a... time..."
```

**Challenges:**
- Chunks can be empty
- Connection drops mid-stream
- Buffering and assembly

### 4. Error Scenarios

LLM APIs can fail in many ways:
- **Rate limits** (429 Too Many Requests)
- **Authentication errors** (401 Unauthorized)
- **Invalid requests** (400 Bad Request)
- **Server errors** (500, 502, 503)
- **Timeouts** (network, request, streaming)

## Test Categories

### 1. Unit Tests (with mocks)

Test adapter logic without external calls:

```python
@pytest.mark.asyncio
async def test_generate_with_mock():
    with patch("openai.AsyncOpenAI") as mock_client:
        # Setup mock
        mock_client.return_value.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Response"))])
        )
        
        adapter = OpenAILLM(api_key="test")
        result = await adapter.generate("Hello")
        
        assert result == "Response"
```

### 2. Integration Tests (with real API)

Test with actual API calls (limited, guarded):

```python
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
async def test_generate_real_api():
    adapter = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"))
    result = await adapter.generate("Say 'test'")
    
    assert "test" in result.lower()
    assert len(result) > 0
```

### 3. Contract Tests

Verify API response structure:

```python
def test_response_contract():
    """Verify response has expected fields."""
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "choices": [{
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop"
        }]
    }
    
    assert "choices" in response
    assert len(response["choices"]) > 0
    assert "content" in response["choices"][0]["message"]
```

## Mocking Strategies

### Strategy 1: MagicMock with Spec

```python
from unittest.mock import MagicMock, AsyncMock

# Create mock with realistic structure
mock_response = MagicMock()
mock_response.choices = [MagicMock()]
mock_response.choices[0].message = MagicMock()
mock_response.choices[0].message.content = "Generated text"

# For async methods
mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
```

### Strategy 2: Async Generators for Streaming

```python
async def mock_stream():
    chunks = ["Hello ", "world", "!"]
    for chunk in chunks:
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content=chunk))])

mock_client.chat.completions.create.return_value = mock_stream()
```

### Strategy 3: Side Effects for Sequential Calls

```python
# First call fails, second succeeds
mock_client.chat.completions.create.side_effect = [
    Exception("First call failed"),
    mock_success_response,
]
```

### Strategy 4: Context Managers

```python
@pytest.fixture
def mock_openai():
    with patch("src.adapters.llm.openai_llm.AsyncOpenAI") as mock:
        client = AsyncMock()
        mock.return_value = client
        yield client
```

## Error Handling Tests

### HTTP Status Code Tests

Test each HTTP error type:

```python
class TestHTTPStatusCodes:
    """Test handling of different HTTP status codes."""
    
    @pytest.mark.parametrize("status_code,error_class,description", [
        (400, BadRequestError, "Bad Request - malformed request"),
        (401, AuthenticationError, "Unauthorized - invalid API key"),
        (403, PermissionDeniedError, "Forbidden - insufficient permissions"),
        (404, NotFoundError, "Not Found - model doesn't exist"),
        (429, RateLimitError, "Too Many Requests - rate limit exceeded"),
        (500, InternalServerError, "Internal Server Error"),
        (502, BadGatewayError, "Bad Gateway"),
        (503, ServiceUnavailableError, "Service Unavailable"),
        (504, GatewayTimeoutError, "Gateway Timeout"),
    ])
    async def test_http_error_handling(self, mock_client, status_code, error_class, description):
        """Test that each HTTP status code raises appropriate error."""
        
        # Setup mock to return specific error
        mock_client.chat.completions.create.side_effect = error_class(
            f"HTTP {status_code}",
            response=MagicMock(status_code=status_code),
            body={"error": {"message": f"Error {status_code}"}}
        )
        
        adapter = OpenAILLM(api_key="test")
        
        with pytest.raises(LLMError) as exc_info:
            await adapter.generate("Hello")
        
        # Verify error details
        assert status_code in str(exc_info.value) or error_class.__name__ in str(exc_info.value)
```

### Network Error Tests

```python
class TestNetworkErrors:
    """Test handling of network-related errors."""
    
    async def test_connection_error(self, mock_client):
        """Test handling of connection failures."""
        from aiohttp import ClientConnectionError
        
        mock_client.chat.completions.create.side_effect = ClientConnectionError(
            "Cannot connect to host"
        )
        
        adapter = OpenAILLM(api_key="test")
        
        with pytest.raises(LLMError) as exc_info:
            await adapter.generate("Hello")
        
        assert "connection" in str(exc_info.value).lower()
    
    async def test_timeout_error(self, mock_client):
        """Test handling of timeout errors."""
        import asyncio
        
        mock_client.chat.completions.create.side_effect = asyncio.TimeoutError()
        
        adapter = OpenAILLM(api_key="test")
        
        with pytest.raises(LLMError) as exc_info:
            await adapter.generate("Hello")
        
        assert "timeout" in str(exc_info.value).lower()
    
    async def test_dns_resolution_error(self, mock_client):
        """Test handling of DNS resolution failures."""
        mock_client.chat.completions.create.side_effect = Exception(
            "getaddrinfo failed"
        )
        
        adapter = OpenAILLM(api_key="test")
        
        with pytest.raises(LLMError) as exc_info:
            await adapter.generate("Hello")
        
        assert "dns" in str(exc_info.value).lower() or "network" in str(exc_info.value).lower()
```

### Response Validation Tests

```python
class TestResponseValidation:
    """Test validation of API responses."""
    
    async def test_missing_choices_field(self, mock_client):
        """Test handling of response missing choices."""
        mock_response = MagicMock()
        mock_response.choices = []  # Empty choices
        
        mock_client.chat.completions.create.return_value = mock_response
        
        adapter = OpenAILLM(api_key="test")
        
        with pytest.raises(LLMError) as exc_info:
            await adapter.generate("Hello")
        
        assert "empty" in str(exc_info.value).lower() or "choices" in str(exc_info.value).lower()
    
    async def test_null_content(self, mock_client):
        """Test handling of null content in response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        
        mock_client.chat.completions.create.return_value = mock_response
        
        adapter = OpenAILLM(api_key="test")
        
        # Should handle gracefully - return empty string or raise error
        result = await adapter.generate("Hello")
        assert result == "" or result is None
    
    async def test_truncated_response(self, mock_client):
        """Test handling of responses with finish_reason='length'."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Partial response..."
        mock_response.choices[0].finish_reason = "length"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        adapter = OpenAILLM(api_key="test")
        
        result = await adapter.generate("Hello")
        
        # Should return content but indicate truncation
        assert result == "Partial response..."
        # Optionally: log warning about truncation
```

## Retry Logic Tests

### Exponential Backoff Tests

```python
class TestRetryLogic:
    """Test retry logic with exponential backoff."""
    
    async def test_retry_on_500_error(self, mock_client):
        """Test retry on 500 Internal Server Error."""
        from openai import InternalServerError
        
        # Fail twice, then succeed
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Success"
        
        mock_client.chat.completions.create.side_effect = [
            InternalServerError("Server error 1", response=MagicMock(status_code=500), body={}),
            InternalServerError("Server error 2", response=MagicMock(status_code=500), body={}),
            mock_response,
        ]
        
        adapter = OpenAILLM(api_key="test", max_retries=3)
        
        result = await adapter.generate("Hello")
        
        assert result == "Success"
        assert mock_client.chat.completions.create.call_count == 3
    
    async def test_no_retry_on_400_error(self, mock_client):
        """Test no retry on 400 Bad Request (client error)."""
        from openai import BadRequestError
        
        mock_client.chat.completions.create.side_effect = BadRequestError(
            "Bad request",
            response=MagicMock(status_code=400),
            body={}
        )
        
        adapter = OpenAILLM(api_key="test", max_retries=3)
        
        with pytest.raises(LLMError):
            await adapter.generate("Hello")
        
        # Should only be called once (no retries for 4xx)
        assert mock_client.chat.completions.create.call_count == 1
    
    async def test_max_retries_exceeded(self, mock_client):
        """Test error raised after max retries exhausted."""
        from openai import InternalServerError
        
        # Always fail
        mock_client.chat.completions.create.side_effect = InternalServerError(
            "Persistent error",
            response=MagicMock(status_code=503),
            body={}
        )
        
        adapter = OpenAILLM(api_key="test", max_retries=2)
        
        with pytest.raises(LLMError) as exc_info:
            await adapter.generate("Hello")
        
        # Should have tried: initial + 2 retries = 3 calls
        assert mock_client.chat.completions.create.call_count == 3
        assert "max retries" in str(exc_info.value).lower() or "persistent error" in str(exc_info.value)
    
    async def test_exponential_backoff_timing(self, mock_client):
        """Test that exponential backoff delays are applied."""
        import time
        from openai import InternalServerError
        
        mock_client.chat.completions.create.side_effect = [
            InternalServerError("Error 1", response=MagicMock(status_code=503), body={}),
            InternalServerError("Error 2", response=MagicMock(status_code=503), body={}),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Success"))]),
        ]
        
        adapter = OpenAILLM(api_key="test", max_retries=2, retry_delay=1.0)
        
        start = time.time()
        await adapter.generate("Hello")
        elapsed = time.time() - start
        
        # With exponential backoff: 1s + 2s = 3s minimum delay
        assert elapsed >= 2.5  # Allow some tolerance
```

## Rate Limiting Tests

### 429 Error Handling

```python
class TestRateLimiting:
    """Test handling of rate limit errors."""
    
    async def test_rate_limit_error_429(self, mock_client):
        """Test proper handling of 429 Too Many Requests."""
        from openai import RateLimitError
        
        mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=MagicMock(
                status_code=429,
                headers={"Retry-After": "60"}
            ),
            body={"error": {"message": "Rate limit exceeded"}}
        )
        
        adapter = OpenAILLM(api_key="test")
        
        with pytest.raises(RateLimitError) as exc_info:
            await adapter.generate("Hello")
        
        # Error should include retry information
        assert "rate limit" in str(exc_info.value).lower()
    
    async def test_rate_limit_with_retry_after(self, mock_client):
        """Test that Retry-After header is parsed and respected."""
        from openai import RateLimitError
        
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        
        mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={}
        )
        
        adapter = OpenAILLM(api_key="test")
        
        try:
            await adapter.generate("Hello")
        except RateLimitError as e:
            # Should have retry_after attribute or in message
            assert hasattr(e, 'retry_after') or "60" in str(e)
    
    async def test_rate_limit_quota_exceeded(self, mock_client):
        """Test handling of quota exceeded errors."""
        mock_client.chat.completions.create.side_effect = Exception(
            "You exceeded your current quota"
        )
        
        adapter = OpenAILLM(api_key="test")
        
        with pytest.raises(LLMError) as exc_info:
            await adapter.generate("Hello")
        
        assert "quota" in str(exc_info.value).lower()
```

## Streaming Tests

### Basic Streaming Tests

```python
class TestStreaming:
    """Test streaming response handling."""
    
    async def test_streaming_success(self, mock_client):
        """Test successful streaming response."""
        
        async def mock_stream():
            chunks = ["Hello ", "world", "!"]
            for chunk in chunks:
                yield MagicMock(
                    choices=[MagicMock(delta=MagicMock(content=chunk))]
                )
        
        mock_client.chat.completions.create.return_value = mock_stream()
        
        adapter = OpenAILLM(api_key="test")
        
        chunks = []
        async for chunk in adapter.generate_stream("Hi"):
            chunks.append(chunk)
        
        assert chunks == ["Hello ", "world", "!"]
    
    async def test_streaming_empty_chunks(self, mock_client):
        """Test handling of empty chunks in stream."""
        
        async def mock_stream():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))])
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=""))])  # Empty
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=" World"))])
        
        mock_client.chat.completions.create.return_value = mock_stream()
        
        adapter = OpenAILLM(api_key="test")
        
        chunks = []
        async for chunk in adapter.generate_stream("Hi"):
            chunks.append(chunk)
        
        # Should filter or handle empty chunks
        assert "Hello" in chunks
        assert " World" in chunks
    
    async def test_streaming_error_mid_stream(self, mock_client):
        """Test error handling during streaming."""
        from aiohttp import ClientConnectionError
        
        async def mock_stream_with_error():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Partial "))])
            raise ClientConnectionError("Connection lost")
        
        mock_client.chat.completions.create.return_value = mock_stream_with_error()
        
        adapter = OpenAILLM(api_key="test")
        
        chunks = []
        with pytest.raises(LLMError):
            async for chunk in adapter.generate_stream("Hi"):
                chunks.append(chunk)
        
        # Should have received partial content before error
        assert len(chunks) >= 1
        assert "Partial " in chunks
    
    async def test_streaming_timeout(self, mock_client):
        """Test timeout handling during streaming."""
        import asyncio
        
        async def slow_stream():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Start"))])
            await asyncio.sleep(10)  # Simulate slow stream
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="End"))])
        
        mock_client.chat.completions.create.return_value = slow_stream()
        
        adapter = OpenAILLM(api_key="test", timeout=1.0)
        
        chunks = []
        with pytest.raises(LLMError) as exc_info:
            async for chunk in adapter.generate_stream("Hi"):
                chunks.append(chunk)
        
        assert "timeout" in str(exc_info.value).lower()
```

## Edge Cases

### Input Validation

```python
class TestInputEdgeCases:
    """Test edge cases for input handling."""
    
    async def test_very_long_input(self, mock_client):
        """Test handling of very long input (near token limit)."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        
        adapter = OpenAILLM(api_key="test")
        
        # Create input near typical token limit (~100KB)
        long_input = "word " * 20000
        
        result = await adapter.generate(long_input)
        
        # Should either truncate or handle without crashing
        assert result is not None
    
    async def test_special_characters(self, mock_client):
        """Test handling of special characters and Unicode."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        
        adapter = OpenAILLM(api_key="test")
        
        special_inputs = [
            "Hello 🎉🎊",  # Emojis
            "你好世界",  # Chinese
            "مرحبا",  # Arabic
            "<>&\"'",  # HTML entities
            "\n\n\r\t",  # Whitespace
            "🔒🔑🔐",  # More emojis
        ]
        
        for test_input in special_inputs:
            result = await adapter.generate(test_input)
            assert result is not None
    
    async def test_empty_input(self, mock_client):
        """Test handling of empty input."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "I need more information"
        mock_client.chat.completions.create.return_value = mock_response
        
        adapter = OpenAILLM(api_key="test")
        
        result = await adapter.generate("")
        
        # Should handle gracefully
        assert result is not None
    
    async def test_none_input(self, mock_client):
        """Test handling of None input."""
        adapter = OpenAILLM(api_key="test")
        
        with pytest.raises((LLMError, TypeError)):
            await adapter.generate(None)
```

### Configuration Edge Cases

```python
class TestConfigurationEdgeCases:
    """Test edge cases for adapter configuration."""
    
    def test_zero_temperature(self):
        """Test that temperature=0 is valid (deterministic)."""
        adapter = OpenAILLM(api_key="test", temperature=0.0)
        assert adapter.temperature == 0.0
    
    def test_high_temperature(self):
        """Test that temperature=2 is valid (very random)."""
        adapter = OpenAILLM(api_key="test", temperature=2.0)
        assert adapter.temperature == 2.0
    
    def test_zero_max_tokens(self):
        """Test that max_tokens=0 is handled."""
        # Should either reject or handle gracefully
        with pytest.raises((ValueError, LLMError)):
            OpenAILLM(api_key="test", max_tokens=0)
    
    def test_very_high_max_tokens(self):
        """Test very high max_tokens."""
        adapter = OpenAILLM(api_key="test", max_tokens=100000)
        assert adapter.max_tokens == 100000
```

## Best Practices

### 1. Always Use Async Mocks

```python
# ✅ Correct - use AsyncMock for async methods
mock_client.chat.completions.create = AsyncMock(return_value=response)

# ❌ Wrong - regular Mock won't work with await
mock_client.chat.completions.create = Mock(return_value=response)
```

### 2. Test Error Messages

```python
async def test_error_includes_details(self, mock_client):
    mock_client.chat.completions.create.side_effect = Exception("API Error: Invalid model")
    
    adapter = OpenAILLM(api_key="test")
    
    with pytest.raises(LLMError) as exc_info:
        await adapter.generate("Hello")
    
    # Verify error message is informative
    assert "API Error" in str(exc_info.value)
    assert "Invalid model" in str(exc_info.value)
```

### 3. Test Both Success and Failure Paths

```python
# Every test should have both positive and negative cases
async def test_generate():
    # Success case
    mock_client.chat.completions.create = AsyncMock(return_value=success_response)
    result = await adapter.generate("Hello")
    assert result == "Response"
    
    # Failure case
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    with pytest.raises(LLMError):
        await adapter.generate("Hello")
```

### 4. Use Parametrized Tests

```python
@pytest.mark.parametrize("error_code", [500, 502, 503, 504])
async def test_server_errors(mock_client, error_code):
    """Test all server error codes with one test."""
    # Setup error
    # Test handling
    pass
```

### 5. Mock at the Right Level

```python
# ✅ Good - mock the external library
with patch("openai.AsyncOpenAI") as mock:
    pass

# ❌ Bad - mock internal methods (brittle)
with patch.object(adapter, "_make_request") as mock:
    pass
```

### 6. Test Async Generators Properly

```python
async def test_streaming():
    async def mock_stream():
        for chunk in ["A", "B", "C"]:
            yield chunk
    
    mock_client.chat.completions.create.return_value = mock_stream()
    
    # Collect all chunks
    chunks = []
    async for chunk in adapter.generate_stream("Hi"):
        chunks.append(chunk)
    
    assert chunks == ["A", "B", "C"]
```

## CI/CD Integration

### Test Organization

```
tests/
├── unit/
│   ├── test_huggingface_adapter.py  # Mock-based
│   ├── test_gemini_adapter.py       # Mock-based
│   └── test_openai_adapter.py       # Mock-based
├── integration/
│   └── test_llm_integration.py      # Real API calls (limited)
└── conftest.py
```

### Pytest Configuration

```python
# conftest.py
import pytest

@pytest.fixture
def mock_openai_client():
    """Provide mocked OpenAI client."""
    with patch("src.adapters.llm.openai_llm.AsyncOpenAI") as mock:
        yield mock

# Markers
pytestmark = [
    pytest.mark.asyncio,
]

# Skip integration tests by default
def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests with real APIs"
    )

# Skip markers
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="Need --run-integration flag")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
```

### GitHub Actions

```yaml
name: LLM Adapter Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run unit tests (mocked)
        run: pytest tests/unit/test_*_adapter.py -v --tb=short
  
  integration-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'  # Only run nightly
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run integration tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/integration/test_llm_integration.py --run-integration -v
```

## Summary

### Test Coverage Checklist

- [ ] Basic functionality (generate, stream)
- [ ] All HTTP error codes (400, 401, 403, 429, 500, 502, 503)
- [ ] Network errors (connection, DNS, timeout)
- [ ] Retry logic (exponential backoff, max retries)
- [ ] Rate limiting (429 errors, quota exceeded)
- [ ] Streaming (chunks, errors mid-stream, timeouts)
- [ ] Edge cases (empty input, long input, special characters)
- [ ] Configuration (temperature, max_tokens, timeout)
- [ ] Response validation (null content, missing fields)

### Key Takeaways

1. **Mock external APIs** - Don't make real calls in unit tests
2. **Test all error scenarios** - LLM APIs are unreliable
3. **Verify error messages** - Should be informative
4. **Test async generators** - Streaming requires special handling
5. **Use parametrized tests** - Efficiently test multiple cases
6. **Separate integration tests** - Run real API calls sparingly
7. **Monitor test coverage** - Aim for >90% on error paths

### Additional Resources

- [OpenAI API Error Handling](https://platform.openai.com/docs/guides/error-codes)
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference/index)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Python Async Testing](https://realpython.com/async-io-python/)
