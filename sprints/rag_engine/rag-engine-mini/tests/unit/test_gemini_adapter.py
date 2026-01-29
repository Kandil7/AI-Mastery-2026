import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.adapters.llm.gemini_llm import GeminiLLM

@pytest.fixture
def mock_genai():
    with patch("src.adapters.llm.gemini_llm.genai") as mock:
        yield mock

@pytest.mark.asyncio
async def test_gemini_generate(mock_genai):
    # Setup mock
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    
    # Mock return value for generate_content_async
    mock_response = MagicMock()
    mock_response.text = "Mocked Gemini Response"
    # generate_content_async is an async method
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    
    adapter = GeminiLLM(api_key="fake-key", model_name="gemini-1.5-flash")
    
    response = await adapter.generate("Hello")
    
    assert response == "Mocked Gemini Response"
    mock_model.generate_content_async.assert_called_once()

@pytest.mark.asyncio
async def test_gemini_generate_stream(mock_genai):
    # Setup mock
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    
    # Mock streaming chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Hello "
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "World"
    
    class AsyncGeneratorMock:
        def __init__(self, items):
            self.items = items
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

    # generate_content_async returns an awaitable that resolves to the generator
    mock_model.generate_content_async = AsyncMock(return_value=AsyncGeneratorMock([mock_chunk1, mock_chunk2]))
    
    adapter = GeminiLLM(api_key="fake-key")
    
    chunks = []
    async for chunk in adapter.generate_stream("Hi"):
        chunks.append(chunk)
        
    assert chunks == ["Hello ", "World"]
