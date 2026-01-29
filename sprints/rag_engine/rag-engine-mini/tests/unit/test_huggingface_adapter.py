import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.adapters.llm.huggingface_llm import HuggingFaceLLM

@pytest.fixture
def mock_huggingface_hub():
    with patch("src.adapters.llm.huggingface_llm.AsyncInferenceClient") as mock:
        yield mock

@pytest.mark.asyncio
async def test_huggingface_generate(mock_huggingface_hub):
    # Setup mock
    mock_client = AsyncMock()
    mock_huggingface_hub.return_value = mock_client
    
    # Mock return value for chat_completion
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Mocked HF Response"
    mock_client.chat_completion.return_value = mock_response
    
    adapter = HuggingFaceLLM(api_key="fake-key", model_name="mistral")
    
    response = await adapter.generate("Hello")
    
    assert response == "Mocked HF Response"
    mock_client.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_huggingface_generate_stream(mock_huggingface_hub):
    # Setup mock
    mock_client = AsyncMock()
    mock_huggingface_hub.return_value = mock_client
    
    # Mock streaming chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.choices[0].delta.content = "Hello "
    mock_chunk2 = MagicMock()
    mock_chunk2.choices[0].delta.content = "World"
    
    async def mock_stream(*args, **kwargs):
        yield mock_chunk1
        yield mock_chunk2

    mock_client.chat_completion.return_value = mock_stream()
    
    adapter = HuggingFaceLLM(api_key="fake-key")
    
    chunks = []
    async for chunk in adapter.generate_stream("Hi"):
        chunks.append(chunk)
        
    assert chunks == ["Hello ", "World"]
