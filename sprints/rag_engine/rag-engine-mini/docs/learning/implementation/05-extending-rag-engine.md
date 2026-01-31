# Extending the RAG Engine: Custom Components Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Extending LLM Adapters](#extending-llm-adapters)
4. [Adding New Embedding Providers](#adding-new-embedding-providers)
5. [Custom Vector Stores](#custom-vector-stores)
6. [Extending Document Processors](#extending-document-processors)
7. [Building Custom Services](#building-custom-services)
8. [Creating New Use Cases](#creating-new-use-cases)
9. [Testing Custom Extensions](#testing-custom-extensions)
10. [Best Practices](#best-practices)

---

## Introduction

The RAG Engine Mini is designed with extensibility in mind, allowing developers to add custom components that integrate seamlessly with the existing architecture. This guide explains how to extend the system by implementing new adapters, services, and use cases while maintaining the clean architecture principles.

### Extension Points

The system provides several extension points:

- **LLM Adapters**: Connect to new LLM providers
- **Embedding Providers**: Add new embedding models
- **Vector Stores**: Integrate different vector databases
- **Document Processors**: Support new document formats
- **Services**: Implement custom business logic
- **Use Cases**: Create new application workflows

### Architecture Principles

Extensions should adhere to the same architectural principles as the core system:

1. **Dependency Inversion**: Depend on abstractions, not concretions
2. **Single Responsibility**: Each component should have one clear purpose
3. **Open/Closed Principle**: Extend behavior without modifying existing code
4. **Consistent Interfaces**: Follow established patterns and contracts

---

## Architecture Overview

### Clean Architecture Layers

```
┌─────────────────┐    ← API Layer (FastAPI routes)
│   API Layer     │
└─────────────────┘
┌─────────────────┐    ← Application Layer (Use Cases & Services)
│ Application     │
│   Layer         │
└─────────────────┘
┌─────────────────┐    ← Domain Layer (Entities & Interfaces)
│   Domain        │
│   Layer         │
└─────────────────┘
┌─────────────────┐    ← Adapters Layer (External implementations)
│  Adapters       │
└─────────────────┘
```

### Port and Adapter Pattern

The system uses the Ports and Adapters pattern to ensure loose coupling:

```python
# Port (interface) - defines contract
class LLMPort(Protocol):
    def generate(self, prompt: str, **kwargs) -> str:
        ...

# Adapter (implementation) - provides concrete implementation
class OpenAILLM:
    def generate(self, prompt: str, **kwargs) -> str:
        # Implementation details
        ...
```

### Dependency Injection Container

The system uses a dependency injection container to wire components together:

```python
# From src/core/bootstrap.py
def get_container() -> Container:
    container = Container()
    
    # Register adapters
    container["llm"] = OpenAILLM(...)
    container["vector_store"] = QdrantAdapter(...)
    
    # Register services and use cases
    container["ask_use_case"] = AskQuestionHybridUseCase(...)
    
    return container
```

---

## Extending LLM Adapters

### Creating a New LLM Adapter

To add a new LLM provider, implement the [LLMPort](file:///k:/learning/technical/ai-ml/AI-Mastery-2026/sprints/rag_engine/rag-engine-mini/src/application/ports/llm_port.py#L20-L21):

```python
from src.application.ports.llm_port import LLMPort
from typing import AsyncIterator

class CustomLLM(LLMPort):
    def __init__(self, api_key: str, model: str = "custom-model"):
        self._api_key = api_key
        self._model = model
        # Initialize client for your LLM provider
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Implement synchronous generation
        # Use your provider's API
        pass
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        # Implement streaming generation
        # Yield chunks as they arrive
        pass
```

### Registering the New Adapter

Add your adapter to the dependency injection container:

```python
# In src/core/bootstrap.py
def get_container() -> Container:
    container = Container()
    
    # Other registrations...
    
    if settings.llm_backend == "custom":
        container["llm"] = CustomLLM(
            api_key=settings.custom_llm_api_key,
            model=settings.custom_llm_model
        )
    
    return container
```

### Configuration Support

Add configuration options to support your new LLM:

```python
# In src/core/config.py
class Settings:
    # Other settings...
    
    custom_llm_api_key: str = Field("", description="API key for custom LLM provider")
    custom_llm_model: str = Field("custom-model", description="Model name for custom LLM")
    llm_backend: str = Field("openai", description="LLM provider to use")
```

---

## Adding New Embedding Providers

### Implementing the EmbeddingsPort

Create a new embedding adapter by implementing [EmbeddingsPort](file:///k:/learning/technical/ai-ml/AI-Mastery-2026/sprints/rag_engine/rag-engine-mini/src/application/ports/embeddings_port.py#L19-L20):

```python
from src.application.ports.embeddings_port import EmbeddingsPort
from typing import List

class CustomEmbeddings(EmbeddingsPort):
    def __init__(self, api_key: str, model: str = "custom-embedding"):
        self._api_key = api_key
        self._model = model
        # Initialize client
    
    def embed(self, text: str) -> List[float]:
        # Generate embedding for single text
        # Return as list of floats
        pass
    
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        # Generate embeddings for multiple texts efficiently
        # Return as list of embedding vectors
        pass
```

### Adding Caching Support

The system provides caching through [CachedEmbeddings](file:///k:/learning/technical/ai-ml/AI-Mastery-2026/sprints/rag_engine/rag-engine-mini/src/adapters/embeddings/cached_embeddings.py#L34-L35). Your adapter will automatically benefit from this:

```python
# The system automatically wraps your adapter with caching
container["cached_embeddings"] = CachedEmbeddings(
    embeddings_port=container["embeddings"],
    cache=container["cache"]
)
```

---

## Custom Vector Stores

### Implementing VectorStorePort

Create a custom vector store by implementing [VectorStorePort](file:///k:/learning/technical/ai-ml/AI-Mastery-2026/sprints/rag_engine/rag-engine-mini/src/application/ports/vector_store.py#L23-L24):

```python
from src.application.ports.vector_store import VectorStorePort
from typing import List, Dict, Any, Optional
from src.domain.entities import DocumentId, TenantId

class CustomVectorStore(VectorStorePort):
    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        # Initialize your vector store client
    
    async def search(
        self,
        query_vector: List[float],
        limit: int,
        tenant_id: str,
        document_id: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        # Implement vector search
        # Return list of results with id, score, and payload
        pass
    
    async def upsert_points(
        self,
        ids: List[str],
        vectors: List[List[float]],
        tenant_id: str,
        document_id: str,
        payloads: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        # Insert or update vectors in the store
        pass
    
    async def delete_points(
        self,
        ids: List[str],
        tenant_id: str,
        document_id: Optional[str] = None
    ) -> bool:
        # Delete vectors by IDs
        pass
```

### Considerations for Vector Stores

When implementing a custom vector store, consider:

- **Filtering**: Support for metadata filters
- **Tenant Isolation**: Ensure proper data separation
- **Performance**: Efficient indexing and search
- **Scalability**: Ability to handle large collections
- **Consistency**: Reliable write/read operations

---

## Extending Document Processors

### Adding New File Type Support

To support new document types, extend the document processing pipeline:

```python
from src.application.ports.extraction_port import ExtractionPort
from src.domain.entities import ExtractedText

class CustomExtractor(ExtractionPort):
    def extract(self, file_path: str, content_type: str) -> ExtractedText:
        if content_type != "application/custom-format":
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Extract text from your custom format
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Process and clean the content
        processed_content = self._process_custom_format(content)
        
        return ExtractedText(text=processed_content, metadata={})
    
    def _process_custom_format(self, content: str) -> str:
        # Implement custom processing logic
        pass
```

### Registering the Extractor

Register your extractor in the container:

```python
# In bootstrap.py
def get_container():
    container = Container()
    
    # Register based on configuration
    if settings.custom_extractor_enabled:
        container["text_extractor"] = CustomExtractor()
    
    return container
```

---

## Building Custom Services

### Creating Domain Services

Domain services contain pure business logic that doesn't fit into entities:

```python
from typing import List, Dict, Any
from src.domain.entities import TenantId, DocumentId

class CustomProcessingService:
    """Service for custom processing logic."""
    
    def __init__(self, llm_port, vector_store):
        self._llm = llm_port
        self._vector_store = vector_store
    
    def perform_analysis(
        self, 
        tenant_id: TenantId, 
        document_id: DocumentId, 
        analysis_type: str
    ) -> Dict[str, Any]:
        """Perform custom analysis on a document."""
        # Implement your custom business logic
        # Use injected dependencies as needed
        pass
```

### Integrating with Use Cases

Use your custom service in a use case:

```python
from src.application.ports.use_case_port import UseCasePort

class AnalyzeDocumentUseCase(UseCasePort):
    def __init__(self, custom_service, document_repo):
        self._custom_service = custom_service
        self._document_repo = document_repo
    
    async def execute(self, request):
        # Validate request
        # Call your custom service
        result = self._custom_service.perform_analysis(
            tenant_id=request.tenant_id,
            document_id=request.document_id,
            analysis_type=request.analysis_type
        )
        
        # Return result
        return result
```

---

## Creating New Use Cases

### Implementing Use Case Port

Create new application workflows by implementing use cases:

```python
from src.application.ports.use_case_port import UseCasePort
from pydantic import BaseModel
from typing import List, Dict, Any

class DocumentAnalysisRequest(BaseModel):
    document_id: str
    analysis_types: List[str]
    tenant_id: str

class DocumentAnalysisResponse(BaseModel):
    analysis_results: Dict[str, Any]
    document_id: str

class DocumentAnalysisUseCase(UseCasePort):
    def __init__(self, analysis_service, document_repo):
        self._analysis_service = analysis_service
        self._document_repo = document_repo
    
    async def execute(self, request: DocumentAnalysisRequest) -> DocumentAnalysisResponse:
        # Validate document exists
        doc = self._document_repo.get_document(
            tenant_id=request.tenant_id,
            document_id=request.document_id
        )
        
        if not doc:
            raise ValueError("Document not found")
        
        # Perform analysis
        results = {}
        for analysis_type in request.analysis_types:
            result = await self._analysis_service.analyze(
                document_id=request.document_id,
                analysis_type=analysis_type
            )
            results[analysis_type] = result
        
        return DocumentAnalysisResponse(
            analysis_results=results,
            document_id=request.document_id
        )
```

### Adding API Endpoints

Expose your use case through an API endpoint:

```python
# In src/api/v1/routes_analysis.py
from fastapi import APIRouter, Depends
from src.api.v1.deps import get_tenant_id
from src.core.bootstrap import get_container

router = APIRouter()

@router.post("/analyze")
async def analyze_document(
    request: DocumentAnalysisRequest,
    tenant_id: str = Depends(get_tenant_id)
) -> DocumentAnalysisResponse:
    container = get_container()
    use_case = container["document_analysis_use_case"]
    
    # Update request with tenant_id
    request.tenant_id = tenant_id
    
    return await use_case.execute(request)
```

### Registering the Endpoint

Add your new endpoint to the main app:

```python
# In src/main.py
from src.api.v1.routes_analysis import router as analysis_router

app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["analysis"])
```

---

## Testing Custom Extensions

### Unit Tests

Write unit tests for your custom components:

```python
# tests/unit/test_custom_llm.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.adapters.llm.custom_llm import CustomLLM

def test_custom_llm_generate():
    # Arrange
    llm = CustomLLM(api_key="test-key")
    mock_client = Mock()
    llm._client = mock_client
    mock_client.generate.return_value = "Test response"
    
    # Act
    result = llm.generate("Test prompt")
    
    # Assert
    assert result == "Test response"
    mock_client.generate.assert_called_once_with("Test prompt")

@pytest.mark.asyncio
async def test_custom_llm_generate_stream():
    # Arrange
    llm = CustomLLM(api_key="test-key")
    mock_client = AsyncMock()
    llm._client = mock_client
    mock_client.generate_stream.return_value = ["chunk1", "chunk2"]
    
    # Act
    chunks = []
    async for chunk in llm.generate_stream("Test prompt"):
        chunks.append(chunk)
    
    # Assert
    assert chunks == ["chunk1", "chunk2"]
```

### Integration Tests

Test your extension integrated with the system:

```python
# tests/integration/test_custom_extension.py
import pytest
from src.core.bootstrap import get_container

@pytest.mark.asyncio
async def test_custom_use_case_integration():
    # Arrange
    container = get_container()
    
    # Ensure your custom component is registered
    assert "custom_component" in container
    
    # Get your use case
    use_case = container["document_analysis_use_case"]
    
    # Act
    request = DocumentAnalysisRequest(
        document_id="test-doc",
        analysis_types=["sentiment", "entities"],
        tenant_id="test-tenant"
    )
    result = await use_case.execute(request)
    
    # Assert
    assert result.analysis_results is not None
```

### Testing Guidelines

- **Mock external dependencies** to isolate your component
- **Test error conditions** and edge cases
- **Verify interface compliance** with ports
- **Check integration points** with other system components
- **Test configuration handling** and validation

---

## Best Practices

### 1. Follow Existing Patterns

- Use the same architectural patterns as the core system
- Follow naming conventions and code structure
- Implement proper error handling consistently

### 2. Maintain Type Safety

- Use type hints for all public interfaces
- Leverage Pydantic for request/response validation
- Ensure compatibility with existing type definitions

### 3. Implement Proper Error Handling

- Create custom exception types for domain-specific errors
- Follow the existing error handling patterns
- Provide meaningful error messages

### 4. Consider Performance Implications

- Optimize for the common case
- Implement appropriate caching strategies
- Consider async implementations for I/O-bound operations

### 5. Document Your Extensions

- Add docstrings to all public methods
- Create usage examples
- Document configuration options
- Explain any specific requirements

### 6. Plan for Testing

- Design components to be easily testable
- Provide test doubles for external dependencies
- Include integration tests with the broader system

### 7. Maintain Security Standards

- Follow secure coding practices
- Implement proper input validation
- Ensure tenant isolation is maintained
- Protect against injection attacks

### 8. Consider Observability

- Add appropriate logging
- Implement metrics collection
- Consider tracing for complex operations
- Follow existing logging patterns

---

## Example: Complete Extension Implementation

Here's a complete example of extending the RAG Engine with a custom sentiment analysis service:

```python
# src/adapters/sentiment/custom_sentiment.py
from typing import Dict, Any
from src.application.ports.sentiment_port import SentimentPort  # Hypothetical port

class CustomSentimentAnalyzer(SentimentPort):
    def __init__(self, api_key: str):
        self._api_key = api_key
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        # Implementation
        pass

# src/application/services/sentiment_service.py
class SentimentAnalysisService:
    def __init__(self, sentiment_port: SentimentPort):
        self._sentiment = sentiment_port
    
    def get_document_sentiment(self, document_text: str) -> Dict[str, Any]:
        return self._sentiment.analyze_sentiment(document_text)

# Register in bootstrap.py
# container["sentiment_analyzer"] = CustomSentimentAnalyzer(settings.sentiment_api_key)
# container["sentiment_service"] = SentimentAnalysisService(container["sentiment_analyzer"])
```

This comprehensive approach ensures that extensions integrate seamlessly with the existing architecture while maintaining the quality and consistency standards of the RAG Engine Mini project.