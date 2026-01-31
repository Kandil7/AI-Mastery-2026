# RAG Engine Mini - Adapters Layer Deep Dive

## Introduction

The adapters layer in RAG Engine Mini provides concrete implementations for the ports defined in the application layer. This layer handles all external dependencies including databases, vector stores, LLM providers, and file storage systems. The adapters layer implements the ports-and-adapters (hexagonal) architecture pattern, allowing the business logic to remain independent of external infrastructure.

## Architecture Overview

The adapters layer is organized into several categories:

1. **LLM Adapters**: Implementations for different LLM providers
2. **Embedding Adapters**: Text embedding implementations
3. **Vector Store Adapters**: Vector database implementations
4. **Persistence Adapters**: Database implementations
5. **File Storage Adapters**: File system implementations
6. **Caching Adapters**: Caching implementations
7. **Reranking Adapters**: Relevance ranking implementations
8. **Queue Adapters**: Background job queue implementations

## LLM Adapters

### OpenAILLM Adapter

The OpenAI adapter implements the `LLMPort` protocol for OpenAI's API:

```python
class OpenAILLM:
    """
    OpenAI LLM adapter implementing LLMPort.
    
    Provides access to OpenAI's language models like GPT-4, GPT-3.5, etc.
    """
```

Key features:
- Handles API authentication and rate limiting
- Implements both synchronous and streaming generation
- Manages model parameters (temperature, max_tokens, etc.)
- Provides error handling and retry logic

### OllamaLLM Adapter

The Ollama adapter enables local LLM usage:

```python
class OllamaLLM:
    """
    Ollama LLM adapter implementing LLMPort.
    
    Provides access to locally hosted open-source models.
    """
```

Benefits:
- No API costs
- Better privacy control
- Offline capability
- Custom model support

### GeminiLLM Adapter

Google's Gemini model adapter:

```python
class GeminiLLM:
    """
    Google Gemini LLM adapter implementing LLMPort.
    
    Provides access to Google's Gemini models.
    """
```

### HuggingFaceLLM Adapter

Adapter for Hugging Face models:

```python
class HuggingFaceLLM:
    """
    Hugging Face LLM adapter implementing LLMPort.
    
    Provides access to models hosted on Hugging Face Hub.
    """
```

## Embedding Adapters

### OpenAIEmbeddings

Standard OpenAI embeddings:

```python
class OpenAIEmbeddings:
    """
    OpenAI embeddings adapter implementing EmbeddingsPort.
    
    Creates vector representations using OpenAI's embedding models.
    """
```

### LocalEmbeddings

Local embedding models for privacy and cost control:

```python
class LocalEmbeddings:
    """
    Local embeddings adapter implementing EmbeddingsPort.
    
    Uses open-source models like Sentence Transformers for embeddings.
    """
```

## Vector Store Adapters

### QdrantVectorStore

The primary vector store adapter:

```python
class QdrantVectorStore:
    """
    Qdrant adapter implementing VectorStorePort.

    Design Decision: Minimal payload approach:
    - Only store tenant_id, document_id in payload
    - NO text in payload (saves storage, fetched from Postgres)
    """
```

Key design decisions:
- **Minimal Payload**: Only stores IDs in the vector store to save space
- **Text Hydration**: Text is fetched separately from the database
- **Tenant Isolation**: Filters by tenant_id to ensure data separation
- **Efficient Storage**: Reduces storage costs and improves performance

## Persistence Adapters

### Postgres Adapters

The system uses PostgreSQL for storing document metadata, chunks, and other structured data:

```python
class PostgresDocumentRepo:
    """
    PostgreSQL implementation of DocumentRepoPort.
    
    Stores document metadata and status information.
    """
```

Features:
- Document metadata storage
- Status tracking (created, queued, processing, indexed, failed)
- Tenant isolation through foreign key constraints
- Full-text search capabilities

### Placeholder Adapters

During development, placeholder implementations are used:

```python
class PlaceholderDocumentRepo:
    """
    In-memory implementation of DocumentRepoPort for development.
    
    Useful for rapid prototyping without external dependencies.
    """
```

## File Storage Adapters

### Local File Storage

Default implementation for storing uploaded files:

```python
class LocalFileStore:
    """
    Local file system adapter implementing FileStorePort.
    
    Stores files in the local file system with tenant isolation.
    """
```

### Cloud Storage Adapters

For production deployments, cloud storage options are available:

- **S3FileStore**: Amazon S3 integration
- **GCSFileStore**: Google Cloud Storage integration  
- **AzureBlobStore**: Microsoft Azure Blob Storage integration

## Caching Adapters

### RedisCache

Redis-based caching for improved performance:

```python
class RedisCache:
    """
    Redis adapter implementing CachePort.
    
    Provides fast caching for embeddings and other frequently accessed data.
    """
```

## Reranking Adapters

### CrossEncoderReranker

Advanced reranking using cross-encoder models:

```python
class CrossEncoderReranker:
    """
    Cross-encoder reranking adapter implementing RerankerPort.
    
    Improves retrieval precision by reranking results with a specialized model.
    """
```

### LLMReranker

Alternative reranking using LLMs:

```python
class LLMReranker:
    """
    LLM-based reranking adapter implementing RerankerPort.
    
    Uses language models to assess relevance and rerank results.
    """
```

## Queue Adapters

### CeleryTaskQueue

Background processing using Celery:

```python
class CeleryTaskQueue:
    """
    Celery adapter implementing TaskQueuePort.
    
    Handles background document indexing and other async operations.
    """
```

## Implementation Patterns

### Configuration-Driven Selection

The bootstrap module selects the appropriate adapter based on configuration:

```python
# LLM Choice
if settings.llm_backend == "ollama":
    from src.adapters.llm.ollama_llm import OllamaLLM
    llm = OllamaLLM(...)
elif settings.llm_backend == "gemini":
    from src.adapters.llm.gemini_llm import GeminiLLM
    llm = GeminiLLM(...)
else:
    llm = OpenAILLM(...)
```

### Error Handling and Fallbacks

Adapters implement robust error handling:

```python
try:
    from src.adapters.rerank.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker(...)
except ModuleNotFoundError:
    reranker = NoopReranker()  # Fallback to no-op implementation
```

### Resource Management

Adapters properly manage resources:

```python
def __init__(self, client: QdrantClient, ...):
    self._client = client  # Store client reference
    
# Proper cleanup is handled by the client library
```

## Multi-Tenancy Implementation

All adapters implement tenant isolation:

```python
def search_scored(
    self,
    *,
    query_vector: list[float],
    tenant_id: TenantId,  # Tenant isolation parameter
    top_k: int,
    document_id: str | None = None,
) -> Sequence[ScoredChunkResult]:
    # Apply tenant filter in query
    must = [
        FieldCondition(
            key="tenant_id",
            match=MatchValue(value=tenant_id.value),
        )
    ]
```

## Performance Optimizations

### Caching Strategies

- **Embedding Caching**: Frequently used embeddings are cached to reduce API calls
- **Query Result Caching**: Common queries can be cached for faster response
- **Connection Pooling**: Database and vector store connections are pooled

### Batch Operations

Adapters implement batch operations where possible:

```python
def upsert_points(
    self,
    *,
    ids: Sequence[str],
    vectors: Sequence[list[float]],
    tenant_id: str,
    document_id: str,
) -> None:
    # Process multiple vectors in a single operation
    points = []
    for point_id, vector in zip(ids, vectors):
        points.append(PointStruct(...))
    
    self._client.upsert(collection_name=self._collection, points=points)
```

## Security Considerations

### Authentication

- API keys are securely stored and transmitted
- Connection strings are properly validated
- Access controls are enforced at the infrastructure level

### Data Isolation

- Tenant IDs are validated in all operations
- Cross-tenant access is prevented through filtering
- File paths are sanitized to prevent directory traversal

## Testing and Validation

### Contract Testing

Adapters must conform to their respective port protocols:

```python
# All LLM adapters must implement the LLMPort interface
class LLMPort(Protocol):
    def generate(self, prompt: str, **kwargs) -> str: ...
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]: ...
```

### Integration Testing

Adapters are tested against real infrastructure when possible, with proper cleanup:

```python
def test_qdrant_adapter():
    # Setup test collection
    adapter = QdrantVectorStore(client, "test_collection", 1536)
    adapter.ensure_collection()
    
    # Test operations
    adapter.upsert_points(...)
    results = adapter.search_scored(...)
    
    # Cleanup
    # (cleanup code here)
```

## Extensibility

The adapter pattern makes it easy to add new implementations:

1. Define a new adapter class implementing the appropriate port
2. Register it in the bootstrap module
3. Add configuration options if needed
4. Update documentation

This allows the system to support additional LLM providers, vector stores, or databases without changing the core business logic.