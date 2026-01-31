# RAG Engine Mini - Application Layer Deep Dive

## Introduction

The application layer serves as the orchestration center of the RAG Engine Mini system. It contains use cases that implement business logic by coordinating domain entities and calling upon infrastructure adapters. This layer defines the ports (interfaces) that abstract away external dependencies, enabling loose coupling and testability.

## Key Components

### Use Cases

Use cases represent the primary business operations of the system. Each use case implements a specific business function by coordinating domain entities and infrastructure adapters.

#### AskQuestionHybridUseCase

This is the core use case that implements the hybrid RAG pipeline:

```python
class AskQuestionHybridUseCase:
    """
    Use case for answering questions with hybrid retrieval.

    Flow:
    1. Expand query (optional)
    2. Embed the question (cached)
    3. Vector search (Qdrant) - semantic similarity
    4. Hydrate vector results (get text from Postgres)
    5. Keyword search (Postgres FTS) - lexical match
    6. RRF fusion - merge results
    7. Rerank (Cross-Encoder) - precision boost
    8. Build prompt with guardrails
    9. Generate answer (LLM)
    10. Return answer with sources
    """
```

The hybrid RAG pipeline involves multiple sophisticated steps:

1. **Query Expansion**: Optionally uses an LLM to generate related queries
2. **Embedding**: Converts the query to a vector representation (with caching)
3. **Vector Search**: Finds semantically similar chunks in the vector store
4. **Keyword Search**: Finds lexically similar chunks in the database
5. **RRF Fusion**: Combines results from both search methods using Reciprocal Rank Fusion
6. **Reranking**: Improves precision using a cross-encoder model
7. **Prompt Building**: Constructs the final prompt with retrieved context
8. **Generation**: Generates the answer using an LLM
9. **Self-Critique**: Evaluates the relevance and accuracy of the answer

#### UploadDocumentUseCase

Handles the document upload workflow:

1. Stores the file in the file system
2. Creates a document record in the database
3. Queues the document for background indexing
4. Returns status information

### Ports (Interfaces)

The application layer defines ports that abstract infrastructure concerns. These are protocol classes that define contracts for external dependencies.

#### VectorStorePort

```python
class VectorStorePort(Protocol):
    """
    Port for vector storage and similarity search.

    Implementations: Qdrant, PGVector, Pinecone, etc.

    Design Decision: Minimal payload approach - only store IDs and metadata
    in vector store, hydrate text from Postgres for cost/storage efficiency.
    """
```

Key methods:
- `ensure_collection()`: Ensures the collection exists
- `upsert_points()`: Adds or updates vectors
- `search_scored()`: Performs similarity search
- `delete_by_document()`: Removes all vectors for a document

#### LLMPort

```python
class LLMPort(Protocol):
    """
    Port for Language Model operations.

    Implementations: OpenAI, Ollama, Gemini, Hugging Face, etc.
    """
```

Key methods:
- `generate()`: Generates text completion
- `generate_stream()`: Generates streaming text completion
- `embed()`: Creates embeddings for text

#### DocumentRepoPort

```python
class DocumentRepoPort(Protocol):
    """
    Port for document metadata operations.

    Implementation: PostgreSQL

    Design Decision: Separate from file storage.
    Documents table stores metadata, file system stores actual files.
    """
```

Key methods:
- `create_document()`: Creates a new document record
- `set_status()`: Updates document processing status
- `get_status()`: Retrieves document status
- `list_documents()`: Lists documents for a tenant
- `delete_document()`: Deletes a document

### Supporting Services

The application layer also includes supporting services that implement specific business capabilities:

#### CachedEmbeddings

Wraps embedding functionality with caching to improve performance and reduce costs:

```python
class CachedEmbeddings:
    """
    Wrapper around embeddings service that adds caching.
    
    Reduces API calls and improves response time by caching embeddings.
    """
```

#### QueryExpansionService

Expands user queries to improve retrieval effectiveness:

```python
class QueryExpansionService:
    """
    Expands queries using LLM to improve retrieval effectiveness.
    
    Generates related queries that might capture different aspects
    of the user's information need.
    """
```

#### SelfCritiqueService

Implements self-evaluation capabilities:

```python
class SelfCritiqueService:
    """
    Evaluates the quality of retrieved results and generated answers.
    
    Checks for relevance, accuracy, and potential hallucinations.
    """
```

## Design Patterns

### Dependency Injection

The application layer uses dependency injection to achieve loose coupling:

```python
def __init__(
    self,
    *,
    cached_embeddings: CachedEmbeddings,
    vector_store: VectorStorePort,
    keyword_store: KeywordStorePort,
    chunk_text_reader: ChunkTextReaderPort,
    reranker: RerankerPort,
    llm: LLMPort,
    query_expansion_service: object | None = None,
    self_critique: object | None = None,
    router: object | None = None,
    privacy: object | None = None,
    search_tool: object | None = None,
) -> None:
```

Dependencies are passed through the constructor, making the use case flexible and testable.

### Ports and Adapters

The ports-and-adapters pattern (also known as hexagonal architecture) is implemented through Python protocols:

```python
class VectorStorePort(Protocol):
    def search_scored(
        self,
        *,
        query_vector: list[float],
        tenant_id: TenantId,
        top_k: int,
        document_id: str | None = None,
    ) -> Sequence[ScoredChunkResult]:
        ...
```

This allows the application layer to remain independent of specific implementations while still defining clear contracts.

### Request/Response Pattern

Use cases typically follow a request/response pattern with dedicated data classes:

```python
@dataclass
class AskHybridRequest:
    """Request data for hybrid RAG question."""

    tenant_id: str
    question: str
    document_id: str | None = None
    k_vec: int = 30
    k_kw: int = 30
    fused_limit: int = 40
    rerank_top_n: int = 8
    expand_query: bool = False
```

### Error Handling

The application layer handles errors gracefully, often with retry mechanisms and fallback strategies:

```python
try:
    # Main business logic
    answer = self._llm.generate(prompt, temperature=0.1)
except Exception as e:
    # Fallback strategy
    strict_prompt = prompt + "\n\nSTRICT: Answer ONLY using provided facts."
    answer = self._llm.generate(strict_prompt, temperature=0.0)
```

## Observability Integration

The application layer integrates observability throughout:

- **Metrics**: Tracks API latency, token usage, cache hits, retrieval scores
- **Tracing**: Provides distributed tracing of the entire RAG pipeline
- **Logging**: Structured logging with correlation IDs

```python
API_REQUEST_LATENCY.labels(method="ask", endpoint="/ask").observe(time.time() - start_time)
TOKEN_USAGE.labels(model="default", type="prompt").inc(len(prompt) // 4)
```

## Multi-Tenancy Implementation

All operations are tenant-aware, ensuring data isolation:

```python
def search_scored(
    self,
    *,
    query_vector: list[float],
    tenant_id: TenantId,  # Tenant isolation
    top_k: int,
    document_id: str | None = None,
) -> Sequence[ScoredChunkResult]:
```

Filters are applied at every level to ensure tenants can only access their own data.

## Advanced Features

### Semantic Routing

The system can route queries based on intent:

```python
if self._router:
    intent = self._router.route(request.question)
    if intent == QueryIntent.CHITCHAT:
        # Handle chit-chat differently
        answer_text = self._llm.generate(f"Respond politely to: {request.question}")
        return Answer(text=answer_text, sources=[])
```

### Privacy Protection

PII redaction and restoration:

```python
# Step 0: Privacy Guard (Redaction)
original_question = request.question
if self._privacy:
    request.question = self._privacy.redact(request.question)

# ... rest of pipeline ...

# Step 5: Restore Privacy (De-redaction)
if self._privacy:
    answer_text = self._privacy.restore(answer_text)
    self._privacy.clear()
```

### Web Search Fallback

When retrieval is insufficient, the system can fall back to web search:

```python
if grade == "irrelevant" and self._search:
    web_results = self._search.search(request.question)
    # Convert web results to Source Chunks
    web_chunks = [
        Chunk(id=f"web_{i}", tenant_id=TenantId("web"), document_id=None,
              text=f"[{r['title']}]({r['url']})\n{r['content']}")
        for i, r in enumerate(web_results)
    ]
    reranked = web_chunks if web_chunks else reranked
```

## Testing Strategy

The application layer is designed for comprehensive testing:

1. **Unit Tests**: Test business logic in isolation using mock adapters
2. **Integration Tests**: Test coordination between components
3. **Contract Tests**: Ensure adapters conform to port specifications

The dependency injection and ports/adapters pattern make it possible to test business logic without external dependencies.