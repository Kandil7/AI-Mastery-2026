# 🚶‍♂️ Code Walkthrough: Document Ingestion & Indexing Pipeline

## 🗺️ The Path of a Document

This guide follows a document from upload to indexed search results, showing how the ingestion pipeline works in this RAG implementation.

## 🏗️ Architecture Overview

```
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   Document  │    │   API Layer          │    │ Application     │
│   Upload    │───▶│  (FastAPI)           │───▶│   Services      │
└─────────────┘    └──────────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌─────────────────┐    └─────────┼─────────┐
│  Infrastructure │    │   Domain        │              │         │
│   Layers       │◀───┤   Models        │◀─────────────┼─────────┤
│                │    │                 │              │         │
│ • Vector DB    │    │ • Entities      │              │         │
│ • Persistence  │    │ • Value Objects │              │         │
│ • File Store   │    │ • Business Rules│              │         │
└─────────────────┘    └─────────────────┘              │         │
                                                        │         │
                                        ┌───────────────▼─────────┴────────┐
                                        │        Core Logic                  │
                                        │                                  │
                                        │ • Chunking                       │
                                        │ • Embeddings                     │
                                        │ • Deduplication                  │
                                        │ • Validation                     │
                                        └──────────────────────────────────┘
```

## 🧭 Step-by-Step Flow

### 1. Document Upload (`src/api/v1/routes_documents.py`)

When a user uploads a document, it hits the `/documents/upload` endpoint:

```python
@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    use_case: UploadDocumentUseCase = Depends(get_upload_document_usecase)
):
    request = UploadDocumentRequest(
        file=file,
        user_id=user_id,
        document_type=file.content_type
    )
    result = await use_case.execute(request)
    return result
```

**Why this design?**
- File validation happens at the API layer
- Authentication/authorization handled via dependency
- Clean separation between HTTP concerns and business logic

### 2. Use Case Orchestration (`src/application/use_cases/upload_document.py`)

The `UploadDocumentUseCase` coordinates the entire ingestion process:

```python
class UploadDocumentUseCase:
    async def execute(self, request: UploadDocumentRequest) -> UploadDocumentResponse:
        # Step 1: Validate file and check idempotency
        # Step 2: Save file to storage
        # Step 3: Extract text from document
        # Step 4: Chunk the text
        # Step 5: Generate embeddings for chunks
        # Step 6: Store chunks in vector database
        # Step 7: Store document metadata
        # Step 8: Return success response
```

**Why this design?**
- Single responsibility: handles the entire ingestion workflow
- Idempotency: prevents duplicate processing
- Error handling: manages failures gracefully
- Transactional: ensures data consistency

### 3. File Storage (`src/adapters/filestore/local_store.py`)

The uploaded file is saved to the configured file store:

```python
class LocalFileStore(FileStorePort):
    async def save_file(self, file: UploadFile, file_id: str) -> str:
        # Save file to local storage
        # Return file path for later retrieval
```

**Why separate file storage?**
- Scalability: can switch to cloud storage later
- Security: centralized file handling
- Efficiency: avoid storing large files in database

### 4. Text Extraction (`src/adapters/extraction/default_extractor.py`)

The document is parsed to extract text content:

```python
class DefaultTextExtractor(TextExtractorPort):
    async def extract_text(self, file_path: str, file_type: str) -> str:
        # Handle different file types (PDF, DOCX, TXT, etc.)
        # Extract text content preserving structure when possible
```

**Why separate extraction?**
- Support multiple document formats
- Handle format-specific parsing challenges
- Maintain clean separation of concerns

### 5. Chunking (`src/application/services/chunking.py`)

The extracted text is split into searchable chunks:

```python
def chunk_text_token_aware(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    separators: List[str] = ["\n\n", "\n", ". ", "! ", "? "]
) -> List[Chunk]:
    # Split text respecting token limits and semantic boundaries
    # Apply overlap to preserve context across splits
```

**Why token-aware chunking?**
- Respects LLM context limits
- Preserves semantic boundaries
- Overlap maintains context across splits

### 6. Embedding Generation (`src/adapters/embeddings/openai_embeddings.py` or `local_embeddings.py`)

Each chunk gets converted to a vector representation:

```python
class OpenAIEmbeddings(EmbeddingsPort):
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Call OpenAI API to generate embeddings
        # Handle rate limiting and retries
```

**Why embedding abstraction?**
- Easy to swap between providers (OpenAI, local, etc.)
- Caching can be added transparently
- Consistent interface regardless of provider

### 7. Deduplication (`src/application/services/embedding_cache.py` and `src/adapters/persistence/postgres/repo_chunks.py`)

Before storing, check for duplicate chunks:

```python
class ChunkDedupRepository(ChunkRepoPort):
    async def store_unique_chunks(self, chunks: List[Chunk]) -> List[ChunkId]:
        # Compute hash of chunk content
        # Check if hash already exists in database
        # Only store new chunks
```

**Why deduplication?**
- Saves storage space
- Reduces processing time
- Improves retrieval quality

### 8. Vector Storage (`src/adapters/vector/qdrant_store.py`)

Chunks are stored in the vector database with their embeddings:

```python
class QdrantVectorStore(VectorStorePort):
    async def store_chunks(self, chunks: List[Chunk]) -> List[ChunkId]:
        # Store chunk content and embeddings in Qdrant
        # Associate with document ID for retrieval
```

**Why Qdrant?**
- High-performance vector similarity search
- Good scalability characteristics
- Supports metadata filtering

### 9. Metadata Storage (`src/adapters/persistence/postgres/repo_documents.py`)

Document metadata is stored in PostgreSQL:

```python
class DocumentRepository(DocumentRepoPort):
    async def save_document_metadata(self, document: Document) -> DocumentId:
        # Store document metadata in PostgreSQL
        # Track relationships between documents and chunks
```

**Why PostgreSQL for metadata?**
- ACID compliance for consistency
- Complex querying capabilities
- Good integration with other tools

## 🎯 Key Design Decisions

### 1. Idempotency
- File hash used to prevent duplicate processing
- Safe to retry uploads without creating duplicates

### 2. Asynchronous Processing
- Non-blocking operations throughout the pipeline
- Better resource utilization

### 3. Error Handling
- Each step validates its inputs
- Failures in one document don't affect others

### 4. Scalability
- Chunking allows parallel processing
- Separate storage systems for different needs

## 🧪 Debugging Tips

1. **Documents not searchable?** Check if chunking is working correctly
2. **Duplicates appearing?** Verify deduplication hash computation
3. **Slow ingestion?** Consider optimizing chunking or embedding generation
4. **Corrupted files?** Check text extraction error handling

## 📚 Further Exploration

- `src/workers/tasks.py` - Background processing for heavy tasks
- `src/core/config.py` - Ingestion-specific configuration options
- `src/application/services/chunking.py` - Different chunking strategies
- `src/application/ports/document_idempotency.py` - Idempotency implementation