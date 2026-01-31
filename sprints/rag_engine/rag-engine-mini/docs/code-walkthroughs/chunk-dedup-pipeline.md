# 🚶‍♂️ Code Walkthrough: The Chunk Deduplication Pipeline

## 🗺️ The Path of a Chunk

This guide follows a text chunk from creation to storage, showing how the deduplication system prevents storing duplicate content in this RAG implementation.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Text      │    │  Chunking       │    │  Hashing        │
│   Input         │───▶│  Service        │───▶│  & Dedup        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐                                    ▼
│  Storage        │◀──────────────────────────────  Decision
│  Layer          │                                   │
│                 │                            ┌──────┴──────┐
│ • Vector DB    │                            │   Unique?   │
│ • Persistence  │                            │             │
└─────────────────┘                            │ Yes │  No   │
                                              └─────┼─────────┘
                                                    ▼
                                            ┌─────────────────┐
                                            │  Store Chunk    │
                                            │  (Vector & Meta)│
                                            └─────────────────┘
```

## 🧭 Step-by-Step Flow

### 1. Chunk Creation (`src/application/services/chunking.py`)

During document processing, text is split into chunks:

```python
def chunk_text_token_aware(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    separators: List[str] = ["\n\n", "\n", ". ", "! ", "? "]
) -> List[Chunk]:
    # Split text into semantic units
    # Each chunk has content, metadata, and position info
```

**Why this design?**
- Token-aware splitting respects LLM context limits
- Overlap preserves context across splits
- Semantic boundaries maintained

### 2. Hash Computation (`src/application/services/embedding_cache.py`)

Each chunk gets a unique hash based on its content:

```python
def compute_chunk_hash(content: str) -> str:
    """Compute SHA-256 hash of chunk content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
```

**Why SHA-256?**
- Cryptographically secure (low collision probability)
- Fast computation
- Consistent across systems

### 3. Deduplication Check (`src/adapters/persistence/postgres/repo_chunks.py`)

Before storing, check if chunk already exists:

```python
class ChunkDedupRepository(ChunkRepoPort):
    async def store_unique_chunks(self, chunks: List[Chunk]) -> List[ChunkId]:
        unique_chunks = []
        
        for chunk in chunks:
            chunk_hash = compute_chunk_hash(chunk.content)
            
            # Check if hash already exists
            if not await self.hash_exists(chunk_hash):
                unique_chunks.append(chunk)
                await self.store_chunk_with_hash(chunk, chunk_hash)
        
        return [chunk.id for chunk in unique_chunks]
```

**Why database-based dedup?**
- Persistent across system restarts
- ACID guarantees prevent race conditions
- Efficient indexing for fast lookups

### 4. Storage Decision

The system decides whether to store each chunk:

```python
# For each chunk:
chunk_hash = compute_chunk_hash(chunk.content)

if not hash_exists_in_db(chunk_hash):
    # Store chunk in vector DB
    vector_id = await vector_store.store(chunk.embedding, chunk.content)
    
    # Store metadata in PostgreSQL with hash
    await postgres_repo.store_chunk_metadata(
        chunk_id=vector_id,
        chunk_hash=chunk_hash,
        document_id=chunk.document_id,
        # ... other metadata
    )
else:
    # Skip storing - duplicate detected
    continue
```

**Why store hash in metadata?**
- Fast duplicate detection
- Audit trail of all processed content
- Ability to identify near-duplicates later

## 🎯 Database Schema

### Chunk Store Table (`src/adapters/persistence/postgres/models_chunk_store.py`)

```python
class ChunkStoreRow(Base):
    __tablename__ = "chunk_store"
    
    id = Column(Integer, primary_key=True, index=True)
    content_hash = Column(String, unique=True, index=True)  # For deduplication
    content = Column(Text)  # Original content (optional, for debugging)
    document_id = Column(String, index=True)  # Link to source document
    created_at = Column(DateTime, default=datetime.utcnow)
```

### Document-Chunk Relationship (`src/adapters/persistence/postgres/models_chunk_store.py`)

```python
class DocumentChunkRow(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, index=True)
    chunk_id = Column(Integer, index=True)  # References chunk_store.id
    chunk_index = Column(Integer)  # Order of chunk in document
    created_at = Column(DateTime, default=datetime.utcnow)
```

**Why separate tables?**
- `chunk_store`: Single source of truth for unique content
- `document_chunks`: Track relationships between docs and chunks
- Normalized schema prevents data duplication

## 🧪 Deduplication Algorithm

### 1. Hash-Based Approach

```python
async def store_chunks_with_dedup(chunks: List[Chunk]) -> StoreResult:
    new_chunks = []
    existing_hashes = set()
    
    # First, check all hashes in a single query for efficiency
    all_hashes = [compute_chunk_hash(c.content) for c in chunks]
    existing_hashes = await self.get_existing_hashes(all_hashes)
    
    for chunk, chunk_hash in zip(chunks, all_hashes):
        if chunk_hash not in existing_hashes:
            # This is a new chunk, store it
            await self._store_single_chunk(chunk, chunk_hash)
            new_chunks.append(chunk)
        else:
            # Duplicate detected, skip
            continue
    
    return StoreResult(
        stored_count=len(new_chunks),
        duplicate_count=len(chunks) - len(new_chunks)
    )
```

**Why batch checking?**
- Reduces database round trips
- More efficient than individual checks
- Better performance for bulk operations

### 2. Near-Duplicate Detection (Future Enhancement)

```python
# Conceptual - not implemented but possible enhancement
def detect_near_duplicates(chunk: Chunk, threshold: float = 0.95) -> bool:
    """Detect chunks that are similar but not identical."""
    # Could use fuzzy matching or semantic similarity
    # Useful for detecting slightly modified content
    pass
```

## 🎯 Key Design Decisions

### 1. Exact Match Deduplication
- SHA-256 hash ensures only exact duplicates removed
- Prevents false positives
- Simple and reliable

### 2. Database-Level Enforcement
- Unique constraint on hash column prevents duplicates at DB level
- Race condition protection
- Consistent across all application instances

### 3. Performance Optimization
- Hash indexing for fast lookups
- Batch operations to minimize DB calls
- Caching frequently accessed hashes

### 4. Transparency
- Log duplicate detection for monitoring
- Maintain statistics on deduplication effectiveness
- Allow inspection of stored hashes for debugging

## 🧪 Debugging Tips

1. **Duplicates still getting through?** Check if hash computation is consistent
2. **Slow ingestion?** Verify database indexes on hash columns
3. **Storage not decreasing?** Confirm deduplication is working as expected
4. **Hash collisions?** Though rare with SHA-256, monitor for unexpected behavior

## 📊 Measuring Effectiveness

The system tracks deduplication metrics:

```python
class DeduplicationMetrics:
    def __init__(self):
        self.total_chunks_processed = 0
        self.unique_chunks_stored = 0
        self.duplicates_removed = 0
        
    @property
    def deduplication_rate(self) -> float:
        if self.total_chunks_processed == 0:
            return 0.0
        return self.duplicates_removed / self.total_chunks_processed
```

**Typical expectations:**
- 10-30% deduplication rate in most document collections
- Higher rates with repetitive content (contracts, policies)
- Lower rates with diverse, unique content

## 📚 Further Exploration

- `src/adapters/persistence/postgres/repo_chunks.py` - Full deduplication implementation
- `src/application/services/chunking.py` - Chunk creation logic
- `src/core/config.py` - Deduplication-related configuration
- `src/application/ports/chunk_repo.py` - Deduplication interface definition