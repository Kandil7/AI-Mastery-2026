# ADR-005: Chunk Deduplication Design

## Status
Accepted

## Context
In RAG systems, documents are often chunked into smaller pieces for retrieval. When ingesting overlapping or similar documents, we may end up with duplicate or near-duplicate chunks that waste storage and processing resources while potentially affecting retrieval quality.

We need to decide how to handle chunk deduplication:
- **No Deduplication**: Allow all chunks regardless of duplication
- **Simple Hash-Based**: Use content hashing to identify duplicates
- **Semantic Deduplication**: Use embeddings to identify semantically similar chunks

## Decision
Implement hash-based deduplication at the chunk level using content hashes stored in PostgreSQL.

## Alternatives Considered
### No Deduplication
- Pros: Simplest implementation, fastest ingestion
- Cons: Wastes storage and processing, may affect retrieval quality

### Hash-Based Deduplication (Chosen)
- Pros: Fast, deterministic, exact matches guaranteed
- Cons: Won't catch semantically similar but differently phrased content

### Semantic Deduplication
- Pros: Catches paraphrased content
- Cons: Computationally expensive, potential false positives/negatives

## Rationale
1. **Performance**: Hash-based comparison is extremely fast
2. **Determinism**: Identical content will always be detected
3. **Simplicity**: Easy to implement and maintain
4. **Storage Savings**: Significant reduction in duplicate storage
5. **Reliability**: Deterministic approach with predictable behavior

## Consequences
### Positive
- Significant storage savings
- Faster retrieval (fewer irrelevant duplicates)
- Cleaner result sets
- Predictable behavior

### Negative
- Won't catch paraphrased content
- Requires additional database storage for hashes
- Minor overhead during ingestion

## Implementation
```python
# During ingestion
chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()
if not chunk_repo.exists(chunk_hash):
    # Store the chunk and its hash
    chunk_repo.store(chunk_content, chunk_hash, doc_id)
```

The hash serves as a unique identifier to prevent storing identical chunks.

## Validation
Hash-based deduplication is widely used in document processing systems and provides a good balance between effectiveness and efficiency.