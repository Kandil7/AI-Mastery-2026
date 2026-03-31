# ADR-004: Minimal Vector Payload vs Rich Metadata Storage

## Status
Accepted

## Context
When storing documents in vector databases, we face a trade-off between storing rich metadata in the vector payload versus keeping payloads minimal and storing metadata separately in a relational database.

Options include:
- **Rich Vector Payloads**: Store all document metadata in the vector database
- **Minimal Vector Payloads**: Store only essential IDs in vector database, metadata elsewhere
- **Hybrid Approach**: Store frequently accessed metadata in vector database

## Decision
Use minimal vector payloads, storing only essential identifiers in the vector database and keeping rich metadata in PostgreSQL.

## Alternatives Considered
### Rich Vector Payloads
- Pros: Single-query retrieval, simpler architecture
- Cons: Higher storage costs, larger vectors, slower transfers

### Minimal Vector Payloads (Chosen)
- Pros: Lower storage costs, faster vector operations, normalized data
- Cons: Additional database join required

### Hybrid Approach
- Pros: Balanced performance and storage
- Cons: More complex architecture, potential inconsistency

## Rationale
1. **Storage Efficiency**: Vectors are expensive to store, especially with high-dimensional embeddings
2. **Performance**: Smaller payloads mean faster retrieval and transfer
3. **Consistency**: Centralized metadata management in PostgreSQL
4. **Flexibility**: Easy to modify metadata schema without touching vector database

## Consequences
### Positive
- Reduced vector database storage costs
- Faster vector operations
- Centralized metadata management
- Easier to maintain referential integrity

### Negative
- Requires additional database join for full document retrieval
- Slightly more complex retrieval logic
- Potential for data inconsistency if not properly managed

## Implementation
Vector database stores:
- `doc_id`: Unique identifier linking to PostgreSQL
- `embedding`: The vector representation
- Minimal metadata needed for filtering

PostgreSQL stores:
- Full document content
- Rich metadata (author, date, source, etc.)
- Relationships between documents

## Validation
This approach is commonly used in production RAG systems where cost and performance are critical factors.