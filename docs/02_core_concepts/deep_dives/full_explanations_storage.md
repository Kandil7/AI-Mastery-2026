# Storage package (storage/*.py)

## storage/vectordb_base.py

### VectorStore (Protocol)
- Defines the interface for vector databases.
- Methods:
  - `upsert(chunks, embeddings, metadata)`
  - `query_by_vector(vector, top_k, filters)`

## storage/pgvector_store.py

### PgVectorStore
- Purpose: PostgreSQL + pgvector backend.
- `__init__(dsn, table, embedding_dim)`:
  - Stores config and ensures table/index.
- `_connect()`:
  - Connects to Postgres and registers vector type.
- `_ensure_table()`:
  - Creates table and IVFFLAT index if missing.
- `upsert(chunks, embeddings, metadata)`:
  - Inserts or updates rows for chunks.
- `query_by_vector(vector, top_k, filters)`:
  - Uses cosine distance and optional JSONB filter.
  - Returns `RetrievedChunk` list.

## storage/qdrant_store.py

### QdrantStore
- Purpose: Qdrant vector DB backend.
- `__init__(endpoint, collection)`:
  - Creates client.
- `_create_client()`:
  - Imports `qdrant_client` and instantiates it.
- `_ensure_collection(vector_size)`:
  - Creates collection if it does not exist.
- `upsert(...)`:
  - Upserts points with vector and payload.
- `query_by_vector(...)`:
  - Executes search with optional filters.
  - Returns `RetrievedChunk` list.

## storage/weaviate_store.py

### WeaviateStore
- Purpose: Weaviate backend.
- `__init__(endpoint, index_name)`:
  - Creates Weaviate client.
- `_create_client()`:
  - Imports and constructs client.
- `_ensure_schema(vector_size)`:
  - Creates class schema if needed.
- `upsert(...)`:
  - Adds data objects with vectors via batch.
- `query_by_vector(...)`:
  - Runs near-vector query with optional filters.
  - Returns `RetrievedChunk` list.
