# RAG Systems

Retrieval-Augmented Generation (RAG) systems combine large language models (LLMs) with external knowledge retrieval to provide more accurate, up-to-date, and contextually relevant responses. This document covers the database aspects of RAG system design.

## Overview

RAG systems address the limitations of LLMs by augmenting their generation capabilities with retrieved information from external sources. For senior AI/ML engineers, understanding the database requirements and patterns for RAG is essential for building production-grade AI applications.

## RAG Architecture Components

### 1. Document Ingestion Pipeline
- **Document processing**: PDF, HTML, text parsing
- **Chunking strategies**: Semantic vs fixed-size chunking
- **Embedding generation**: Text → vector embeddings
- **Metadata extraction**: Source, author, date, tags

### 2. Vector Database
- **Storage**: High-dimensional vectors + metadata
- **Indexing**: ANN search optimization
- **Retrieval**: Similarity search with filtering
- **Scalability**: Horizontal scaling for large corpora

### 3. Query Processing
- **Query embedding**: User query → vector
- **Hybrid search**: Vector + keyword + metadata filtering
- **Re-ranking**: Post-retrieval scoring and ordering
- **Context assembly**: Building prompt context from retrieved documents

### 4. LLM Integration
- **Prompt engineering**: Context injection strategies
- **Response generation**: LLM with retrieved context
- **Citation handling**: Source attribution in responses
- **Fallback mechanisms**: When retrieval fails

## Database Design Patterns

### Document Storage Schema
```sql
-- Core document table
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    source_url TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    published_date DATE,
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunk table (for fine-grained retrieval)
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id),
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    token_count INT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Metadata indexing for filtering
CREATE INDEX idx_documents_source ON documents(source_url);
CREATE INDEX idx_documents_date ON documents(published_date);
CREATE INDEX idx_documents_author ON documents(author);
CREATE INDEX idx_document_chunks_metadata ON document_chunks USING GIN (metadata);
```

### Hybrid Search Implementation
```sql
-- Vector search with metadata filtering
SELECT 
    d.id,
    d.title,
    d.content,
    1 - (d.embedding <=> $1) AS similarity,
    d.published_date,
    d.author
FROM documents d
WHERE 
    d.published_date >= '2024-01-01'
    AND d.author = 'John Doe'
    AND d.source_url LIKE '%tech%'
ORDER BY d.embedding <=> $1
LIMIT 10;

-- Multi-stage retrieval: coarse → fine
WITH coarse_results AS (
    SELECT id, embedding, similarity
    FROM documents
    ORDER BY embedding <=> $1
    LIMIT 100
),
fine_results AS (
    SELECT 
        cr.id,
        cr.similarity,
        d.title,
        d.content,
        d.metadata
    FROM coarse_results cr
    JOIN documents d ON cr.id = d.id
    WHERE d.published_date >= '2024-01-01'
      AND d.metadata->>'category' = 'technology'
)
SELECT * FROM fine_results
ORDER BY similarity
LIMIT 10;
```

## Performance Optimization

### Indexing Strategies
- **Composite indexes**: Embedding + metadata fields
- **Partial indexes**: For frequently queried subsets
- **Covering indexes**: Include commonly needed metadata
- **BRIN indexes**: For time-based filtering

### Caching Patterns
- **Query result caching**: Cache frequent query results
- **Embedding caching**: Cache expensive embedding computations
- **Context caching**: Cache assembled prompt contexts
- **LRU eviction**: Manage memory efficiently

### Scalability Approaches
- **Sharding by domain**: Separate indexes for different content types
- **Time-based partitioning**: Recent vs historical documents
- **Hot/cold separation**: Frequently accessed vs archival data
- **Read replicas**: Scale query capacity

## AI/ML Specific Considerations

### Real-time RAG
- **Low-latency requirements**: < 500ms end-to-end
- **Streaming responses**: Progressive generation
- **Caching strategies**: Aggressive caching of common queries
- **Edge deployment**: Local vector databases for mobile apps

### Multi-Tenant RAG
- **Isolation**: Tenant-specific indexes or shared with filtering
- **Quota management**: Resource limits per tenant
- **Data governance**: Access control and compliance
- **Cost allocation**: Usage-based billing

### Enterprise RAG
- **Security**: Encryption at rest and in transit
- **Audit logging**: Full traceability of queries and responses
- **Compliance**: GDPR, HIPAA, SOC 2 requirements
- **Governance**: Content moderation and approval workflows

## Implementation Examples

### LangChain RAG Pattern
```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Vector store setup
vectorstore = Chroma(
    collection_name="documents",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Retrieval chain
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5}
)

# Prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Custom RAG with PostgreSQL
```sql
-- Advanced RAG query with re-ranking
WITH retrieved_docs AS (
    SELECT 
        id,
        title,
        content,
        metadata,
        1 - (embedding <=> $1) as similarity,
        ROW_NUMBER() OVER (ORDER BY embedding <=> $1) as rank
    FROM documents
    WHERE 
        published_date >= $2
        AND metadata->>'category' = $3
    ORDER BY embedding <=> $1
    LIMIT 20
),
re_ranked_docs AS (
    SELECT 
        *,
        -- Re-ranking score: similarity + metadata relevance
        similarity * 0.7 + 
        CASE WHEN metadata->>'author' = $4 THEN 0.3 ELSE 0 END as final_score
    FROM retrieved_docs
)
SELECT id, title, content, final_score
FROM re_ranked_docs
ORDER BY final_score DESC
LIMIT 5;
```

## Best Practices

1. **Chunk strategically**: Balance between context completeness and retrieval precision
2. **Tune similarity thresholds**: Avoid irrelevant results while maintaining recall
3. **Implement fallbacks**: Handle cases where retrieval fails
4. **Monitor hallucination**: Track when LLM generates unsupported content
5. **Test with real queries**: Use production-like query patterns
6. **Optimize end-to-end latency**: Focus on the complete pipeline, not just database

## Related Resources

- [Vector Databases] - Deep dive into vector database implementation
- [Index Optimization] - Advanced indexing for RAG workloads
- [AI/ML System Design] - RAG in broader ML system architecture
- [Database Security] - Secure RAG implementations