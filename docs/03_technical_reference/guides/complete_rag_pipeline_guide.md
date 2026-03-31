# RAG Engine Mini - Complete RAG Pipeline Guide

## Introduction

The Retrieval-Augmented Generation (RAG) pipeline in RAG Engine Mini is a sophisticated system that combines document retrieval with language model generation to provide accurate, contextually relevant answers to user queries. This guide walks through the complete pipeline from document upload to answer generation.

## Overview of the RAG Process

The RAG pipeline consists of two main phases:

1. **Indexing Phase**: Documents are processed, chunked, embedded, and stored for retrieval
2. **Query Phase**: User questions are processed, relevant documents are retrieved, and answers are generated

## Phase 1: Document Indexing

### Step 1: Document Upload

The process begins when a user uploads a document:

1. **API Request**: User sends a POST request to `/api/v1/documents/upload` with the document file
2. **Validation**: The API validates file type and size against configured limits
3. **Storage**: The file is saved to the configured file storage (local, S3, etc.)
4. **Database Record**: A document record is created in the database with status "created"
5. **Queue Task**: A background indexing task is queued using Celery

### Step 2: Background Processing

The `index_document` Celery task processes the uploaded document:

1. **Status Update**: Document status is updated to "processing"
2. **Text Extraction**: The document is parsed and text is extracted using the appropriate extractor
3. **Multi-Modal Processing**: Images and tables are extracted from documents (Stage 4)
4. **Document Summarization**: An LLM generates a summary of the document for context
5. **Hierarchical Chunking**: The document is split into parent-child chunks using the `chunk_hierarchical` function

### Step 3: Embedding Generation

1. **Text Preparation**: Unique chunks are identified to avoid redundant embedding
2. **Embedding Request**: Text chunks are sent to the embedding service (OpenAI, local, etc.)
3. **Caching**: Embeddings are cached to avoid repeated API calls for identical content
4. **Vector Storage**: Generated vectors are stored in Qdrant with minimal payload (only IDs)

### Step 4: Knowledge Graph Creation

1. **Entity Extraction**: The system extracts entities and relationships from chunks
2. **Triplet Formation**: Subject-Predicate-Object triplets are created
3. **Graph Storage**: Triplets are stored in the graph repository for later retrieval

### Step 5: Finalization

1. **Status Update**: Document status is updated to "indexed"
2. **Metadata Recording**: Chunk counts and other metadata are recorded
3. **Completion Notification**: The system is ready to serve queries for this document

## Phase 2: Query Processing

### Step 1: Query Reception

1. **API Request**: User sends a POST request to `/api/v1/ask` with their question
2. **Authentication**: API key is validated and tenant ID is extracted
3. **Request Parsing**: Query parameters are parsed and validated

### Step 2: Pre-Processing

1. **Privacy Guard**: PII redaction is applied to the query (Stage 5)
2. **Semantic Routing**: Query intent is classified (knowledge vs. chit-chat) (Stage 5)
3. **Query Expansion**: If enabled, related queries are generated using an LLM (Stage 5)

### Step 3: Hybrid Retrieval

The system performs dual-path retrieval:

#### Vector Search Path
1. **Embedding**: The query is converted to a vector using the same embedding model
2. **Similarity Search**: Qdrant performs cosine similarity search against stored vectors
3. **Tenant Filtering**: Results are filtered by tenant ID for isolation
4. **Scoring**: Results are scored based on similarity

#### Keyword Search Path
1. **Full-Text Search**: PostgreSQL performs lexical matching against stored text
2. **Ranking**: Results are ranked using PostgreSQL's built-in text search ranking
3. **Tenant Filtering**: Results are filtered by tenant ID

### Step 4: Result Fusion

1. **RRF Fusion**: Reciprocal Rank Fusion combines results from both search paths
2. **Duplicate Removal**: Identical results from both paths are deduplicated
3. **Limiting**: Results are limited to the configured fusion limit (default 40)

### Step 5: Reranking

1. **Cross-Encoder**: A cross-encoder model reranks the fused results for precision
2. **Top-N Selection**: The top N results (default 8) are selected for generation
3. **Context Hydration**: Full text content is retrieved from the database for the selected chunks

### Step 6: Answer Generation

1. **Prompt Construction**: A RAG prompt is built combining the query and retrieved context
2. **Guardrail Application**: Prompt is checked for safety and compliance
3. **LLM Generation**: The LLM generates an answer based on the context
4. **Token Tracking**: Input and output tokens are counted for cost tracking

### Step 7: Post-Processing

1. **Self-Critique**: The system evaluates the answer's relevance and accuracy
2. **Hallucination Check**: Answers are verified against the provided context
3. **Privacy Restoration**: PII is restored to the answer if it was redacted
4. **Response Formatting**: The answer and sources are formatted for the response

### Step 8: Response Delivery

1. **API Response**: The answer and source information are returned to the client
2. **Metrics Collection**: Performance metrics are recorded (latency, tokens, etc.)
3. **Observability**: The request is traced and logged for monitoring

## Advanced Features

### Stage 2: Enhanced Retrieval

- **Hierarchical Retrieval**: Parent-child relationships between chunks allow for broader context
- **Contextual Retrieval**: Document summaries provide additional context for queries
- **RRF Fusion**: Reciprocal Rank Fusion optimally combines vector and keyword results

### Stage 3: Knowledge Graph Integration

- **Entity Relationships**: Knowledge graphs capture relationships between entities
- **Graph Retrieval**: Graph queries supplement vector and keyword search
- **Reasoning**: Graph traversal enables complex reasoning over document content

### Stage 4: Multi-Modal Capabilities

- **Image Understanding**: Images extracted from documents are described using vision models
- **Table Preservation**: Tabular data maintains structure during processing
- **Visual Context**: Images contribute to the overall document understanding

### Stage 5: Autonomy & Optimization

- **Semantic Routing**: Distinguishes between knowledge queries and chit-chat
- **Privacy Protection**: PII redaction and restoration
- **Self-Critique**: Automatic evaluation of retrieval and generation quality
- **Web Search Fallback**: Falls back to web search when document retrieval is insufficient
- **Optimization**: Continuous improvement through feedback and A/B testing

## Performance Optimizations

### Caching Strategies

- **Embedding Cache**: Frequently used embeddings are cached in Redis
- **Query Result Cache**: Common queries can be cached for faster response
- **LLM Response Cache**: Deterministic queries can be cached

### Batch Processing

- **Bulk Operations**: Multiple documents can be processed in batches
- **Vector Upserts**: Multiple vectors are inserted in single operations
- **Parallel Processing**: Independent operations are executed in parallel

### Resource Management

- **Connection Pooling**: Database and vector store connections are pooled
- **Memory Management**: Large documents are processed in chunks to manage memory
- **Timeout Handling**: Long-running operations have configurable timeouts

## Security & Isolation

### Multi-Tenancy

- **Tenant IDs**: All operations include tenant identification
- **Data Isolation**: Results are filtered by tenant ID at every layer
- **Resource Separation**: Each tenant gets isolated storage and compute resources

### Authentication

- **API Keys**: Secure API key authentication for all endpoints
- **Role-Based Access**: Different access levels for different user types
- **Audit Logging**: All operations are logged for security review

## Error Handling & Resilience

### Fault Tolerance

- **Retry Mechanisms**: Failed operations are retried with exponential backoff
- **Fallback Strategies**: Alternative approaches when primary methods fail
- **Graceful Degradation**: System continues operating with reduced functionality

### Monitoring

- **Health Checks**: Regular system health verification
- **Performance Metrics**: Latency, throughput, and error rate tracking
- **Alerting**: Automated notifications for system issues

## Scalability Considerations

### Horizontal Scaling

- **Stateless API Servers**: Multiple API instances can serve requests
- **Distributed Workers**: Celery workers can run on multiple machines
- **Database Sharding**: Tenant data can be distributed across multiple database instances

### Load Distribution

- **Task Queues**: Background tasks are distributed across worker pools
- **Load Balancing**: Requests are distributed across API instances
- **Resource Allocation**: Compute resources are allocated based on demand

The RAG pipeline in RAG Engine Mini represents a production-ready, scalable solution for document-based question answering, incorporating advanced techniques for retrieval, generation, and optimization while maintaining security and performance standards.