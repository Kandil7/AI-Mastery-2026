# Production RAG System Architecture

## Overview

This document describes the production architecture for the mini-RAG system, designed to bridge the gap between Jupyter notebook prototypes and enterprise-grade applications. The system implements a hybrid retrieval approach combining dense (semantic) and sparse (keyword) search methods for optimal performance across different query types.

## Architecture Components

### 1. API Layer (FastAPI)
- **Purpose**: Exposes RESTful endpoints for document indexing and querying
- **Components**:
  - Health check endpoints
  - Document indexing endpoints
  - Query processing endpoints
  - Metrics and monitoring endpoints
- **Features**:
  - Input validation using Pydantic models
  - Comprehensive error handling
  - Request/response logging
  - Rate limiting and resource protection

### 2. Business Logic Layer
- **RAG Pipeline**: Orchestrates the complete RAG workflow
  - Manages retrieval and generation processes
  - Handles query processing and response formatting
  - Maintains configuration and state
- **Hybrid Retriever**: Combines dense and sparse retrieval
  - DenseRetriever: Uses sentence transformers for semantic search
  - SparseRetriever: Uses BM25 algorithm for keyword matching
  - Fusion strategies: RRF, weighted, densite, combsum, combmnz

### 3. Data Access Layer
- **Document Storage**: Persistent storage for documents
  - MongoDB for document metadata and content
  - ChromaDB for vector embeddings
- **Index Management**: Handles indexing and retrieval operations
  - Vector index maintenance
  - Document metadata management

## Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  API Handler   │───▶│  RAG Pipeline   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐                                     │
│  Document       │◀────────────────────────────────────┘
│  Storage        │
│  (MongoDB)      │
└─────────────────┘
        │
┌─────────────────┐
│  Vector Store   │
│  (ChromaDB)     │
└─────────────────┘
```

## Key Classes and Functions

### Document Class
Represents a document in the knowledge base with content and metadata.

**Attributes**:
- `id`: Unique identifier for the document
- `content`: The actual text content of the document
- `metadata`: Additional information about the document
- `source`: Origin of the document
- `doc_type`: Type of document
- `embedding_vector`: Precomputed embedding vector if available

**Methods**:
- `validate_access()`: Validates user access based on access controls
- `get_content_length()`: Returns the length of document content
- `get_metadata_summary()`: Returns a summary of key metadata fields

### DenseRetriever Class
Implements dense retrieval using sentence embeddings and vector similarity.

**Features**:
- Sentence transformer-based embeddings
- Persistent storage with ChromaDB
- In-memory fallback option
- Cosine similarity scoring
- Batch processing for improved performance

**Methods**:
- `add_documents()`: Adds documents to the index
- `retrieve()`: Retrieves top-k most relevant documents
- `clear_index()`: Clears all indexed documents
- `get_document_count()`: Gets the number of indexed documents

### SparseRetriever Class
Implements sparse retrieval using BM25 algorithm for advanced keyword matching.

**Features**:
- BM25 algorithm for improved keyword matching
- Configurable parameters (k1, b) for tuning
- Efficient sparse matrix operations
- Term frequency normalization
- Document length normalization

**Methods**:
- `add_documents()`: Adds documents to the index using BM25
- `retrieve()`: Retrieves top-k most relevant documents using BM25
- `_compute_idf()`: Computes Inverse Document Frequency
- `_score_bm25()`: Calculates BM25 score for query-document pairs

### HybridRetriever Class
Combines dense and sparse retrieval using configurable fusion strategies.

**Features**:
- Dual retrieval pipeline (dense + sparse)
- Multiple configurable fusion strategies
- Automatic score normalization
- Flexible weighting between approaches
- Performance optimization for large result sets

**Fusion Strategies**:
- RRF (Reciprocal Rank Fusion): Robust fusion that works well across different score distributions
- Weighted: Linear combination of normalized scores with configurable weights
- Densité: Density-based fusion that considers the distribution of scores
- CombSUM: Sum of normalized scores from both systems
- CombMNZ: Product of normalized scores and count of systems that retrieved the document

**Methods**:
- `index()`: Indexes documents in both dense and sparse retrievers
- `retrieve()`: Retrieves top-k results using hybrid approach
- `_rrf_fusion()`: Applies Reciprocal Rank Fusion
- `_weighted_fusion()`: Applies weighted linear fusion
- `_densite_fusion()`: Applies density-based fusion
- `_combsum_fusion()`: Applies CombSUM fusion
- `_combmnz_fusion()`: Applies CombMNZ fusion

### RAGPipeline Class
Main RAG pipeline orchestrating retrieval and generation processes.

**Features**:
- Configurable components and parameters
- Structured output format for UI and evaluation
- Error handling and fallback mechanisms
- Performance optimization considerations

**Methods**:
- `__init__()`: Initializes the pipeline with configuration
- `index()`: Indexes documents in the retrieval system
- `retrieve()`: Retrieves relevant documents for a query
- `generate()`: Generates a response based on query and contexts
- `query()`: Executes a complete RAG query from retrieval to generation

### RAGConfig Class
Configuration class for RAG pipeline parameters.

**Attributes**:
- `top_k`: Number of top documents to retrieve
- `max_new_tokens`: Maximum number of tokens to generate
- `generator_model`: Name of the language model for generation
- `dense_model`: Name of the sentence transformer model
- `alpha`: Weight for dense retrieval in hybrid fusion
- `fusion`: Fusion strategy for hybrid retrieval

## API Endpoints

### GET /
Basic health check endpoint returning system status.

### GET /health
Comprehensive health check with detailed status information.

### POST /index
Add documents to the knowledge base with validation.

### POST /query
Query the RAG system to get an answer with supporting evidence.

### GET /metrics
Prometheus-compatible metrics endpoint.

## Production Considerations

### Scalability
- Horizontal scaling through load balancing
- Database connection pooling
- Caching strategies for frequent queries
- Asynchronous processing for heavy operations

### Reliability
- Circuit breaker patterns for external services
- Graceful degradation when components fail
- Comprehensive error handling and logging
- Health checks and monitoring

### Security
- Input validation to prevent injection attacks
- Authentication and authorization for API access
- Secure handling of sensitive data
- Rate limiting to prevent abuse

### Observability
- Structured logging with correlation IDs
- Distributed tracing for request flows
- Performance metrics collection
- Alerting for system anomalies

## Deployment

### Containerization
- Docker containers for consistent deployment
- Multi-stage builds for optimized images
- Environment-specific configurations

### Orchestration
- Kubernetes for container orchestration
- Service discovery and load balancing
- Auto-scaling based on demand
- Rolling updates with zero downtime

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock external dependencies
- Edge case validation

### Integration Tests
- End-to-end workflow testing
- API contract validation
- Database interaction testing

### Performance Tests
- Load testing with realistic scenarios
- Latency and throughput measurements
- Resource utilization monitoring

### Chaos Engineering
- Failure injection testing
- Resilience validation
- Recovery procedure verification