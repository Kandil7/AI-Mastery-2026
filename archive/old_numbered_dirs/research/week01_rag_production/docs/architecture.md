# Architecture

The Production RAG System follows a layered architecture with clear separation of concerns, designed to support enterprise-grade applications with high availability, scalability, and maintainability.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   UI Layer      │    │   API Layer      │    │  Data Layer     │
│                 │    │                  │    │                 │
│  Streamlit UI   │◄──►│   FastAPI        │◄──►│  MongoDB        │
│                 │    │                  │    │                 │
│  Query Form     │    │  RAG Pipeline    │    │  Vector Store   │
│  Dashboard      │    │  Query Router    │    │  (ChromaDB)     │
└─────────────────┘    │  Ingestion       │    │                 │
                       │  Services        │    └─────────────────┘
                       └──────────────────┘
```

## Architecture Components

### 1. Presentation Layer

#### Streamlit UI (`ui.py`)
- **Purpose**: Interactive dashboard for system interaction
- **Features**:
  - Query interface with adjustable parameters
  - Document management tools
  - Performance metrics visualization
  - System health monitoring
  - Debugging tools

### 2. API Layer (FastAPI)

#### Core API (`api.py`)
- **Purpose**: Exposes RESTful endpoints for system interaction
- **Endpoints**:
  - `GET /`: Health check endpoint
  - `POST /index`: Add documents to the knowledge base
  - `POST /query`: Query the RAG system
  - `GET /health`: Health check with detailed status
  - `GET /metrics`: Prometheus-compatible metrics endpoint
  - `POST /upload`: Upload documents for indexing
  - `GET /documents`: List stored documents
  - `GET /documents/{doc_id}`: Get specific document

- **Features**:
  - Input validation using Pydantic models
  - Comprehensive error handling with appropriate HTTP status codes
  - Request/response logging and tracing
  - Rate limiting and resource protection
  - Health checks and readiness probes
  - Detailed API documentation with OpenAPI/Swagger
  - Performance monitoring and metrics collection

### 3. Business Logic Layer

#### RAG Pipeline (`src/pipeline.py`)
- **Purpose**: Orchestrates the complete RAG workflow
- **Components**:
  - `RAGConfig`: Centralized configuration management
  - `RAGPipeline`: Main orchestrator for retrieval and generation

- **Responsibilities**:
  - Manages retrieval and generation processes
  - Handles query processing and response formatting
  - Maintains configuration and state
  - Provides structured responses with sources and metadata

#### Hybrid Retriever (`src/retrieval/retrieval.py`)
- **Purpose**: Combines dense and sparse retrieval methods
- **Components**:
  - `DenseRetriever`: Semantic search using sentence transformers
  - `SparseRetriever`: Keyword-based search using BM25 algorithm
  - `HybridRetriever`: Combines both approaches with configurable fusion strategies

- **Fusion Strategies**:
  - RRF (Reciprocal Rank Fusion): Robust fusion that works well across different score distributions
  - Weighted: Linear combination with configurable weights
  - Densité: Density-based fusion considering score distribution spread
  - CombSUM: Sum of normalized scores
  - CombMNZ: Product of normalized scores and system count

#### Query Processing (`src/retrieval/query_processing.py`)
- **Purpose**: Advanced query understanding and processing
- **Components**:
  - `QueryClassifier`: Classifies query types and extracts information
  - `QueryExpander`: Expands queries with synonyms and related terms
  - `ResponseGenerator`: Generates responses based on context
  - `CitationExtractor`: Extracts citations from responses
  - `RAGQueryProcessor`: Main query processing orchestrator
  - `QueryRouter`: Routes queries to appropriate processors

### 4. Ingestion Layer

#### Document Ingestion (`src/ingestion/__init__.py`)
- **Purpose**: Handles document processing from various formats
- **Components**:
  - `IngestionRequest`: Request model for ingestion operations
  - `IngestionResult`: Result model for ingestion operations
  - `IngestionPipeline`: Main ingestion orchestrator
  - `ContentValidator`: Validates document content quality and security
  - `MetadataEnricher`: Automatically enriches document metadata
  - `DocumentChunker`: Splits documents into smaller chunks

#### File Processing (`src/ingestion/file_processor.py`)
- **Purpose**: Handles file uploads and content extraction
- **Components**:
  - `FileManager`: Manages file uploads and processing
  - `FileType`: Enumeration for supported file types
  - `FileUploadRequest`: Request model for file uploads
  - `FileProcessingResult`: Result model for file processing

#### MongoDB Integration (`src/ingestion/mongo_storage.py`)
- **Purpose**: Persistent storage for documents and metadata
- **Components**:
  - `MongoDocument`: Pydantic model for MongoDB documents
  - `MongoConnectionManager`: Manages MongoDB connections
  - `DocumentCollection`: Handles document operations
  - `MongoStorage`: Main MongoDB storage class

### 5. Data Layer

#### Vector Storage (`src/retrieval/vector_store.py`)
- **Purpose**: Manages vector embeddings for semantic search
- **Components**:
  - `VectorConfig`: Configuration for vector storage
  - `VectorRecord`: Represents a vector embedding with metadata
  - `BaseVectorStore`: Abstract base class for vector storage implementations
  - `ChromaVectorStore`: ChromaDB implementation
  - `FAISSVectorStore`: FAISS implementation
  - `InMemoryVectorStore`: Development/testing implementation
  - `VectorStoreFactory`: Factory for creating vector store implementations
  - `VectorManager`: Main vector storage orchestrator

#### MongoDB Storage (`src/ingestion/mongo_storage.py`)
- **Purpose**: Persistent storage for document metadata and content
- **Features**:
  - Asynchronous MongoDB operations
  - Connection pooling and management
  - Document schema validation
  - Index management for performance
  - Bulk operations for efficiency
  - Change streams for real-time updates

### 6. Evaluation Layer

#### RAG Evaluation (`src/eval/__init__.py`)
- **Purpose**: Comprehensive evaluation capabilities for RAG systems
- **Components**:
  - `RAGEvaluator`: Main evaluation orchestrator
  - Integration with RAGAS framework
  - Custom evaluation metrics

- **Metrics Measured**:
  - Context Recall: How well retrieval captures relevant information
  - Faithfulness: How factually consistent generation is with retrieved context
  - Answer Relevancy: How relevant the answer is to the question
  - Context Precision: How precise the retrieved context is
  - Context Relevancy: How relevant the retrieved context is to the question

### 7. Configuration Layer

#### Configuration Management (`src/config.py`)
- **Purpose**: Centralized configuration management using Pydantic settings
- **Components**:
  - `RAGConfig`: Main configuration class
  - `DatabaseConfig`: Database configuration
  - `ModelConfig`: Model configuration
  - `RetrievalConfig`: Retrieval configuration
  - `APIConfig`: API configuration
  - `LoggingConfig`: Logging configuration
  - `SecurityConfig`: Security configuration

### 8. Observability Layer

#### Monitoring and Logging (`src/observability/__init__.py`)
- **Purpose**: Comprehensive observability capabilities
- **Components**:
  - `Logger`: Structured logging with correlation IDs
  - `MetricsCollector`: Performance metrics collection
  - `Tracer`: Distributed tracing for multi-component systems
  - `ObservabilityManager`: Main observability orchestrator

- **Metrics Tracked**:
  - Request latency (p50, p95, p99 percentiles)
  - Request throughput (requests per second)
  - Error rates (by type and severity)
  - Cache hit rates
  - Embedding generation times
  - Retrieval performance
  - Quality metrics over time

### 9. Service Layer

#### Indexing Service (`src/services/indexing.py`)
- **Purpose**: Business logic for indexing operations
- **Function**:
  - `index_documents`: Coordinates indexing to both MongoDB and vector stores

## Data Flow Architecture

### Ingestion Flow
```
File Upload → File Validation → Content Extraction → Document Processing → Chunking → Validation → Indexing → Storage (MongoDB + Vector Store)
```

### Query Flow
```
Query Input → Query Classification → Query Expansion → Hybrid Retrieval → Response Generation → Citation Extraction → Response Formatting → Output
```

## Security Architecture

### Input Validation
- All user inputs are validated and sanitized
- Content filtering prevents malicious content
- File upload restrictions prevent dangerous file types

### Authentication and Authorization
- API key support for access control
- Role-based access controls for sensitive operations
- Rate limiting to prevent abuse

### Data Protection
- Encryption in transit for all communications
- Secure storage of sensitive configuration
- Access logging for audit trails

## Scalability Architecture

### Horizontal Scaling
- Load balancer for multiple API instances
- Shared vector database cluster
- Distributed MongoDB setup

### Vertical Scaling
- Larger instance types for increased resources
- GPU acceleration for faster processing
- More memory for larger models

### Auto-scaling
- Container orchestration platforms (Kubernetes)
- Cloud platform auto-scaling groups
- Resource-based scaling triggers