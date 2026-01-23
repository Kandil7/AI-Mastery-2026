# Production RAG System - Overview

The Production RAG (Retrieval-Augmented Generation) System is an enterprise-grade solution that bridges the gap between Jupyter notebook prototypes and production-ready applications. It implements a hybrid retrieval approach combining dense (semantic) and sparse (keyword) search methods for optimal performance across different query types.

## Key Features

- **Hybrid Retrieval**: Combines dense (semantic) and sparse (keyword) search methods
- **Production Ready**: Built with enterprise-grade features including monitoring, error handling, and scalability
- **Persistent Storage**: Supports both MongoDB for document metadata and vector databases for embeddings
- **Configurable**: Highly customizable through centralized configuration
- **Secure**: Includes input validation, sanitization, and access controls

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

### 3. Data Layer
- **MongoDB**: Stores document metadata and content
  - Document schema validation
  - Index management for performance
  - Bulk operations for efficiency
- **Vector Stores**: For semantic search capabilities
  - ChromaDB: Persistent vector storage
  - FAISS: High-performance similarity search
  - In-memory: Development/testing option

### 4. Ingestion Layer
- **File Processing**: Handles multiple document formats
  - PDF, DOCX, TXT, MD, CSV, HTML support
  - Content extraction and validation
  - Security scanning and sanitization
- **Document Chunking**: Various strategies for optimal segmentation
  - Recursive chunking
  - Semantic chunking
  - Code-specific chunking
  - Markdown-aware chunking

### 5. Evaluation Layer
- **Quality Metrics**: Measures system performance
  - Context Recall: How well retrieval captures relevant information
  - Faithfulness: How factually consistent generation is with retrieved context
  - Answer Relevancy: How relevant the answer is to the question
  - Context Precision: How precise the retrieved context is
- **RAGAS Integration**: Framework for evaluation-as-a-judge

### 6. Observability Layer
- **Monitoring**: Tracks system performance
  - Request latency (p50, p95, p99 percentiles)
  - Request throughput (requests per second)
  - Error rates (by type and severity)
  - Cache hit rates
  - Embedding generation times
- **Logging**: Structured logging with correlation IDs
- **Tracing**: Distributed tracing for multi-component systems

## Technology Stack

- **Python 3.10+**: Core programming language
- **FastAPI**: Web framework with automatic API documentation
- **MongoDB**: Document storage for metadata and content
- **ChromaDB**: Vector database for semantic search
- **Sentence Transformers**: For generating semantic embeddings
- **Hugging Face Transformers**: For text generation
- **Streamlit**: For the interactive dashboard UI
- **Pydantic**: For data validation and settings management
- **NumPy/SciPy**: For numerical computations
- **Scikit-learn**: For machine learning algorithms

## Problem Statement

Most RAG tutorials stop at "chat with your PDF." This system focuses on the **Production** gap:

- How to retrieve accurately when keywords fail (Semantic Search)?
- How to retrieve specific terms like "Schema 1.2" (Keyword Search)?
- How to know if the answer is actually correct (Ragas Evaluation)?
- How to deploy and scale the solution in production environments?

## Solution Approach

The system implements a hybrid retrieval approach that combines the strengths of both semantic and keyword search:

1. **Dense Retrieval**: Uses sentence transformers to create semantic embeddings, excelling at capturing conceptual relationships
2. **Sparse Retrieval**: Uses BM25 algorithm for keyword matching, excelling at finding specific terms and IDs
3. **Fusion Strategies**: Combines results from both approaches using various methods:
   - RRF (Reciprocal Rank Fusion): Robust fusion that works well across different score distributions
   - Weighted: Linear combination with configurable weights
   - CombSUM: Sum of normalized scores
   - CombMNZ: Product of scores and system count

## Data Flow

1. **Ingestion**: Documents are uploaded, processed, and stored in both MongoDB and vector stores
2. **Indexing**: Content is transformed into embeddings and indexed for retrieval
3. **Query Processing**: Incoming queries are classified, expanded, and routed appropriately
4. **Retrieval**: Hybrid search retrieves relevant documents using both semantic and keyword matching
5. **Generation**: Retrieved context is used to generate a response with proper citations
6. **Response**: Results are returned with sources and metadata

## Security Considerations

- Input validation and sanitization to prevent injection attacks
- Secure file handling with type and size validation
- Authentication and authorization for API access
- Encryption in transit for all communications
- Access logging for audit trails