# Source Code Documentation

This section provides detailed documentation for the source code modules of the Production RAG System, including class definitions, method signatures, and implementation details.

## 1. Configuration Module (`src/config.py`)

This module implements a comprehensive configuration management system using Pydantic BaseSettings. It provides centralized configuration for all components of the RAG system including database connections, API keys, model parameters, and operational settings.

### `Environment` (Enum)
Enumeration for application environments:
- `DEVELOPMENT`
- `TESTING`
- `STAGING`
- `PRODUCTION`

### `DatabaseConfig` (BaseModel)
Configuration for database connections with the following attributes:
- `url`: Database connection URL (default: "mongodb://localhost:27017")
- `name`: Database name (default: "minirag")
- `username`: Database username (optional)
- `password`: Database password (optional, marked as secret)
- `pool_size`: Connection pool size (default: 10, range: 1-100)
- `max_overflow`: Maximum overflow connections (default: 20, range: 0-100)
- `echo`: Enable SQL query logging (default: False)

The class includes a validator `validate_pool_sizes` that ensures pool sizes are non-negative.

### `ModelConfig` (BaseModel)
Configuration for ML models and embeddings with the following attributes:
- `generator_model`: Name of the text generation model (default: "gpt2")
- `dense_model`: Name of the dense embedding model (default: "all-MiniLM-L6-v2")
- `sparse_model`: Name of the sparse embedding model (default: "bm25")
- `max_new_tokens`: Maximum tokens for generation (default: 300, range: 1-2048)
- `temperature`: Temperature for generation diversity (default: 0.7, range: 0.0-2.0)
- `top_p`: Top-p sampling parameter (default: 0.9, range: 0.0-1.0)
- `top_k`: Top-k sampling parameter (default: 5, range: 1-20)

Includes validators `validate_probability_params` that ensure probability-based parameters are between 0 and 1.

### `RetrievalConfig` (BaseModel)
Configuration for retrieval components with the following attributes:
- `alpha`: Weight for dense retrieval in hybrid fusion (default: 0.5, range: 0.0-1.0)
- `fusion_method`: Fusion strategy ('rrf', 'weighted', 'densite', 'combsum', 'combmnz') (default: "rrf")
- `rrf_k`: Smoothing constant for RRF calculation (default: 60)
- `sparse_k1`: BM25 k1 parameter (default: 1.5)
- `sparse_b`: BM25 b parameter (default: 0.75)
- `max_candidates`: Maximum candidates for reranking (default: 50)

Includes validator `validate_alpha` that ensures alpha is between 0 and 1.

### `APIConfig` (BaseModel)
Configuration for API and networking with the following attributes:
- `host`: Host address for the API server (default: "0.0.0.0")
- `port`: Port number for the API server (default: 8000, range: 1-65535)
- `cors_origins`: Allowed origins for CORS (default: ["*"])
- `rate_limit_requests`: Max requests per minute (default: 100)
- `rate_limit_window`: Time window for rate limiting in seconds (default: 60)
- `request_timeout`: Request timeout in seconds (default: 30)

### `LoggingConfig` (BaseModel)
Configuration for logging and monitoring with the following attributes:
- `level`: Logging level (default: "INFO")
- `format`: Log format string (default: "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
- `file_path`: Path to log file (optional)
- `max_bytes`: Maximum log file size in bytes (default: 10485760)
- `backup_count`: Number of backup log files (default: 5)

### `SecurityConfig` (BaseModel)
Configuration for security settings with the following attributes:
- `secret_key`: Secret key for cryptographic operations (default: "secret", marked as secret)
- `jwt_algorithm`: Algorithm for JWT encoding (default: "HS256")
- `access_token_expire_minutes`: Access token expiration in minutes (default: 30)
- `enable_authentication`: Enable authentication middleware (default: False)
- `allowed_hosts`: Allowed hosts for security headers (default: ["localhost", "127.0.0.1"])

### `RAGConfig` (BaseSettings)
Main configuration class that aggregates all configuration sections and provides the primary interface for accessing configuration values throughout the application.

#### Attributes:
- `app_name`: Name of the application (default: "Production RAG API")
- `app_version`: Version of the application (default: "1.0.0")
- `environment`: Application environment (default: Environment.DEVELOPMENT)
- `debug`: Enable debug mode (default: False)
- `database`: Database configuration (default factory: DatabaseConfig)
- `models`: Model configuration (default factory: ModelConfig)
- `retrieval`: Retrieval configuration (default factory: RetrievalConfig)
- `api`: API configuration (default factory: APIConfig)
- `logging`: Logging configuration (default factory: LoggingConfig)
- `security`: Security configuration (default factory: SecurityConfig)
- `openai_api_key`: OpenAI API key (optional, marked as secret)
- `huggingface_token`: Hugging Face token (optional, marked as secret)

Includes:
- Pydantic configuration with env_file, env_file_encoding, case_sensitive, and env_nested_delimiter
- Validator `validate_environment` for environment value validation
- Methods `is_development()` and `is_production()` for environment checks
- Method `get_database_url()` to get database URL with credentials if available

Global instance `settings` is created and exported for use in other modules.

## 2. RAG Pipeline (`src/pipeline.py`)

This module implements a comprehensive RAG (Retrieval-Augmented Generation) pipeline that combines hybrid retrieval with language model generation to produce accurate, well-sourced responses to user queries. The pipeline is designed for production environments with focus on reliability, performance, and traceability.

### `RAGConfig` (dataclass)
Configuration class for RAG pipeline parameters that centralizes all configurable parameters for the RAG pipeline, allowing for easy experimentation and tuning of different components.

#### Attributes:
- `top_k` (int): Number of top documents to retrieve for each query (default: 3)
- `max_new_tokens` (int): Maximum number of tokens to generate in response (default: 200)
- `generator_model` (str): Name of the language model to use for generation (default: "gpt2")
- `dense_model` (str): Name of the sentence transformer model for dense retrieval (default: "all-MiniLM-L6-v2")
- `alpha` (float): Weight for dense retrieval in hybrid fusion (sparse weight = 1 - alpha) (default: 0.5)
- `fusion` (str): Fusion strategy for hybrid retrieval ('rrf' or 'weighted') (default: "rrf")

### `RAGPipeline` (class)
Main RAG pipeline orchestrator that serves as the central orchestrator for the RAG system, managing the complete flow from query input to response generation with cited sources.

#### Constructor:
- `config` (RAGConfig, optional): Configuration object with pipeline parameters

#### Attributes:
- `config`: Configuration object
- `retriever`: HybridRetriever instance
- `generator`: Hugging Face pipeline for text generation

#### Methods:
- `__init__(config)`: Initialize the RAG pipeline with specified configuration
- `index(documents)`: Index documents in the retrieval system
- `retrieve(query, top_k)`: Retrieve relevant documents for the given query
- `generate(query, contexts)`: Generate a response based on the query and retrieved contexts
- `query(query, top_k)`: Execute a complete RAG query from retrieval to generation

## 3. Retrieval System (`src/retrieval/retrieval.py`)

This module implements a sophisticated hybrid retrieval system that combines dense and sparse retrieval techniques to achieve optimal performance across different query types. The system leverages both semantic understanding and keyword matching to provide robust information retrieval capabilities suitable for production environments.

### `Document` (dataclass)
Represents a document in the knowledge base with content and metadata.

#### Attributes:
- `id` (str): Unique identifier for the document
- `content` (str): The actual text content of the document
- `metadata` (Dict[str, str]): Additional information about the document
- `source` (str): Origin of the document (e.g., 'pdf', 'web', 'database')
- `doc_type` (str): Type of document (e.g., 'report', 'manual', 'transcript')
- `created_at` (str): Timestamp when document was created/ingested
- `updated_at` (str): Timestamp when document was last updated
- `access_control` (Dict[str, str]): Access permissions and restrictions
- `page_number` (Optional[int]): Page number if extracted from multi-page document
- `section_title` (Optional[str]): Section title if extracted from structured document
- `embedding_vector` (Optional[np.ndarray]): Precomputed embedding vector if available
- `checksum` (Optional[str]): SHA256 hash of content for integrity verification

Includes validation in `__post_init__` method and methods for access validation and content analysis.

### `RetrievalResult` (dataclass)
Represents a single result from the retrieval process, encapsulating the relationship between a retrieved document and its relevance score/rank in the context of a specific query.

#### Attributes:
- `document` (Document): The retrieved document object
- `score` (float): Normalized relevance score (higher is better)
- `rank` (int): Position in the ranked results list (1-indexed)

### `DenseRetriever` (class)
Implements dense retrieval using sentence embeddings and vector similarity. Dense retrieval leverages neural network-based encoders to create semantic representations of documents and queries. This approach excels at capturing semantic relationships and understanding meaning beyond exact keyword matches.

#### Constructor:
- `dense_model` (str): Name of the sentence transformer model to use (default: "all-MiniLM-L6-v2")
- `collection_name` (str): Name of the ChromaDB collection (default: "week01_rag")
- `persist_directory` (str): Path for persistent storage (default: "data/chroma")
- `use_chroma` (bool): Whether to use ChromaDB for persistence (default: True)
- `batch_size` (int): Size of batches for encoding (for memory efficiency) (default: 32)

#### Attributes:
- `encoder`: SentenceTransformer instance for encoding
- `documents`: List of indexed documents
- `use_chroma`: Flag indicating if ChromaDB is used
- `collection_name`: ChromaDB collection name
- `persist_directory`: Directory for persistent storage
- `batch_size`: Size of batches for encoding
- `_chroma_collection`: ChromaDB collection instance
- `_embeddings`: Numpy array of document embeddings

#### Methods:
- `add_documents(documents)`: Add documents to the dense retriever index
- `retrieve(query, top_k)`: Retrieve top-k most relevant documents for the given query
- `clear_index()`: Clear all indexed documents and embeddings
- `get_document_count()`: Get the number of documents currently indexed

### `SparseRetriever` (class)
Implements sparse retrieval using BM25 algorithm for advanced keyword matching. Sparse retrieval focuses on keyword matching and exact term correspondence. This approach excels at finding documents containing specific terms like IDs, codes, or technical terminology that might be missed by dense methods.

#### Constructor:
- `k1` (float): BM25 parameter controlling term frequency saturation (default: 1.5)
- `b` (float): BM25 parameter controlling document length normalization (default: 0.75)
- `max_features` (int): Maximum number of features in vocabulary (default: 10000)

#### Attributes:
- `k1`: Term frequency saturation parameter
- `b`: Document length normalization parameter
- `max_features`: Maximum number of features in vocabulary
- `documents`: List of indexed documents
- `tokenizer`: Function to tokenize text
- `avg_doc_len`: Average document length
- `idf`: Inverse document frequency dictionary
- `doc_freqs`: List of token frequencies per document
- `doc_lens`: Length of each document

#### Methods:
- `add_documents(documents)`: Add documents to the sparse retriever index using BM25 algorithm
- `_compute_idf()`: Compute Inverse Document Frequency (IDF) for all terms in the corpus
- `_score_bm25(query_tokens, doc_idx)`: Calculate BM25 score for a query and a specific document
- `retrieve(query, top_k)`: Retrieve top-k most relevant documents for the given query using BM25

### `HybridRetriever` (class)
Combines dense and sparse retrieval using configurable fusion strategies. Hybrid retrieval addresses the limitations of individual approaches by leveraging the strengths of both dense (semantic understanding) and sparse (keyword precision) methods. The fusion strategy determines how results from both retrievers are combined into a final ranked list.

#### Constructor:
- `alpha` (float): Weight for dense retrieval (sparse weight = 1 - alpha) (default: 0.5)
- `fusion` (str): Fusion strategy ('rrf', 'weighted', 'densite', 'combsum', 'combmnz') (default: "rrf")
- `dense_model` (str): Sentence transformer model name (default: "all-MiniLM-L6-v2")
- `sparse_k1` (float): BM25 k1 parameter for sparse retrieval (default: 1.5)
- `sparse_b` (float): BM25 b parameter for sparse retrieval (default: 0.75)
- `rrf_k` (int): Smoothing constant for RRF calculation (default: 60)

#### Attributes:
- `alpha`: Weight for dense retrieval
- `fusion`: Fusion strategy
- `rrf_k`: Smoothing constant for RRF calculation
- `dense_retriever`: DenseRetriever instance
- `sparse_retriever`: SparseRetriever instance

#### Methods:
- `index(documents)`: Index documents in both dense and sparse retrievers
- `retrieve(query, top_k)`: Retrieve top-k results using hybrid approach
- `_rrf_fusion(dense_results, sparse_results, top_k)`: Apply Reciprocal Rank Fusion to combine results
- `_weighted_fusion(dense_results, sparse_results, top_k)`: Apply weighted linear fusion
- `_densite_fusion(dense_results, sparse_results, top_k)`: Apply density-based fusion
- `_combsum_fusion(dense_results, sparse_results, top_k)`: Apply CombSUM fusion
- `_combmnz_fusion(dense_results, sparse_results, top_k)`: Apply CombMNZ fusion
- `get_fusion_strategies()`: Get a list of available fusion strategies

## 4. Vector Storage (`src/retrieval/vector_store.py`)

This module implements a comprehensive vector storage and retrieval system for the RAG system. It manages vector embeddings for documents, provides efficient similarity search capabilities, and integrates with various vector databases for scalable storage and retrieval.

### `VectorDBType` (Enum)
Enumeration for supported vector database types:
- `CHROMA`
- `FAISS`
- `IN_MEMORY`

### `VectorConfig` (BaseModel)
Configuration for vector storage and retrieval with the following attributes:
- `db_type` (VectorDBType): Type of vector database to use (default: VectorDBType.IN_MEMORY)
- `collection_name` (str): Name of the collection (default: "rag_vectors")
- `persist_directory` (str): Directory for persistent storage (default: "./data/vector_store")
- `dimension` (int): Dimension of the vectors (default: 384, range: 1+)
- `metric` (str): Distance metric for similarity search (default: "cosine")
- `batch_size` (int): Batch size for vector operations (default: 32, range: 1-1024)
- `ef_construction` (int): HNSW construction parameter (default: 200)
- `ef_search` (int): HNSW search parameter (default: 50)
- `m` (int): HNSW M parameter (default: 16)

### `VectorRecord` (BaseModel)
Model for vector records in the storage system with the following attributes:
- `id` (str): Unique identifier for the vector record
- `vector` (List[float]): The vector embedding
- `metadata` (Dict[str, Any]): Associated metadata
- `document_id` (str): Reference to the original document
- `text_content` (Optional[str]): Original text content (optional)

Includes validation in `__init__` method to ensure vector dimension consistency.

### `BaseVectorStore` (ABC)
Abstract base class for vector storage implementations with abstract methods:
- `initialize()`: Initialize the vector store
- `add_vectors(vectors)`: Add vectors to the store
- `search(query_vector, k)`: Search for similar vectors
- `get_vector(vector_id)`: Get a vector by ID
- `delete_vector(vector_id)`: Delete a vector by ID
- `update_vector(vector_record)`: Update an existing vector
- `get_count()`: Get the total number of vectors in the store
- `close()`: Close the vector store and release resources

### `InMemoryVectorStore` (BaseVectorStore)
In-memory vector store implementation for development and testing.

### `ChromaVectorStore` (BaseVectorStore)
ChromaDB vector store implementation with persistent storage capabilities.

### `FAISSVectorStore` (BaseVectorStore)
FAISS vector store implementation for high-performance similarity search.

### `VectorStoreFactory` (class)
Factory for creating appropriate vector store implementations with static method `create_vector_store(config)`.

### `VectorManager` (class)
Manager class for handling vector operations in the RAG system.

## 5. Ingestion Pipeline (`src/ingestion/__init__.py`)

This module implements a comprehensive document ingestion pipeline that handles the complete flow from file upload to document indexing in the RAG system. It includes validation, processing, transformation, and storage of documents with appropriate metadata and error handling.

### `IngestionRequest` (BaseModel)
Request model for document ingestion operations with the following attributes:
- `source_type` (str): Type of source ('file_upload', 'api_import', 'database', etc.) (default: "file_upload")
- `metadata` (Dict[str, Any]): Additional metadata to attach to documents (default: {})
- `chunk_size` (int): Size of text chunks (default: 1000, range: 100-10000)
- `chunk_overlap` (int): Overlap between chunks (default: 200, range: 0-1000)
- `validate_content` (bool): Whether to validate content quality (default: True)
- `enrich_metadata` (bool): Whether to enrich metadata automatically (default: True)

### `IngestionResult` (BaseModel)
Result model for ingestion operations with the following attributes:
- `success` (bool): Whether the operation was successful
- `message` (str): Human-readable message about the result
- `processed_documents` (int): Number of documents processed
- `indexed_documents` (int): Number of documents successfully indexed
- `processing_time_ms` (float): Time taken for processing in milliseconds
- `errors` (List[str]): List of errors encountered (default: [])
- `warnings` (List[str]): List of warnings during processing (default: [])
- `metadata` (Dict[Any]): Additional metadata about the operation (default: {})

### `ContentValidator` (class)
Validator for document content quality and security with methods:
- `validate_content(content)`: Validate document content for quality and security issues
- `sanitize_content(content)`: Sanitize document content to remove potentially harmful elements

### `MetadataEnricher` (class)
Enricher for document metadata with method:
- `enrich_metadata(document)`: Enrich document metadata with automatically derived information

### `DocumentChunker` (class)
Chunker for splitting documents into smaller pieces with method:
- `chunk_document(document)`: Split a document into smaller chunks

### `IngestionPipeline` (class)
Main ingestion pipeline orchestrator with methods:
- `ingest_from_file(file, ingestion_request)`: Ingest documents from an uploaded file
- `ingest_from_text(text, ingestion_request, doc_id, title)`: Ingest documents from raw text content
- `ingest_batch(documents, ingestion_request)`: Ingest a batch of documents

Global instances `ingestion_pipeline` and functions `initialize_ingestion_pipeline` are provided.

## 6. File Processing (`src/ingestion/file_processor.py`)

This module implements comprehensive file upload and processing functionality for the RAG system. It handles various document formats, performs content extraction, validation, and preprocessing before indexing in the RAG pipeline.

### `FileType` (Enum)
Enumeration for supported file types:
- `PDF`
- `TXT`
- `DOCX`
- `MD`
- `CSV`
- `HTML`

### `FileUploadRequest` (BaseModel)
Request model for file uploads with the following attributes:
- `filename` (str): Name of the uploaded file
- `content_type` (str): MIME type of the uploaded file
- `file_size` (int): Size of the uploaded file in bytes
- `metadata` (Dict[str, Any]): Additional metadata associated with the file

### `FileProcessingResult` (BaseModel)
Result model for file processing operations with the following attributes:
- `success` (bool): Whether the processing was successful
- `message` (str): Human-readable message about the result
- `documents` (List[Document]): List of extracted documents
- `processing_time_ms` (float): Time taken for processing in milliseconds
- `errors` (List[str]): List of errors encountered during processing
- `warnings` (List[str]): List of warnings during processing
- `metadata` (Dict[Any]): Additional metadata about the processing

### `FileManager` (class)
Main file manager class that handles file uploads, validation, and processing with methods:
- `get_file_type(filename)`: Determine the file type based on extension
- `validate_file_upload(request)`: Validate file upload request
- `save_uploaded_file(content, original_filename)`: Save uploaded file to temporary location
- `process_file(filepath, filename, metadata)`: Process a file and extract documents
- `cleanup_temp_file(filepath)`: Clean up temporary file
- `cleanup_old_temp_files(max_age_hours)`: Clean up old temporary files

## 7. MongoDB Integration (`src/ingestion/mongo_storage.py`)

This module implements MongoDB integration for the RAG system, providing persistent storage for documents, metadata, and related information. It includes proper connection management, data modeling, and CRUD operations for document management.

### `MongoDocument` (BaseModel)
Pydantic model for MongoDB documents with validation and the following attributes:
- `id` (Optional[PyObjectId]): MongoDB document ID
- `rag_document_id` (str): Original RAG document ID
- `content` (str): Document content
- `source` (str): Document source
- `doc_type` (str): Document type
- `metadata` (Dict[Any]): Document metadata
- `content_hash` (str): Hash of content for duplicate detection
- `created_at` (datetime): Creation timestamp
- `updated_at` (datetime): Last update timestamp
- `embedding_vector` (Optional[List[float]]): Embedding vector if available
- `access_control` (Dict[str, str]): Access control information

### `MongoConnectionManager` (class)
Manages MongoDB connections with pooling and error handling with methods:
- `connect()`: Establish asynchronous connection to MongoDB
- `connect_sync()`: Establish synchronous connection to MongoDB
- `disconnect()`: Close asynchronous connection to MongoDB
- `disconnect_sync()`: Close synchronous connection to MongoDB

### `DocumentCollection` (class)
Manages document operations in MongoDB with methods:
- `get_collection()`: Get the documents collection
- `create_indexes()`: Create necessary indexes for optimal performance
- `insert_document(rag_document)`: Insert a RAG document into MongoDB
- `insert_documents(rag_documents)`: Insert multiple RAG documents into MongoDB
- `find_by_id(rag_document_id)`: Find a document by its RAG ID
- `find_by_content_hash(content_hash)`: Find a document by its content hash
- `search(query, limit)`: Search for documents using text search
- `get_all(skip, limit)`: Get all documents with pagination
- `update_document(rag_document_id, rag_document)`: Update an existing document
- `delete_document(rag_document_id)`: Delete a document by its RAG ID
- `count_documents()`: Count total number of documents

### `MongoStorage` (class)
Main class for MongoDB storage operations with methods:
- `initialize()`: Initialize MongoDB connection and create indexes
- `close()`: Close MongoDB connection
- `store_document(rag_document)`: Store a single RAG document
- `store_documents(rag_documents)`: Store multiple RAG documents
- `retrieve_document(rag_document_id)`: Retrieve a RAG document by its ID
- `search_documents(query, limit)`: Search for documents using text search
- `get_all_documents(skip, limit)`: Get all documents with pagination
- `update_document(rag_document_id, rag_document)`: Update an existing document
- `delete_document(rag_document_id)`: Delete a document by its ID
- `get_document_count()`: Get total number of documents

Global instances `mongo_storage` and functions `initialize_mongo_storage`, `close_mongo_storage` are provided.

## 8. Query Processing (`src/retrieval/query_processing.py`)

This module implements sophisticated query processing for the RAG system, including query understanding, routing, and response generation. It handles complex query types, implements various retrieval strategies, and provides mechanisms for improving response quality and relevance.

### `QueryType` (Enum)
Enumeration for different types of queries:
- `SIMPLE_FACT`
- `COMPLEX_REASONING`
- `COMPARATIVE`
- `PROCEDURAL`
- `DEFINITIONAL`
- `ANALYTICAL`
- `UNCERTAIN`

### `QueryClassificationResult` (class)
Result of query classification with attributes:
- `query_type` (QueryType): Type of the query
- `confidence` (float): Confidence score of the classification
- `keywords` (List[str]): Important keywords identified in the query
- `entities` (List[str]): Named entities identified in the query
- `intent` (str): Intent of the query

### `QueryProcessingResult` (dataclass)
Result of query processing with attributes:
- `query` (str): Original query
- `response` (str): Generated response
- `sources` (List[RetrievalResult]): Retrieved sources
- `query_type` (QueryType): Type of the query
- `processing_time_ms` (float): Time taken for processing
- `confidence_score` (float): Confidence in the response
- `citations` (List[Dict[Any]]): Citations for the response
- `metadata` (Dict[Any]): Additional metadata about the processing

### `QueryClassifier` (class)
Classifier for determining query type and characteristics with method:
- `classify_query(query)`: Classify the query type and extract relevant information

### `QueryExpander` (class)
Expands queries to improve retrieval effectiveness with method:
- `expand_query(query)`: Expand the query with synonyms and related terms

### `ResponseGenerator` (class)
Generates responses based on retrieved context and query with method:
- `generate_response(query, context, max_new_tokens)`: Generate a response based on query and context

### `CitationExtractor` (class)
Extracts citations from responses and matches them to sources with method:
- `extract_citations(response, sources)`: Extract citations from the response and match them to sources

### `RAGQueryProcessor` (class)
Main class for processing RAG queries with advanced features with methods:
- `process_query(query, top_k)`: Process a query through the RAG system with advanced features
- `process_complex_query(query, top_k)`: Process a complex query that may require multiple steps or reasoning
- `_calculate_confidence_score(retrieval_results, classification_confidence)`: Calculate a confidence score for the response

### `QueryRouter` (class)
Routes queries to appropriate processing handlers based on type with method:
- `route_and_process(query, top_k)`: Route the query to appropriate processor based on its type

Global instances `query_router` and function `initialize_query_router` are provided.

## 9. Observability (`src/observability/__init__.py`)

This module implements comprehensive observability capabilities for RAG systems, following the 2026 production standards for monitoring, logging, and tracing. It provides insights into system performance, quality metrics, and operational health to ensure reliable production deployments.

### `LogLevel` (Enum)
Enumeration for log levels:
- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

### `LogEntry` (dataclass)
Structure for log entries with attributes:
- `timestamp` (datetime)
- `level` (LogLevel)
- `message` (str)
- `service` (str)
- `trace_id` (str)
- `span_id` (str)
- `properties` (Dict[Any])

### `MetricPoint` (dataclass)
Structure for metric data points with attributes:
- `name` (str)
- `value` (float)
- `timestamp` (datetime)
- `labels` (Dict[str, str])
- `unit` (str)

### `Logger` (class)
Structured logger for RAG systems with methods:
- `add_handler(handler)`: Add a log handler
- `debug(message, trace_id, span_id, **properties)`: Log a debug message
- `info(message, trace_id, span_id, **properties)`: Log an info message
- `warning(message, trace_id, span_id, **properties)`: Log a warning message
- `error(message, trace_id, span_id, **properties)`: Log an error message
- `critical(message, trace_id, span_id, **properties)`: Log a critical message
- `flush()`: Flush any buffered log entries

### `MetricsCollector` (class)
Collects and aggregates metrics for RAG systems with methods:
- `record_metric(name, value, labels, unit)`: Record a metric value
- `record_request(duration_ms, success, user_id)`: Record a request metric
- `get_percentile(percentile)`: Calculate a percentile of request latencies
- `get_error_rate()`: Calculate the error rate
- `export_metrics(name)`: Export metrics for a specific name
- `export_all()`: Export all collected metrics
- `reset()`: Reset all metrics

### `Tracer` (class)
Distributed tracing for RAG systems with methods:
- `start_span(operation_name, parent_trace_id)`: Start a new tracing span
- `end_span(span_id, status, attributes)`: End a tracing span
- `add_event(span_id, name, attributes)`: Add an event to a span
- `get_completed_spans()`: Get all completed spans
- `clear_completed_spans()`: Clear completed spans

### `ObservabilityContext` (class)
Context manager for observability operations with methods:
- `set_result(result, success)`: Set the result of the operation
- `add_property(key, value)`: Add a property to the context

### `ObservabilityManager` (class)
Main observability manager for RAG systems with methods:
- `trace_request(operation, user_id, **properties)`: Create a tracing context for a request
- `add_log_handler(handler)`: Add a custom log handler
- `export_logs_json()`: Export logs in JSON format
- `export_metrics_prometheus()`: Export metrics in Prometheus format

Global instance `default_obs_manager` and function `get_observability_manager` are provided.

## 10. Evaluation (`src/eval/__init__.py`)

This module implements comprehensive evaluation capabilities for RAG systems, following the 2026 production standards for measuring RAG quality. It includes integration with RAGAS (RAG Assessment) framework and custom evaluation metrics to assess different aspects of RAG performance.

### `EvaluationResult` (dataclass)
Structure for evaluation results with attributes:
- `question` (str): Original question
- `answer` (str): Generated answer
- `contexts` (List[str]): Retrieved contexts
- `ground_truth` (str): Ground truth answer
- `metrics` (Dict[str, float]): Evaluation metrics
- `metadata` (Dict[Any]): Additional metadata

### `RAGEvaluator` (class)
Main evaluation orchestrator with methods:
- `evaluate_single(question, answer, contexts, ground_truth)`: Evaluate a single query-response pair
- `evaluate_batch(queries)`: Evaluate multiple query-response pairs
- `get_context_recall(contexts, ground_truth)`: Calculate context recall metric
- `get_faithfulness(answer, contexts)`: Calculate faithfulness metric
- `get_answer_relevancy(question, answer)`: Calculate answer relevancy metric
- `get_context_precision(contexts, ground_truth)`: Calculate context precision metric
- `get_context_utilization(contexts, answer)`: Calculate context utilization metric

## 11. Chunking (`src/chunking/__init__.py`)

This module implements various text chunking strategies for the RAG system, allowing for optimal document segmentation based on content type and retrieval needs. Different chunking strategies are available to handle various document types and preserving semantic coherence while enabling effective retrieval.

### `ChunkingStrategy` (Enum)
Enumeration for chunking strategies:
- `RECURSIVE`: Recursive character chunking
- `SEMANTIC`: Semantic chunking based on sentence boundaries
- `CODE`: Code-specific chunking for programming documents
- `MARKDOWN`: Markdown-aware chunking preserving structure
- `PARAGRAPH`: Paragraph-aware chunking

### `ChunkingConfig` (BaseModel)
Configuration for chunking with the following attributes:
- `chunk_size` (int): Size of each chunk (default: 1000, range: 100-10000)
- `chunk_overlap` (int): Overlap between chunks (default: 200, range: 0-1000)
- `strategy` (ChunkingStrategy): Chunking strategy to use (default: ChunkingStrategy.RECURSIVE)
- `separators` (List[str]): Separators to use for recursive chunking (default: ["\n\n", "\n", " ", ""])
- `length_function` (str): Function to use for calculating length (default: "len")

### `Chunk` (dataclass)
Structure for text chunks with attributes:
- `id` (str): Unique identifier for the chunk
- `content` (str): Content of the chunk
- `metadata` (Dict[Any]): Metadata associated with the chunk
- `source_document_id` (str): ID of the source document
- `position` (int): Position of the chunk in the source document

### `BaseChunker` (ABC)
Abstract base class for chunkers with abstract method:
- `chunk_document(document, config)`: Chunk a document according to the configuration

### `RecursiveCharacterChunker` (BaseChunker)
Recursive character chunker implementation with method:
- `chunk_document(document, config)`: Chunk a document using recursive character splitting

### `SemanticChunker` (BaseChunker)
Semantic chunker implementation with method:
- `chunk_document(document, config)`: Chunk a document using semantic boundaries

### `CodeChunker` (BaseChunker)
Code-specific chunker implementation with method:
- `chunk_document(document, config)`: Chunk a document using code-aware splitting

### `MarkdownChunker` (BaseChunker)
Markdown-aware chunker implementation with method:
- `chunk_document(document, config)`: Chunk a document preserving markdown structure

### `ParagraphChunker` (BaseChunker)
Paragraph-aware chunker implementation with method:
- `chunk_document(document, config)`: Chunk a document using paragraph boundaries

### `ChunkerFactory` (class)
Factory for creating appropriate chunker implementations with static method:
- `create_chunker(strategy)`: Create a chunker instance based on the strategy

## 12. Indexing Service (`src/services/indexing.py`)

This module provides business logic for indexing operations, coordinating the storage of documents in both MongoDB and vector stores.

### `index_documents` (function)
Coordinates indexing to both MongoDB and vector stores with parameters:
- `rag_pipeline`: RAG pipeline instance
- `documents`: Iterable of documents to index
- `persist_vectors`: Whether to persist vectors to vector store (default: True)
- `persist_documents`: Whether to persist documents to MongoDB (default: True)