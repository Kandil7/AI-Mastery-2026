# Core Components

This section provides detailed documentation for the core components of the Production RAG System, including their responsibilities, interfaces, and implementation details.

## 1. API Layer

### FastAPI Application (`api.py`)

The main FastAPI application serves as the entry point for the system, exposing RESTful endpoints for document indexing and querying.

#### Key Responsibilities:
- Expose RESTful endpoints for system interaction
- Handle request validation and response formatting
- Manage application lifecycle (startup/shutdown)
- Integrate with observability systems
- Handle error responses and status codes

#### Endpoints:
- `GET /`: Basic health check
- `GET /health`: Detailed health status
- `POST /index`: Add documents to knowledge base
- `POST /query`: Query the RAG system
- `POST /upload`: Upload documents for indexing
- `GET /documents`: List stored documents
- `GET /documents/{doc_id}`: Get specific document
- `GET /metrics`: Prometheus-compatible metrics

#### Request Models:
- `DocumentRequest`: Model for adding documents to the knowledge base
- `QueryRequest`: Model for querying the RAG system
- `DocumentSource`: Enumeration for document sources

#### Response Models:
- `QueryResponse`: Response model for RAG queries
- `SourceDocument`: Model representing a source document in the response
- `HealthStatus`: Model for health check responses

## 2. RAG Pipeline (`src/pipeline.py`)

The RAG Pipeline orchestrates the complete RAG workflow, combining retrieval and generation processes.

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

## 3. Retrieval System (`src/retrieval/`)

The retrieval system handles document storage and retrieval using multiple strategies.

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

The vector storage system manages vector embeddings for documents and provides efficient similarity search capabilities.

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

### `VectorManager` (class)
Manager class for handling vector operations in the RAG system with methods:
- `initialize()`: Initialize the vector manager and underlying store
- `add_document_vector(document_id, vector, text_content, metadata)`: Add a vector representation of a document
- `search_similar(query_vector, k)`: Search for similar vectors to the query vector
- `get_vector(vector_id)`: Get a vector by ID
- `delete_vector(vector_id)`: Delete a vector by ID
- `update_vector(vector_record)`: Update an existing vector
- `get_count()`: Get the total number of vectors in the store
- `close()`: Close the vector manager and underlying store

## 5. Ingestion Pipeline (`src/ingestion/`)

The ingestion pipeline handles document processing from various formats.

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

### `IngestionPipeline` (class)
Main ingestion pipeline orchestrator with methods:
- `ingest_from_file(file, ingestion_request)`: Ingest documents from an uploaded file
- `ingest_from_text(text, ingestion_request, doc_id, title)`: Ingest documents from raw text content
- `ingest_batch(documents, ingestion_request)`: Ingest a batch of documents

### `FileManager` (class)
Main file manager class that handles file uploads, validation, and processing with methods:
- `get_file_type(filename)`: Determine the file type based on extension
- `validate_file_upload(request)`: Validate file upload request
- `save_uploaded_file(content, original_filename)`: Save uploaded file to temporary location
- `process_file(filepath, filename, metadata)`: Process a file and extract documents
- `cleanup_temp_file(filepath)`: Clean up temporary file
- `cleanup_old_temp_files(max_age_hours)`: Clean up old temporary files

## 6. Configuration (`src/config.py`)

The configuration system provides centralized configuration management.

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

## 7. Evaluation (`src/eval/`)

The evaluation system provides comprehensive evaluation capabilities for RAG systems.

### `RAGEvaluator` (class)
Main evaluation orchestrator with methods:
- `evaluate_single(question, answer, contexts, ground_truth)`: Evaluate a single query-response pair
- `evaluate_batch(queries)`: Evaluate multiple query-response pairs
- `get_context_recall(contexts, ground_truth)`: Calculate context recall metric
- `get_faithfulness(answer, contexts)`: Calculate faithfulness metric
- `get_answer_relevancy(question, answer)`: Calculate answer relevancy metric
- `get_context_precision(contexts, ground_truth)`: Calculate context precision metric
- `get_context_utilization(contexts, answer)`: Calculate context utilization metric

## 8. Observability (`src/observability/`)

The observability system provides monitoring and logging infrastructure.

### `ObservabilityManager` (class)
Main observability manager for RAG systems with methods:
- `trace_request(operation, user_id, **properties)`: Create a tracing context for a request
- `add_log_handler(handler)`: Add a custom log handler
- `export_logs_json()`: Export logs in JSON format
- `export_metrics_prometheus()`: Export metrics in Prometheus format

## 9. Chunking (`src/chunking/`)

The chunking system provides various text chunking strategies.

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

## 10. Services (`src/services/indexing.py`)

The indexing service provides business logic for indexing operations.

### `index_documents` (function)
Coordinates indexing to both MongoDB and vector stores with parameters:
- `rag_pipeline`: RAG pipeline instance
- `documents`: Iterable of documents to index
- `persist_vectors`: Whether to persist vectors to vector store (default: True)
- `persist_documents`: Whether to persist documents to MongoDB (default: True)