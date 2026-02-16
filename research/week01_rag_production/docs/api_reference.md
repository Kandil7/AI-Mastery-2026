# API Reference

This section provides comprehensive documentation for all API endpoints available in the Production RAG System.

## Base URL

All API endpoints are served from the base URL:
```
http://localhost:8000
```
or
```
https://your-domain.com
```

## Authentication

Most endpoints do not require authentication by default. However, authentication can be enabled through configuration:

- **Header**: `Authorization: Bearer {token}`
- **Configuration**: Set `SECURITY__ENABLE_AUTHENTICATION=true`

## Common Headers

- `Content-Type: application/json`
- `Accept: application/json`

## Error Responses

All error responses follow the same structure:

```json
{
  "detail": "Error message describing the issue"
}
```

## Endpoints

### 1. Health Check

#### GET /

Basic health check endpoint.

**Description**: Returns basic status information about the API.

**Response Codes**:
- `200`: Success
- `503`: Service unavailable

**Response Body**:
```json
{
  "status": "healthy",
  "message": "Production RAG API is running",
  "version": "1.0.0",
  "endpoints": ["/docs", "/health", "/query", "/index"]
}
```

**Example Request**:
```bash
curl -X GET http://localhost:8000/
```

**Example Response**:
```
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "healthy",
  "message": "Production RAG API is running",
  "version": "1.0.0",
  "endpoints": ["/docs", "/health", "/query", "/index"]
}
```

### 2. Detailed Health Check

#### GET /health

Comprehensive health status with detailed information.

**Description**: Returns detailed health status including model status, query capability, and document count.

**Response Codes**:
- `200`: Success
- `503`: Service unavailable

**Response Body**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "details": {
    "model_status": "initialized",
    "can_query": true,
    "document_count": 100,
    "service": "rag-api"
  }
}
```

**Example Request**:
```bash
curl -X GET http://localhost:8000/health
```

**Example Response**:
```
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "details": {
    "model_status": "initialized",
    "can_query": true,
    "document_count": 100,
    "service": "rag-api"
  }
}
```

### 3. Index Documents

#### POST /index

Add documents to the knowledge base.

**Description**: Adds multiple documents to the RAG system's index. Documents are validated before being added to prevent malicious content.

**Request Body**:
Array of `DocumentRequest` objects with the following structure:
- `id` (string): Unique identifier for the document (alphanumeric, underscores, hyphens)
- `content` (string): The actual text content of the document (1-10000 characters)
- `metadata` (object): Additional metadata about the document
- `source` (string): Source of the document (default: "manual")
- `doc_type` (string): Type of document (default: "unspecified")

**Response Codes**:
- `200`: Success
- `400`: Bad request (no documents provided)
- `413`: Too many documents in request
- `422`: Validation error
- `500`: Internal server error
- `503`: RAG model not initialized

**Response Body**:
```json
{
  "message": "Successfully added 2 documents",
  "total_docs": 102,
  "added_docs": ["doc_123", "doc_456"]
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '[
    {
      "id": "doc_123",
      "content": "This is the content of document 1",
      "metadata": {"source": "web", "category": "tech"},
      "source": "manual",
      "doc_type": "article"
    },
    {
      "id": "doc_456",
      "content": "This is the content of document 2",
      "metadata": {"source": "pdf", "author": "John Doe"},
      "source": "file_upload",
      "doc_type": "report"
    }
  ]'
```

**Example Response**:
```
HTTP/1.1 200 OK
Content-Type: application/json

{
  "message": "Successfully added 2 documents",
  "total_docs": 102,
  "added_docs": ["doc_123", "doc_456"]
}
```

### 4. Query Documents

#### POST /query

Query the RAG system to get an answer with supporting evidence.

**Description**: Processes a natural language query and returns a generated response along with the source documents that informed the answer.

**Request Body**:
`QueryRequest` object with the following structure:
- `query` (string): The search query (1-500 characters)
- `k` (integer): Number of documents to retrieve (1-20, default: 3)
- `include_sources` (boolean): Whether to include source documents in response (default: true)
- `timeout_seconds` (number): Timeout for the query operation (1-60 seconds, default: 30.0)

**Response Codes**:
- `200`: Success
- `408`: Query timed out
- `422`: Invalid query parameters
- `500`: Internal error processing query
- `503`: RAG model not initialized

**Response Body**:
```json
{
  "query": "What is RAG?",
  "response": "RAG stands for Retrieval Augmented Generation...",
  "sources": [
    {
      "id": "doc_123",
      "content": "RAG stands for Retrieval Augmented Generation...",
      "score": 0.95,
      "rank": 1,
      "metadata": {"source": "web", "category": "tech"}
    }
  ],
  "query_time_ms": 125.5,
  "total_documents_indexed": 102
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "k": 3,
    "include_sources": true,
    "timeout_seconds": 30.0
  }'
```

**Example Response**:
```
HTTP/1.1 200 OK
Content-Type: application/json

{
  "query": "What is RAG?",
  "response": "RAG stands for Retrieval Augmented Generation...",
  "sources": [
    {
      "id": "doc_123",
      "content": "RAG stands for Retrieval Augmented Generation...",
      "score": 0.95,
      "rank": 1,
      "metadata": {"source": "web", "category": "tech"}
    }
  ],
  "query_time_ms": 125.5,
  "total_documents_indexed": 102
}
```

### 5. Advanced Query

#### POST /advanced_query

Query the RAG system using advanced query processing with classification and routing.

**Description**: Processes a natural language query using advanced techniques including query classification, expansion, and multi-step reasoning.

**Request Body**:
Same as `/query` endpoint.

**Response Codes**:
- `200`: Success
- `408`: Query timed out
- `422`: Invalid query parameters
- `500`: Internal error processing query
- `503`: Query router not initialized

**Response Body**:
Same structure as `/query` endpoint but with additional `query_type` field in the result.

### 6. Upload Document

#### POST /upload

Upload a document for indexing in the RAG system.

**Description**: Accepts file uploads and processes them for inclusion in the knowledge base. Handles various document formats and performs automatic chunking and indexing.

**Parameters**:
- `file` (file): The document file to upload
- `chunk_size` (integer): Size of text chunks for processing (default: 1000)
- `chunk_overlap` (integer): Overlap between chunks (default: 200)
- `metadata` (string): Additional metadata as JSON string (default: "{}")

**Response Codes**:
- `200`: Success
- `400`: File validation failed or invalid metadata JSON
- `500`: Error processing document
- `503`: Ingestion pipeline not initialized

**Response Body**:
```json
{
  "message": "Processed 1 documents, indexed 1",
  "processed_documents": 1,
  "indexed_documents": 1,
  "processing_time_ms": 150.2,
  "warnings": []
}
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200" \
  -F 'metadata={"source": "uploaded", "category": "manual"}'
```

**Example Response**:
```
HTTP/1.1 200 OK
Content-Type: application/json

{
  "message": "Processed 1 documents, indexed 1",
  "processed_documents": 1,
  "indexed_documents": 1,
  "processing_time_ms": 150.2,
  "warnings": []
}
```

### 7. List Documents

#### GET /documents

List documents in the system.

**Description**: Retrieves a paginated list of documents stored in the system.

**Query Parameters**:
- `skip` (integer): Number of documents to skip (default: 0)
- `limit` (integer): Maximum number of documents to return (default: 100, max: 1000)

**Response Codes**:
- `200`: Success
- `500`: Error retrieving documents

**Response Body**:
```json
{
  "documents": [
    {
      "id": "doc_123",
      "source": "web",
      "doc_type": "article",
      "metadata": {"source": "web", "category": "tech"},
      "content_preview": "This is the beginning of the document content..."
    }
  ],
  "total_count": 102,
  "returned_count": 1,
  "skip": 0,
  "limit": 1
}
```

**Example Request**:
```bash
curl -X GET "http://localhost:8000/documents?skip=0&limit=10"
```

**Example Response**:
```
HTTP/1.1 200 OK
Content-Type: application/json

{
  "documents": [
    {
      "id": "doc_123",
      "source": "web",
      "doc_type": "article",
      "metadata": {"source": "web", "category": "tech"},
      "content_preview": "This is the beginning of the document content..."
    }
  ],
  "total_count": 102,
  "returned_count": 1,
  "skip": 0,
  "limit": 1
}
```

### 8. Get Document

#### GET /documents/{doc_id}

Get a specific document by ID.

**Description**: Retrieves a specific document by its unique identifier.

**Path Parameters**:
- `doc_id` (string): ID of the document to retrieve

**Response Codes**:
- `200`: Success
- `404`: Document not found
- `500`: Error retrieving document

**Response Body**:
```json
{
  "id": "doc_123",
  "content": "Full content of the document...",
  "source": "web",
  "doc_type": "article",
  "metadata": {"source": "web", "category": "tech"},
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

**Example Request**:
```bash
curl -X GET http://localhost:8000/documents/doc_123
```

**Example Response**:
```
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "doc_123",
  "content": "Full content of the document...",
  "source": "web",
  "doc_type": "article",
  "metadata": {"source": "web", "category": "tech"},
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### 9. Metrics

#### GET /metrics

Prometheus-compatible metrics endpoint.

**Description**: Returns metrics in Prometheus text format for monitoring and alerting.

**Response Codes**:
- `200`: Success

**Response Body**:
Plain text in Prometheus format:
```
# HELP request_count Total number of requests
# TYPE request_count counter
request_count{service="rag-service",success="true"} 50
request_count{service="rag-service",success="false"} 2

# HELP request_duration_ms Request duration in milliseconds
# TYPE request_duration_ms histogram
request_duration_ms{service="rag-service",quantile="0.5"} 120.5
request_duration_ms{service="rag-service",quantile="0.95"} 450.2
request_duration_ms{service="rag-service",quantile="0.99"} 890.7
```

**Example Request**:
```bash
curl -X GET http://localhost:8000/metrics
```

**Example Response**:
```
HTTP/1.1 200 OK
Content-Type: text/plain

# HELP request_count Total number of requests
# TYPE request_count counter
request_count{service="rag-service",success="true"} 50
request_count{service="rag-service",success="false"} 2

# HELP request_duration_ms Request duration in milliseconds
# TYPE request_duration_ms histogram
request_duration_ms{service="rag-service",quantile="0.5"} 120.5
request_duration_ms{service="rag-service",quantile="0.95"} 450.2
request_duration_ms{service="rag-service",quantile="0.99"} 890.7
```

## Data Models

### DocumentRequest
Request model for adding documents to the knowledge base.

**Fields**:
- `id` (string): Unique identifier for the document (alphanumeric, underscores, hyphens)
- `content` (string): The actual text content of the document (1-10000 characters)
- `metadata` (object): Additional metadata about the document
- `source` (string): Source of the document (enum: manual, file_upload, database, web_crawler, api_import)
- `doc_type` (string): Type of document (default: "unspecified", 1-50 characters)

### QueryRequest
Request model for querying the RAG system.

**Fields**:
- `query` (string): The search query (1-500 characters)
- `k` (integer): Number of documents to retrieve (1-20, default: 3)
- `include_sources` (boolean): Whether to include source documents in response (default: true)
- `timeout_seconds` (number): Timeout for the query operation (1-60 seconds, default: 30.0)

### QueryResponse
Response model for RAG queries.

**Fields**:
- `query` (string): Original query
- `response` (string): Generated response
- `sources` (array): Retrieved source documents
- `query_time_ms` (number): Time taken for the query
- `total_documents_indexed` (integer): Total number of documents in the index

### SourceDocument
Model representing a source document in the response.

**Fields**:
- `id` (string): Document identifier
- `content` (string): Document content snippet
- `score` (number): Relevance score
- `rank` (integer): Rank in the results
- `metadata` (object): Document metadata

### HealthStatus
Model for health check responses.

**Fields**:
- `status` (string): Overall health status
- `timestamp` (string): ISO timestamp of the check
- `details` (object): Detailed health information

## Error Codes

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request - Invalid input parameters
- `404`: Not Found - Requested resource not found
- `408`: Request Timeout - Query took too long
- `413`: Payload Too Large - Too many documents in request
- `422`: Unprocessable Entity - Validation error
- `500`: Internal Server Error - General server error
- `503`: Service Unavailable - Service not initialized or unavailable

### Common Error Messages
- `"RAG model not initialized"`: The RAG pipeline hasn't been initialized
- `"No documents provided"`: Empty document array in index request
- `"Too many documents in request"`: More than 100 documents in index request
- `"Validation error"`: Input validation failed
- `"Query timed out"`: Query exceeded timeout threshold
- `"Document not found"`: Requested document ID doesn't exist