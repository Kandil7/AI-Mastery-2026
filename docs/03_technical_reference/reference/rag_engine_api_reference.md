# 📖 API Reference

> Complete API documentation for RAG Engine Mini.

---

## Base URL

```
http://localhost:8000
```

## Authentication

All endpoints (except health checks) require an API key in the header:

```
X-API-KEY: your_api_key_here
```

---

## Health Endpoints

### GET /health

Basic health check.

**Response:**
```json
{
  "status": "ok",
  "env": "dev",
  "app_name": "rag-engine-mini"
}
```

### GET /health/ready

Readiness probe for Kubernetes.

**Response:**
```json
{
  "ready": true,
  "checks": {
    "database": "ok",
    "redis": "ok",
    "qdrant": "ok"
  }
}
```

---

## Document Endpoints

### POST /api/v1/documents/upload

Upload a document for indexing.

**Headers:**
```
X-API-KEY: your_api_key
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "X-API-KEY: demo_api_key_12345678" \
  -F "file=@document.pdf"
```

**Response (202 Accepted):**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Document queued for indexing"
}
```

**Response (409 Conflict - Duplicate):**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "already_exists",
  "message": "Document with same content already indexed"
}
```

**Supported File Types:**
- PDF (`application/pdf`)
- DOCX (`application/vnd.openxmlformats-officedocument.wordprocessingml.document`)
- TXT (`text/plain`)

**Limits:**
- Max file size: 20 MB (configurable)

---

### GET /api/v1/documents/{document_id}/status

Get document processing status.

**Request:**
```bash
curl http://localhost:8000/api/v1/documents/550e8400-e29b-41d4-a716-446655440000/status \
  -H "X-API-KEY: demo_api_key_12345678"
```

**Response:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "status": "indexed",
  "chunks_count": 42,
  "created_at": "2026-01-29T12:00:00Z",
  "updated_at": "2026-01-29T12:01:30Z"
}
```

**Status Values:**
| Status | Description |
|--------|-------------|
| `created` | Document uploaded, not yet queued |
| `queued` | Waiting in indexing queue |
| `processing` | Currently being indexed |
| `indexed` | Successfully indexed |
| `failed` | Indexing failed (see `error` field) |

---

### GET /api/v1/documents

List documents for the authenticated user.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 100 | Max results (1-1000) |
| `offset` | int | 0 | Pagination offset |

**Request:**
```bash
curl "http://localhost:8000/api/v1/documents?limit=10&offset=0" \
  -H "X-API-KEY: demo_api_key_12345678"
```

**Response:**
```json
{
  "documents": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "document.pdf",
      "status": "indexed",
      "chunks_count": 42,
      "created_at": "2026-01-29T12:00:00Z"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

---

### DELETE /api/v1/documents/{document_id}

Delete a document and its chunks.

**Request:**
```bash
curl -X DELETE http://localhost:8000/api/v1/documents/550e8400-e29b-41d4-a716-446655440000 \
  -H "X-API-KEY: demo_api_key_12345678"
```

**Response (200 OK):**
```json
{
  "deleted": true,
  "document_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Query Endpoints

### POST /api/v1/queries/ask-hybrid

Ask a question using hybrid retrieval (vector + keyword).

**Headers:**
```
X-API-KEY: your_api_key
Content-Type: application/json
```

**Request Body:**
```json
{
  "question": "What is machine learning?",
  "document_id": null,
  "k_vec": 30,
  "k_kw": 30,
  "rerank_top_n": 8
}
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | required | The question to answer (2-2000 chars) |
| `document_id` | string | null | Restrict to single document (ChatPDF mode) |
| `k_vec` | int | 30 | Top-K for vector search (1-200) |
| `k_kw` | int | 30 | Top-K for keyword search (1-200) |
| `rerank_top_n` | int | 8 | Results after reranking (1-50) |

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/queries/ask-hybrid \
  -H "X-API-KEY: demo_api_key_12345678" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main types of machine learning?",
    "k_vec": 30,
    "rerank_top_n": 8
  }'
```

**Response:**
```json
{
  "answer": "Based on the provided context, the main types of machine learning are:\n\n1. **Supervised Learning**: The algorithm learns from labeled training data...\n\n2. **Unsupervised Learning**: Works with unlabeled data to find patterns...\n\n3. **Reinforcement Learning**: An agent learns by interacting with an environment...",
  "sources": [
    "chunk_abc123",
    "chunk_def456",
    "chunk_ghi789"
  ]
}
```

---

### POST /api/v1/queries/ask

Alias for `/ask-hybrid`.

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

### Common HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request (validation error) |
| 401 | Unauthorized (missing/invalid API key) |
| 404 | Not Found |
| 413 | File Too Large |
| 415 | Unsupported Media Type |
| 429 | Rate Limited |
| 500 | Internal Server Error |

### Example Error Responses

**401 Unauthorized:**
```json
{
  "detail": "Invalid or missing API key"
}
```

**413 Payload Too Large:**
```json
{
  "detail": "File too large: 25.5MB exceeds limit of 20.0MB"
}
```

**415 Unsupported Media Type:**
```json
{
  "detail": "Unsupported file type: .xlsx. Allowed: pdf, docx, txt"
}
```

---

## Rate Limiting

| Endpoint | Limit |
|----------|-------|
| `/api/v1/documents/upload` | 10/minute |
| `/api/v1/queries/ask-hybrid` | 60/minute |
| All other endpoints | 120/minute |

Rate limit headers:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 55
X-RateLimit-Reset: 1706529600
```

---

## Webhooks (Future)

Planned webhook events:
- `document.indexed` - Document successfully indexed
- `document.failed` - Document indexing failed

---

## OpenAPI Specification

Full OpenAPI 3.0 spec available at:
```
GET /openapi.json
GET /docs       # Swagger UI
GET /redoc      # ReDoc
```
