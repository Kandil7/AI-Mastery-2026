# REST API Reference

**Version**: 1.0.0  
**Last Updated**: March 27, 2026

---

## 📡 API Overview

The RAG system provides a RESTful API for querying and management.

**Base URL**: `http://localhost:8000`

**Interactive Docs**: `http://localhost:8000/docs`

---

## 🔐 Authentication

Currently, the API is open for local development. For production:

```bash
# Add to .env
API_KEY=your-secret-key

# Include in requests
curl -H "Authorization: Bearer your-secret-key" ...
```

---

## 📚 Endpoints

### Health Check

#### `GET /health`

Check system health.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "stats": {
    "total_chunks": 170000,
    "vector_count": 170000
  }
}
```

---

### Query

#### `POST /api/v1/query`

Query the RAG system.

**Request Body:**
```json
{
  "query": "ما هو التوحيد في الإسلام؟",
  "top_k": 5,
  "filters": {
    "category": "العقيدة"
  },
  "stream": false
}
```

**Response:**
```json
{
  "query": "ما هو التوحيد في الإسلام؟",
  "answer": "التوحيد في الإسلام هو إفراد الله بالعبادة...",
  "sources": [
    {
      "book_title": "التوحيد",
      "author": "ابن تيمية",
      "category": "العقيدة",
      "score": 0.95,
      "content_preview": "التوحيد هو إفراد الله..."
    }
  ],
  "latency_ms": 234.5,
  "tokens_used": 450,
  "model": "gpt-4o"
}
```

**Error Response:**
```json
{
  "detail": "Pipeline not initialized. Please index documents first."
}
```

---

### Streaming Query

#### `POST /api/v1/query/stream`

Stream query response.

**Request Body:**
```json
{
  "query": "ما هو التوحيد؟",
  "top_k": 5
}
```

**Response**: Server-Sent Events (SSE)

```
data: {"token": "الت"}
data: {"token": "وحيد"}
data: {"token": "هو"}
...
data: [DONE]
```

---

### Index Documents

#### `POST /api/v1/index`

Start indexing documents.

**Request Body:**
```json
{
  "limit": 100,
  "categories": ["التفسير", "الحديث"]
}
```

**Response:**
```json
{
  "status": "started",
  "message": "Indexing started in background",
  "progress": {
    "status": "started"
  }
}
```

---

### Index Status

#### `GET /api/v1/index/status`

Check indexing progress.

**Response:**
```json
{
  "status": "in_progress",
  "message": "Indexing in progress",
  "progress": {
    "status": "indexing",
    "progress": 45,
    "current": 45,
    "total": 100
  }
}
```

---

### Statistics

#### `GET /api/v1/stats`

Get system statistics.

**Response:**
```json
{
  "indexed": true,
  "total_chunks": 170000,
  "bm25_documents": 170000,
  "vector_count": 170000,
  "config": {
    "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "llm_model": "gpt-4o",
    "chunk_size": 512
  }
}
```

---

### Categories

#### `GET /api/v1/categories`

Get available categories.

**Response:**
```json
{
  "categories": [
    "التفسير",
    "كتب السنة",
    "الفقه الحنفي",
    "الفقه المالكي",
    "الفقه الشافعي",
    "الفقه الحنبلي",
    "العقيدة",
    "اللغة العربية",
    "التاريخ",
    "الرقائق"
  ]
}
```

---

## 🔧 Usage Examples

### cURL Examples

**Basic Query:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ما هو التوحيد؟",
    "top_k": 5
  }'
```

**Filtered Query:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ما حكم الصلاة؟",
    "top_k": 3,
    "filters": {
      "category": "الفقه العام"
    }
  }'
```

**Streaming Query:**
```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هو التوحيد؟"}' \
  -N
```

**Start Indexing:**
```bash
curl -X POST http://localhost:8000/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 100,
    "categories": ["التفسير"]
  }'
```

### Python Examples

**Using requests:**
```python
import requests

# Basic query
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query": "ما هو التوحيد؟",
        "top_k": 5
    }
)
result = response.json()
print(result['answer'])

# Streaming query
response = requests.post(
    "http://localhost:8000/api/v1/query/stream",
    json={"query": "ما هو التوحيد؟"},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode())
```

**Using httpx (async):**
```python
import httpx
import asyncio

async def query():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/query",
            json={"query": "ما هو التوحيد؟"}
        )
        return response.json()

result = asyncio.run(query())
```

---

## ⚠️ Error Handling

### Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid JSON) |
| 404 | Not Found |
| 500 | Internal Server Error |
| 503 | Service Unavailable (not indexed) |

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

---

## 📊 Rate Limiting

Default rate limits (configurable):

- **100 requests per minute** per IP
- **1000 requests per hour** per IP

**Rate Limit Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1647360000
```

---

## 🔗 Related Documents

- [Python API Reference](python_api.md) - Python client library
- [API Examples](examples.md) - More usage examples
- [Deployment Guide](../06_deployment/docker.md) - Deploy with API

---

**Interactive API Docs**: Visit `http://localhost:8000/docs` when server is running
