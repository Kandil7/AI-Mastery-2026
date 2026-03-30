# API Documentation

<div align="center">

![API Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![API Status](https://img.shields.io/badge/status-stable-green.svg)

**Complete API reference for AI-Mastery-2026**

[Overview](#-overview) • [Authentication](authentication.md) • [Endpoints](endpoints/) • [SDK](sdk/) • [Examples](examples/)

</div>

---

## 📖 Overview

The AI-Mastery-2026 API provides programmatic access to all core functionalities including:

- LLM inference and fine-tuning
- RAG system operations
- Vector database management
- Model evaluation and benchmarking
- Agent orchestration

### API Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Applications                     │
│         (Web App, Mobile, CLI, Third-party Integrations)     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway                             │
│              (Rate Limiting, Auth, Routing)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    REST API Layer                            │
│              (FastAPI Endpoints)                             │
├─────────────────────────────────────────────────────────────┤
│  /v1/chat      /v1/completions    /v1/embeddings            │
│  /v1/rag       /v1/models         /v1/fine-tuning           │
│  /v1/agents    /v1/evaluation     /v1/vector-db             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Service Layer                              │
│           (Business Logic & Orchestration)                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Core Implementations                         │
│    (LLM, RAG, Embeddings, Vector DB, Agents, Evaluation)     │
└─────────────────────────────────────────────────────────────┘
```

### Base URL

```
Production:  https://api.ai-mastery-2026.com/v1
Staging:     https://staging-api.ai-mastery-2026.com/v1
Local:       http://localhost:8000/v1
```

### API Versions

| Version | Status | End of Life |
|---------|--------|-------------|
| v1 | Current | - |
| v0 | Deprecated | April 30, 2026 |

---

## 🔐 Authentication

All API requests require authentication using API keys.

### Getting an API Key

1. Log in to your dashboard
2. Navigate to Settings → API Keys
3. Click "Create New Key"
4. Copy and securely store your key

### Using API Keys

Include your API key in the `Authorization` header:

```bash
curl -X GET "https://api.ai-mastery-2026.com/v1/models" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

```python
from ai_mastery import Client

client = Client(api_key="YOUR_API_KEY")
```

### Authentication Methods

| Method | Use Case | Security Level |
|--------|----------|----------------|
| API Key | Server-to-server | High |
| OAuth 2.0 | User-facing apps | Highest |
| JWT | Session-based | High |

See [Authentication Guide](authentication.md) for details.

---

## 📡 Endpoints

### Chat & Completion

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat with LLM |
| `/v1/completions` | POST | Text completion |
| `/v1/chat/stream` | POST | Streaming chat |

### RAG System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/rag/query` | POST | Query RAG system |
| `/v1/rag/documents` | POST | Add documents |
| `/v1/rag/documents/{id}` | DELETE | Remove document |
| `/v1/rag/collections` | GET | List collections |

### Embeddings

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/embeddings` | POST | Generate embeddings |
| `/v1/embeddings/models` | GET | List embedding models |

### Models

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/models/{id}` | GET | Get model details |
| `/v1/models/{id}/deploy` | POST | Deploy model |

### Fine-Tuning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/fine-tuning/jobs` | POST | Create fine-tuning job |
| `/v1/fine-tuning/jobs/{id}` | GET | Get job status |
| `/v1/fine-tuning/jobs/{id}/cancel` | POST | Cancel job |

### Vector Database

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/vector-db/collections` | POST | Create collection |
| `/v1/vector-db/collections/{id}` | GET | Get collection |
| `/v1/vector-db/search` | POST | Vector search |
| `/v1/vector-db/upsert` | POST | Upsert vectors |

### Agents

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/agents` | POST | Create agent |
| `/v1/agents/{id}` | GET | Get agent |
| `/v1/agents/{id}/run` | POST | Run agent |
| `/v1/agents/{id}/tasks` | POST | Add task |

### Evaluation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/evaluation/benchmark` | POST | Run benchmark |
| `/v1/evaluation/metrics` | GET | Get metrics |
| `/v1/evaluation/compare` | POST | Compare models |

---

## 💻 SDK Documentation

### Python SDK

#### Installation

```bash
pip install ai-mastery-sdk
```

#### Quick Start

```python
from ai_mastery import Client

# Initialize client
client = Client(api_key="YOUR_API_KEY")

# Chat completion
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# RAG query
result = client.rag.query(
    collection="my_docs",
    query="What is machine learning?"
)
print(result.answer)

# Generate embeddings
embeddings = client.embeddings.create(
    model="text-embedding-ada-002",
    input=["Hello world", "AI is amazing"]
)
```

#### SDK Reference

| Module | Description |
|--------|-------------|
| `client.chat` | Chat completions |
| `client.rag` | RAG operations |
| `client.embeddings` | Embedding generation |
| `client.models` | Model management |
| `client.fine_tuning` | Fine-tuning jobs |
| `client.vector_db` | Vector database |
| `client.agents` | Agent orchestration |
| `client.evaluation` | Model evaluation |

### JavaScript SDK

#### Installation

```bash
npm install @ai-mastery/sdk
```

#### Quick Start

```javascript
import { Client } from '@ai-mastery/sdk';

const client = new Client({ apiKey: 'YOUR_API_KEY' });

// Chat completion
const response = await client.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello!' }]
});

console.log(response.choices[0].message.content);
```

### CLI Tool

#### Installation

```bash
pip install ai-mastery-cli
```

#### Usage

```bash
# Authenticate
ai-mastery auth login

# List models
ai-mastery models list

# Chat
ai-mastery chat --model gpt-4 --message "Hello!"

# RAG query
ai-mastery rag query --collection docs --query "What is AI?"

# Generate embeddings
ai-mastery embeddings create --input "text.txt" --output vectors.json
```

---

## 📊 Rate Limits

| Tier | Requests/min | Tokens/min | Concurrent |
|------|--------------|------------|------------|
| Free | 60 | 10,000 | 5 |
| Pro | 600 | 100,000 | 20 |
| Enterprise | 6,000 | 1,000,000 | 100 |

### Rate Limit Headers

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1647360000
```

### Handling Rate Limits

```python
from ai_mastery import Client, RateLimitError

client = Client(api_key="YOUR_API_KEY")

try:
    response = client.chat.completions.create(...)
except RateLimitError as e:
    # Wait and retry
    import time
    time.sleep(e.retry_after)
    response = client.chat.completions.create(...)
```

---

## ❌ Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "invalid_request",
    "message": "The request was invalid",
    "type": "invalid_request_error",
    "param": "model",
    "details": {
      "field": "model",
      "reason": "Model not found"
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request` | 400 | Invalid request parameters |
| `authentication_error` | 401 | Invalid or missing API key |
| `authorization_error` | 403 | Insufficient permissions |
| `not_found` | 404 | Resource not found |
| `rate_limit_error` | 429 | Rate limit exceeded |
| `internal_error` | 500 | Internal server error |
| `service_unavailable` | 503 | Service temporarily unavailable |

### Error Handling Example

```python
from ai_mastery import (
    Client,
    APIError,
    AuthenticationError,
    RateLimitError,
    NotFoundError
)

client = Client(api_key="YOUR_API_KEY")

try:
    response = client.chat.completions.create(...)
except AuthenticationError:
    print("Invalid API key")
except NotFoundError:
    print("Model not found")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except APIError as e:
    print(f"API error: {e.message}")
```

---

## 🔍 Examples

### Basic Chat

```python
from ai_mastery import Client

client = Client(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### RAG System

```python
from ai_mastery import Client

client = Client(api_key="YOUR_API_KEY")

# Add documents
client.rag.documents.create(
    collection="knowledge_base",
    documents=[
        {"id": "doc1", "content": "Machine learning is..."},
        {"id": "doc2", "content": "Deep learning uses..."}
    ]
)

# Query
result = client.rag.query(
    collection="knowledge_base",
    query="What is machine learning?",
    top_k=3
)

print(f"Answer: {result.answer}")
print(f"Sources: {result.sources}")
```

### Fine-Tuning

```python
from ai_mastery import Client

client = Client(api_key="YOUR_API_KEY")

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    model="base-llm",
    training_data="sft_dataset.jsonl",
    hyperparameters={
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 16
    }
)

# Monitor progress
while job.status == "running":
    job = client.fine_tuning.jobs.get(job.id)
    print(f"Progress: {job.progress}%")
    time.sleep(60)

print(f"Fine-tuning complete: {job.fine_tuned_model}")
```

### Vector Search

```python
from ai_mastery import Client

client = Client(api_key="YOUR_API_KEY")

# Create collection
client.vector_db.collections.create(
    name="products",
    dimension=1536,
    metric="cosine"
)

# Upsert vectors
client.vector_db.upsert(
    collection="products",
    vectors=[
        {"id": "p1", "vector": [...], "metadata": {"name": "Product 1"}}
    ]
)

# Search
results = client.vector_db.search(
    collection="products",
    query_vector=[...],
    top_k=5
)
```

---

## 📈 Monitoring

### Request Logging

All API requests are logged with:

- Request ID
- Timestamp
- Endpoint
- Response time
- Status code

### Usage Metrics

Track your usage in the dashboard:

- Requests per day
- Token consumption
- Cost breakdown
- Error rates

### Webhooks

Configure webhooks for events:

```python
client.webhooks.create(
    url="https://your-server.com/webhook",
    events=[
        "fine_tuning.completed",
        "fine_tuning.failed",
        "usage.threshold"
    ]
)
```

---

## 🔒 Security Best Practices

1. **Keep API Keys Secret**
   - Never commit to version control
   - Use environment variables
   - Rotate keys regularly

2. **Use HTTPS**
   - All API requests must use HTTPS
   - Never use HTTP in production

3. **Implement Rate Limiting**
   - Add client-side rate limiting
   - Handle 429 errors gracefully

4. **Validate Input**
   - Sanitize user input
   - Validate request parameters

5. **Monitor Usage**
   - Set up usage alerts
   - Review logs regularly

---

## 📚 Additional Resources

- [SDK Documentation](sdk/) - Complete SDK reference
- [Code Examples](examples/) - Working code samples
- [Error Codes](error-codes.md) - Detailed error reference
- [Changelog](../reference/changelog.md) - API version history

---

**API Version:** 1.0.0  
**Last Updated:** March 28, 2026  
**Support:** api-support@ai-mastery-2026.com
