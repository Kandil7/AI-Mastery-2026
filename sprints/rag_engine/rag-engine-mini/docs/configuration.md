# 🔧 Configuration Guide

> Complete guide to configuring RAG Engine Mini.

---

## Overview

All configuration is done via environment variables, loaded by Pydantic Settings.
Create a `.env` file or set environment variables directly.

---

## Quick Start

```bash
cp .env.example .env
# Edit .env with your settings
```

---

## Configuration Sections

### Application

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_NAME` | str | `rag-engine-mini` | Application name |
| `ENV` | enum | `dev` | Environment: `dev`, `staging`, `prod` |
| `DEBUG` | bool | `false` | Enable debug mode |
| `LOG_LEVEL` | enum | `INFO` | Logging: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

**Example:**
```env
ENV=prod
DEBUG=false
LOG_LEVEL=INFO
```

---

### Security

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_KEY_HEADER` | str | `X-API-KEY` | Header name for API key |

**Example:**
```env
API_KEY_HEADER=X-API-KEY
```

---

### Database (PostgreSQL)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATABASE_URL` | str | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` | Connection URL |
| `DB_POOL_SIZE` | int | `5` | Connection pool size (1-20) |
| `DB_MAX_OVERFLOW` | int | `10` | Max overflow connections (0-50) |

**Example:**
```env
DATABASE_URL=postgresql+psycopg://user:password@db.example.com:5432/rag
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
```

**Connection URL Format:**
```
postgresql+psycopg://<user>:<password>@<host>:<port>/<database>
```

---

### Redis

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_URL` | str | `redis://localhost:6379/0` | Redis URL for caching |
| `CELERY_BROKER_URL` | str | `redis://localhost:6379/1` | Celery broker URL |
| `CELERY_RESULT_BACKEND` | str | `redis://localhost:6379/2` | Celery result backend |
| `EMBEDDING_CACHE_TTL` | int | `604800` | Cache TTL in seconds (7 days) |

**Example:**
```env
REDIS_URL=redis://:password@redis.example.com:6379/0
CELERY_BROKER_URL=redis://:password@redis.example.com:6379/1
EMBEDDING_CACHE_TTL=604800
```

---

### Vector Store (Qdrant)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `QDRANT_HOST` | str | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | int | `6333` | Qdrant HTTP port |
| `QDRANT_COLLECTION` | str | `chunks` | Collection name |
| `QDRANT_API_KEY` | str | null | API key (for Qdrant Cloud) |
| `EMBEDDING_DIM` | int | `1536` | Vector dimension (must match model) |

**Example:**
```env
# Local Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Qdrant Cloud
QDRANT_HOST=abc123.us-east-1.aws.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

**Embedding Dimensions:**
| Model | Dimension |
|-------|-----------|
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
| `all-MiniLM-L6-v2` | 384 |
| `all-mpnet-base-v2` | 768 |

---

### Embeddings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EMBEDDINGS_BACKEND` | enum | `openai` | Backend: `openai`, `local` |
| `OPENAI_EMBED_MODEL` | str | `text-embedding-3-small` | OpenAI model |
| `LOCAL_EMBED_MODEL` | str | `all-MiniLM-L6-v2` | Local model |
| `LOCAL_EMBED_DEVICE` | enum | `cpu` | Device: `cpu`, `cuda` |

**Example:**
```env
# OpenAI Embeddings
EMBEDDINGS_BACKEND=openai
OPENAI_EMBED_MODEL=text-embedding-3-small

# Local Embeddings (no API costs)
EMBEDDINGS_BACKEND=local
LOCAL_EMBED_MODEL=all-MiniLM-L6-v2
LOCAL_EMBED_DEVICE=cuda
```

---

### LLM Provider

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LLM_BACKEND` | enum | `openai` | Backend: `openai`, `ollama`, `gemini`, `huggingface` |

#### OpenAI

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENAI_API_KEY` | str | null | **Required** for OpenAI |
| `OPENAI_CHAT_MODEL` | str | `gpt-4o-mini` | Chat model |
| `OPENAI_MAX_TOKENS` | int | `700` | Max response tokens |
| `OPENAI_TEMPERATURE` | float | `0.2` | Temperature (0-2) |

#### Ollama

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OLLAMA_BASE_URL` | str | `http://localhost:11434` | Ollama server |
| `OLLAMA_CHAT_MODEL` | str | `llama3.1` | Chat model |
| `OLLAMA_EMBED_MODEL` | str | `nomic-embed-text` | Embedding model |

**Example:**
```env
# OpenAI
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2

# Ollama (local, no API costs)
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1
```

#### Gemini

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GEMINI_API_KEY` | str | null | **Required** for Gemini |
| `GEMINI_MODEL` | str | `gemini-1.5-flash` | Chat model |

**Example:**
```env
LLM_BACKEND=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash
```

#### Hugging Face

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HF_API_KEY` | str | null | **Required** for HF API |
| `HF_MODEL` | str | `mistralai/Mistral-7B-Instruct-v0.2` | Model ID |
| `HF_USE_INFERENCE_API` | bool | `true` | Use Hugging Face Inference API |

**Example:**
```env
LLM_BACKEND=huggingface
HF_API_KEY=your_key_here
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

---

### Reranking

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RERANK_BACKEND` | enum | `cross_encoder` | Backend: `cross_encoder`, `llm`, `none` |
| `CROSS_ENCODER_MODEL` | str | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Model |
| `CROSS_ENCODER_DEVICE` | enum | `cpu` | Device: `cpu`, `cuda` |
| `RERANK_TOP_N` | int | `8` | Results after reranking |

**Example:**
```env
# Cross-Encoder (recommended)
RERANK_BACKEND=cross_encoder
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CROSS_ENCODER_DEVICE=cuda

# LLM-based (slower, more expensive)
RERANK_BACKEND=llm

# Disable reranking
RERANK_BACKEND=none
```

---

### Retrieval

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEFAULT_K_VECTOR` | int | `30` | Vector search top-K |
| `DEFAULT_K_KEYWORD` | int | `30` | Keyword search top-K |
| `DEFAULT_FUSED_LIMIT` | int | `40` | Max after fusion |
| `RRF_K` | int | `60` | RRF constant |

**Example:**
```env
DEFAULT_K_VECTOR=30
DEFAULT_K_KEYWORD=30
DEFAULT_FUSED_LIMIT=40
RRF_K=60
```

---

### Chunking

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CHUNK_MAX_TOKENS` | int | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | int | `50` | Overlap between chunks |
| `CHUNK_ENCODING` | str | `cl100k_base` | tiktoken encoding |

**Example:**
```env
CHUNK_MAX_TOKENS=512
CHUNK_OVERLAP_TOKENS=50
```

---

### File Upload

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `UPLOAD_DIR` | str | `./uploads` | Upload directory |
| `MAX_UPLOAD_MB` | int | `20` | Max file size (MB) |
| `ALLOWED_EXTENSIONS` | str | `pdf,docx,txt` | Allowed extensions |

**Example:**
```env
UPLOAD_DIR=/data/uploads
MAX_UPLOAD_MB=50
ALLOWED_EXTENSIONS=pdf,docx,txt,md
```

---

### Workers (Celery)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CELERY_WORKER_CONCURRENCY` | int | `4` | Worker processes |
| `CELERY_TASK_TIME_LIMIT` | int | `600` | Task timeout (seconds) |

**Example:**
```env
CELERY_WORKER_CONCURRENCY=8
CELERY_TASK_TIME_LIMIT=900
```

---

### Observability

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_METRICS` | bool | `true` | Enable Prometheus metrics |
| `METRICS_PATH` | str | `/metrics` | Metrics endpoint |
| `REQUEST_ID_HEADER` | str | `X-Request-ID` | Request ID header |

---

## Configuration Profiles

### Development
```env
ENV=dev
DEBUG=true
LOG_LEVEL=DEBUG
LLM_BACKEND=ollama
EMBEDDINGS_BACKEND=local
RERANK_BACKEND=none
```

### Staging
```env
ENV=staging
DEBUG=false
LOG_LEVEL=INFO
LLM_BACKEND=openai
EMBEDDINGS_BACKEND=openai
RERANK_BACKEND=cross_encoder
```

### Production
```env
ENV=prod
DEBUG=false
LOG_LEVEL=WARNING
LLM_BACKEND=openai
EMBEDDINGS_BACKEND=openai
RERANK_BACKEND=cross_encoder
EMBEDDING_CACHE_TTL=2592000  # 30 days
```

---

## Validation

Settings are validated at startup. Invalid values will cause the application to fail fast with clear error messages.

```python
from src.core.config import settings

# Access any setting
print(settings.openai_chat_model)
print(settings.chunk_max_tokens)
print(settings.allowed_extensions_list)  # Parsed list
```
