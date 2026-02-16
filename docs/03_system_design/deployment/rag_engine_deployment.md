# 🚀 Deployment Guide

> Complete guide for deploying RAG Engine Mini to production.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Environment Variables](#environment-variables)
7. [Database Setup](#database-setup)
8. [Monitoring & Observability](#monitoring--observability)

---

## Prerequisites

### Required Services

| Service | Version | Purpose |
|---------|---------|---------|
| Python | 3.11+ | Runtime |
| PostgreSQL | 16+ | Metadata, chunks, FTS |
| Redis | 7+ | Cache, Celery broker |
| Qdrant | Latest | Vector storage |

### Optional Services

| Service | Purpose |
|---------|---------|
| Ollama | Local LLM (no API costs) |
| Prometheus | Metrics collection |
| Grafana | Dashboards |

---

## Local Development

### 1. Clone and Install

```bash
cd sprints/rag_engine/rag-engine-mini

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

### 2. Start Infrastructure

```bash
# Start PostgreSQL, Redis, Qdrant
docker compose -f docker/docker-compose.yml up -d

# Verify services
docker compose -f docker/docker-compose.yml ps
```

### 3. Configure Environment

```bash
# Copy example env
cp .env.example .env

# Edit with your settings
# IMPORTANT: Set OPENAI_API_KEY
```

### 4. Run Migrations

```bash
# Apply database migrations
alembic upgrade head

# Seed demo user (optional)
python scripts/seed_user.py
```

### 5. Start Application

```bash
# Terminal 1: API Server
make run

# Terminal 2: Celery Worker
make celery-worker

# Terminal 3: Celery Beat (optional, for scheduled tasks)
make celery-beat
```

### 6. Verify Setup

```bash
# Health check
curl http://localhost:8000/health

# Test with API key
curl -X POST http://localhost:8000/api/v1/queries/ask-hybrid \
  -H "X-API-KEY: demo_api_key_12345678" \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello?"}'
```

---

## Docker Deployment

### Build Image

```bash
docker build -f docker/Dockerfile -t rag-engine-mini:latest .
```

### Run with Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: "3.9"

services:
  api:
    image: rag-engine-mini:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+psycopg://postgres:${DB_PASSWORD}@postgres:5432/rag
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_HOST=qdrant
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENV=prod
    depends_on:
      - postgres
      - redis
      - qdrant

  worker:
    image: rag-engine-mini:latest
    command: celery -A src.workers.celery_app worker -l info -Q indexing
    environment:
      - DATABASE_URL=postgresql+psycopg://postgres:${DB_PASSWORD}@postgres:5432/rag
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_HOST=qdrant
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - redis
      - qdrant

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: rag
    volumes:
      - pg_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  pg_data:
  qdrant_data:
```

```bash
# Deploy
docker compose -f docker-compose.prod.yml up -d
```

### Production Readiness Audit
Before deploying, run the containerized auditor to verify all connectivity and secrets:
```bash
make prod-check
```
This ensures the API can reach Postgres, Redis, and Qdrant within the Docker network.

---

## Kubernetes Deployment

### Helm Chart Structure

```
helm/rag-engine-mini/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment-api.yaml
│   ├── deployment-worker.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   └── ingress.yaml
```

### Key Configurations

```yaml
# values.yaml
replicaCount:
  api: 3
  worker: 2

image:
  repository: your-registry/rag-engine-mini
  tag: latest

resources:
  api:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"
  worker:
    requests:
      memory: "1Gi"  # Workers need more memory for models
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

---

## Cloud Deployment

### AWS

| Component | AWS Service |
|-----------|-------------|
| API | ECS Fargate / EKS |
| Worker | ECS Fargate / EKS |
| PostgreSQL | RDS PostgreSQL |
| Redis | ElastiCache Redis |
| Qdrant | EC2 / Qdrant Cloud |
| Storage | S3 |

### GCP

| Component | GCP Service |
|-----------|-------------|
| API | Cloud Run / GKE |
| Worker | Cloud Run / GKE |
| PostgreSQL | Cloud SQL |
| Redis | Memorystore |
| Qdrant | GCE / Qdrant Cloud |
| Storage | GCS |

### Azure

| Component | Azure Service |
|-----------|---------------|
| API | Container Apps / AKS |
| Worker | Container Apps / AKS |
| PostgreSQL | Azure Database for PostgreSQL |
| Redis | Azure Cache for Redis |
| Qdrant | ACI / Qdrant Cloud |
| Storage | Blob Storage |

---

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql+psycopg://user:pass@host:5432/rag` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `QDRANT_HOST` | Qdrant hostname | `localhost` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `dev` | Environment (dev/staging/prod) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `EMBEDDING_DIM` | `1536` | Embedding dimension |
| `CHUNK_MAX_TOKENS` | `512` | Max tokens per chunk |
| `RERANK_BACKEND` | `cross_encoder` | Reranking method |
| `USE_REAL_DB` | `false` | Use Postgres vs placeholder |

---

## Database Setup

### Initial Setup

```bash
# Create database
createdb rag

# Run all migrations
alembic upgrade head
```

### Migration Commands

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Upgrade to latest
alembic upgrade head

# Downgrade one version
alembic downgrade -1

# Show current version
alembic current

# Show migration history
alembic history
```

---

## Monitoring & Observability

### Structured Logging

Logs are JSON-formatted for easy parsing:

```json
{
  "timestamp": "2026-01-29T12:00:00Z",
  "level": "info",
  "event": "indexing_completed",
  "document_id": "abc-123",
  "chunks_count": 42,
  "duration_ms": 1500
}
```

### Metrics Endpoints

```bash
# Prometheus metrics
GET /metrics

# Health check
GET /health

# Readiness probe
GET /health/ready
```

### Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `rag_request_duration_seconds` | API latency | P95 > 5s |
| `rag_embedding_cache_hit_ratio` | Cache effectiveness | < 0.5 |
| `rag_llm_errors_total` | LLM failures | > 10/min |
| `rag_indexing_queue_size` | Pending documents | > 100 |

### Grafana Dashboard

Import the dashboard from `monitoring/grafana-dashboard.json` for:
- Request latency distribution
- Cache hit ratios
- Error rates by endpoint
- Worker queue depth

---

## Production Checklist

- [ ] PostgreSQL with replication
- [ ] Redis with persistence
- [ ] Qdrant with snapshots
- [ ] SSL/TLS on all endpoints
- [ ] API rate limiting
- [ ] Proper API key management
- [ ] Log aggregation (ELK/Loki)
- [ ] Metrics collection (Prometheus)
- [ ] Alerting configured
- [ ] Backup strategy
- [ ] Disaster recovery plan
