# 🔍 Troubleshooting Guide

> Common issues and solutions for RAG Engine Mini.

---

## Quick Diagnostics

```bash
# Check all services
docker compose -f docker/docker-compose.yml ps

# Check API health
curl http://localhost:8000/health

# Check API readiness
curl http://localhost:8000/health/ready

# Check logs
docker compose -f docker/docker-compose.yml logs -f
```

---

## Common Issues

### 1. Connection Errors

#### PostgreSQL Connection Refused

**Symptom:**
```
sqlalchemy.exc.OperationalError: connection refused
```

**Solutions:**
1. Check PostgreSQL is running:
   ```bash
   docker compose ps postgres
   ```
2. Verify DATABASE_URL in .env
3. Check port 5432 is not blocked
4. Ensure database exists:
   ```bash
   docker exec -it rag_postgres psql -U postgres -c "\l"
   ```

---

#### Redis Connection Error

**Symptom:**
```
redis.exceptions.ConnectionError: Error connecting to redis://localhost:6379
```

**Solutions:**
1. Check Redis is running:
   ```bash
   docker compose ps redis
   redis-cli ping
   ```
2. Verify REDIS_URL in .env
3. Check port 6379 is available

---

#### Qdrant Connection Error

**Symptom:**
```
qdrant_client.http.exceptions.ResponseHandlingException
```

**Solutions:**
1. Check Qdrant is running:
   ```bash
   docker compose ps qdrant
   curl http://localhost:6333/health
   ```
2. Verify QDRANT_HOST and QDRANT_PORT
3. Ensure collection exists (auto-created on first use)

---

### 2. API Errors

#### 401 Unauthorized

**Symptom:**
```json
{"detail": "Invalid or missing API key"}
```

**Solutions:**
1. Include `X-API-KEY` header:
   ```bash
   curl -H "X-API-KEY: your_key" http://localhost:8000/api/v1/...
   ```
2. Create a user with API key in database
3. For development, use any 8+ char string as API key

---

#### 413 File Too Large

**Symptom:**
```json
{"detail": "File too large: 25.5MB exceeds limit of 20.0MB"}
```

**Solutions:**
1. Increase `MAX_UPLOAD_MB` in .env
2. Compress the file
3. Split into smaller documents

---

#### 415 Unsupported Media Type

**Symptom:**
```json
{"detail": "Unsupported file type: .xlsx. Allowed: pdf, docx, txt"}
```

**Solutions:**
1. Convert to supported format (PDF, DOCX, TXT)
2. Add new extractor for the format
3. Update `ALLOWED_EXTENSIONS`

---

### 3. Indexing Issues

#### Document Stuck in "queued"

**Symptom:**
Document status remains "queued" indefinitely.

**Solutions:**
1. Check Celery worker is running:
   ```bash
   celery -A src.workers.celery_app inspect active
   ```
2. Start worker:
   ```bash
   make celery-worker
   ```
3. Check Redis broker connection
4. View worker logs for errors

---

#### Document Stuck in "processing"

**Symptom:**
Document never reaches "indexed" status.

**Solutions:**
1. Check worker logs for errors
2. Verify OpenAI API key is valid
3. Check for rate limiting
4. Increase task timeout:
   ```env
   CELERY_TASK_TIME_LIMIT=1200
   ```

---

#### Indexing Failed

**Symptom:**
```json
{"status": "failed", "error": "..."}
```

**Solutions:**
1. Check the `error` field for details
2. Common causes:
   - Invalid/empty document
   - Text extraction failure
   - API rate limiting
   - Network issues
3. Retry by re-uploading

---

### 4. Search/Query Issues

#### No Results Returned

**Symptom:**
Empty `sources` array, generic answer.

**Solutions:**
1. Verify documents are indexed (status: "indexed")
2. Check embeddings were generated correctly
3. Try rephrasing the question
4. Lower the retrieval threshold
5. Check Qdrant collection has data:
   ```bash
   curl http://localhost:6333/collections/chunks
   ```

---

#### Poor Quality Answers

**Symptom:**
Answers are irrelevant or hallucinated.

**Solutions:**
1. Improve retrieval:
   - Increase `k_vec` and `k_kw`
   - Enable reranking: `RERANK_BACKEND=cross_encoder`
2. Improve chunking:
   - Reduce `CHUNK_MAX_TOKENS` for more granular chunks
   - Increase overlap
3. Improve prompt (see prompt-engineering.md)
4. Use better LLM model

---

#### Slow Response Times

**Symptom:**
API takes >10 seconds to respond.

**Solutions:**
1. Enable embedding cache:
   ```env
   REDIS_URL=redis://localhost:6379/0
   ```
2. Reduce retrieval size:
   ```json
   {"k_vec": 20, "k_kw": 20, "rerank_top_n": 5}
   ```
3. Use faster models:
   - `gpt-4o-mini` instead of `gpt-4o`
   - Smaller Cross-Encoder model
4. Add GPU for local models:
   ```env
   CROSS_ENCODER_DEVICE=cuda
   ```

---

### 5. LLM Errors

#### OpenAI Rate Limit

**Symptom:**
```
LLMRateLimitError: Rate limit exceeded
```

**Solutions:**
1. Implement retry with exponential backoff (already in adapter)
2. Reduce request frequency
3. Upgrade OpenAI plan
4. Switch to Ollama for development

---

#### OpenAI API Key Invalid

**Symptom:**
```
LLMError: Invalid API key
```

**Solutions:**
1. Verify `OPENAI_API_KEY` is set correctly
2. Check API key is active at platform.openai.com
3. Ensure no extra whitespace in .env

---

#### Ollama Not Responding

**Symptom:**
```
LLMError: Ollama connection error
```

**Solutions:**
1. Check Ollama is running:
   ```bash
   ollama list
   ```
2. Start Ollama server:
   ```bash
   ollama serve
   ```
3. Pull required model:
   ```bash
   ollama pull llama3.1
   ```
4. Verify `OLLAMA_BASE_URL`

---

### 6. Memory Issues

#### Worker Out of Memory

**Symptom:**
```
MemoryError or OOMKilled
```

**Solutions:**
1. Reduce concurrent tasks:
   ```env
   CELERY_WORKER_CONCURRENCY=2
   ```
2. Process smaller batches
3. Use smaller embedding model
4. Increase container memory limits

---

#### Cross-Encoder OOM

**Symptom:**
Crash when loading Cross-Encoder model.

**Solutions:**
1. Use smaller model:
   ```env
   CROSS_ENCODER_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2
   ```
2. Disable reranking:
   ```env
   RERANK_BACKEND=none
   ```
3. Increase available memory

---

## Logging

### Enable Debug Logging

```env
LOG_LEVEL=DEBUG
DEBUG=true
```

### View Structured Logs

```bash
# API logs
docker compose logs -f api

# Worker logs
docker compose logs -f worker

# All logs
docker compose logs -f
```

### Log Locations

| Component | Location |
|-----------|----------|
| API | stdout (Docker logs) |
| Worker | stdout (Docker logs) |
| PostgreSQL | `/var/log/postgresql/` |
| Redis | stdout |
| Qdrant | `/qdrant/storage/logs/` |

---

## Health Checks

### API Health

```bash
curl http://localhost:8000/health
```

### Database Health

```bash
docker exec -it rag_postgres pg_isready -U postgres
```

### Redis Health

```bash
redis-cli -h localhost ping
```

### Qdrant Health

```bash
curl http://localhost:6333/health
```

### Celery Health

```bash
celery -A src.workers.celery_app inspect ping
```

---

## Getting Help

1. Check this troubleshooting guide
2. Search existing issues
3. Check logs for detailed error messages
4. Create a minimal reproduction case
5. Open an issue with:
   - Python version
   - OS
   - Error message
   - Steps to reproduce
