# 📘 Developer Guide

> Guide for developers extending and customizing RAG Engine Mini.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Adding New Adapters](#adding-new-adapters)
3. [Adding New Endpoints](#adding-new-endpoints)
4. [Adding New Use Cases](#adding-new-use-cases)
5. [Writing Tests](#writing-tests)
6. [Code Style](#code-style)
7. [Database Migrations](#database-migrations)
8. [Debugging Tips](#debugging-tips)

---

## Project Structure

```
rag-engine-mini/
├── src/
│   ├── core/           # Configuration, logging, DI container
│   ├── domain/         # Pure entities and errors (no deps)
│   ├── application/
│   │   ├── ports/      # Interfaces (protocols)
│   │   ├── services/   # Pure logic functions
│   │   └── use_cases/  # Business orchestration
│   ├── adapters/       # External implementations
│   │   ├── llm/        # LLM providers
│   │   ├── embeddings/ # Embedding providers
│   │   ├── vector/     # Vector stores
│   │   ├── rerank/     # Rerankers
│   │   ├── cache/      # Cache implementations
│   │   ├── filestore/  # File storage
│   │   ├── extraction/ # Text extractors
│   │   ├── queue/      # Task queues
│   │   └── persistence/# Database repos
│   ├── api/            # FastAPI routes
│   └── workers/        # Celery tasks
├── tests/
├── docs/
├── notebooks/
└── docker/
```

---

## Adding New Adapters

### Example: Adding Anthropic LLM

#### 1. Create the Port (if new interface needed)

Ports are already defined in `src/application/ports/llm.py`:

```python
class LLMPort(Protocol):
    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str: ...
```

#### 2. Create the Adapter

`src/adapters/llm/anthropic_llm.py`:

```python
"""
Anthropic Claude LLM Adapter
"""

import anthropic

from src.domain.errors import LLMError


class AnthropicLLM:
    """Anthropic Claude adapter implementing LLMPort."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
    
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            raise LLMError(f"Anthropic error: {e}") from e
```

#### 3. Register in Bootstrap

`src/core/bootstrap.py`:

```python
if settings.llm_backend == "anthropic":
    from src.adapters.llm.anthropic_llm import AnthropicLLM
    llm = AnthropicLLM(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
    )
```

#### 4. Add Configuration

`src/core/config.py`:

```python
# Anthropic
anthropic_api_key: str | None = Field(default=None)
anthropic_model: str = Field(default="claude-3-sonnet-20240229")
```

#### 5. Add Tests

`tests/unit/test_anthropic_llm.py`:

```python
def test_anthropic_generate():
    # Mock the client
    ...
```

---

## Adding New Endpoints

### Example: Chat Sessions API

#### 1. Create route file

`src/api/v1/routes_chat.py`:

```python
"""Chat session endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.v1.deps import get_tenant_id
from src.core.bootstrap import get_container

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class CreateSessionResponse(BaseModel):
    session_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    tenant_id: str = Depends(get_tenant_id),
) -> CreateSessionResponse:
    """Create a new chat session."""
    container = get_container()
    chat_repo = container["chat_repo"]
    
    session_id = chat_repo.create_session(
        tenant_id=TenantId(tenant_id),
    )
    
    return CreateSessionResponse(session_id=session_id)
```

#### 2. Register router

`src/main.py`:

```python
from src.api.v1.routes_chat import router as chat_router

app.include_router(chat_router)
```

---

## Adding New Use Cases

### Example: Delete Document Use Case

#### 1. Create use case

`src/application/use_cases/delete_document.py`:

```python
"""Delete Document Use Case."""

from dataclasses import dataclass

from src.application.ports.document_repo import DocumentRepoPort
from src.application.ports.chunk_repo import ChunkRepoPort
from src.application.ports.vector_store import VectorStorePort
from src.domain.entities import TenantId, DocumentId


@dataclass
class DeleteDocumentRequest:
    tenant_id: str
    document_id: str


class DeleteDocumentUseCase:
    """
    Deletes a document and all associated data.
    
    Flow:
    1. Delete chunks from vector store
    2. Delete chunk mappings
    3. Delete document record
    4. Delete stored file
    """
    
    def __init__(
        self,
        *,
        document_repo: DocumentRepoPort,
        chunk_repo: ChunkRepoPort,
        vector_store: VectorStorePort,
    ) -> None:
        self._doc_repo = document_repo
        self._chunk_repo = chunk_repo
        self._vector = vector_store
    
    def execute(self, request: DeleteDocumentRequest) -> bool:
        tenant = TenantId(request.tenant_id)
        doc_id = DocumentId(request.document_id)
        
        # Delete from vector store
        self._vector.delete_by_document(
            tenant_id=tenant,
            document_id=doc_id.value,
        )
        
        # Delete chunk mappings
        self._chunk_repo.delete_document_chunks(
            tenant_id=tenant,
            document_id=doc_id.value,
        )
        
        # Delete document
        return self._doc_repo.delete_document(
            tenant_id=tenant,
            document_id=doc_id,
        )
```

#### 2. Wire in bootstrap

Add to `src/core/bootstrap.py`.

---

## Writing Tests

### Unit Tests

Test pure logic without external dependencies:

```python
# tests/unit/test_chunking.py
def test_chunk_text_token_aware():
    chunks = chunk_text_token_aware("Hello world", ChunkSpec(max_tokens=100))
    assert len(chunks) == 1
```

### Integration Tests

Test with mocked or real services:

```python
# tests/integration/test_upload_flow.py
def test_upload_document(client, auth_headers):
    response = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers,
        files={"file": ("test.txt", b"Hello world", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
```

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/unit/test_chunking.py -v

# Specific test
pytest tests/unit/test_chunking.py::test_chunk_text_token_aware -v
```

---

## Code Style

### Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Fast linting
- **MyPy**: Type checking

### Commands

```bash
# Format code
make format

# Check linting
make lint

# Type check
mypy src/
```

### Conventions

```python
# Docstrings: Google style with bilingual support
def my_function(arg: str) -> bool:
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        arg: Description of arg
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When arg is invalid
        
    وصف قصير بالعربية
    """
    pass
```

---

## Database Migrations

### Create Migration

```bash
# Auto-generate from model changes
alembic revision --autogenerate -m "add_new_table"

# Manual migration
alembic revision -m "add_index"
```

### Migration Structure

```python
"""Add new table

Revision ID: 004_add_new_table
Revises: 003_add_chat_tables
Create Date: 2026-01-29

"""

from alembic import op
import sqlalchemy as sa

revision = "004_add_new_table"
down_revision = "003_add_chat_tables"


def upgrade() -> None:
    op.create_table(
        "new_table",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(256), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("new_table")
```

### Apply Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade one step
alembic upgrade +1

# Downgrade one step
alembic downgrade -1

# Show current version
alembic current
```

---

## Debugging Tips

### Enable Debug Logging

```env
DEBUG=true
LOG_LEVEL=DEBUG
```

### Interactive Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use ipdb for better experience
import ipdb; ipdb.set_trace()
```

### Inspect Container

```python
from src.core.bootstrap import get_container

container = get_container()
print(container.keys())
print(container["llm"])
```

### Test API Manually

```bash
# Health check
curl http://localhost:8000/health

# With verbose output
curl -v http://localhost:8000/api/v1/queries/ask-hybrid \
  -H "X-API-KEY: demo_api_key_12345678" \
  -H "Content-Type: application/json" \
  -d '{"question": "Test?"}'
```

### Monitor Celery

```bash
# Watch tasks
celery -A src.workers.celery_app flower

# Inspect active tasks
celery -A src.workers.celery_app inspect active
```
