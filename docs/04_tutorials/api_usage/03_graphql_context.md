# GraphQL Context Injection Guide

## Introduction

This guide explains how to properly configure GraphQL context in FastAPI applications, including dependency injection patterns and best practices.

## Learning Objectives

By the end of this guide, you will understand:
- **What is GraphQL context and why it's important**
- **How to inject dependencies into GraphQL resolvers**
- **FastAPI dependency injection patterns**
- **Context propagation strategies**
- **Common pitfalls with context management**
- **Type hints for GraphQL context**

---

## What is GraphQL Context?

GraphQL **context** is a shared object that's available to all resolvers during a single GraphQL operation. It's the primary way to provide:

- **Authentication/Authorization data** (user info, tenant ID)
- **Database connections** and repositories
- **External service clients** (LLM, vector store)
- **Configuration settings**
- **Request-scoped data** (request headers, client info)

### Lifecycle of GraphQL Context

```
Request arrives
    ↓
FastAPI creates request object
    ↓
get_graphql_context() is called
    ↓
Context is built with all dependencies
    ↓
GraphQL operation executes
    ↓
Each resolver receives info.context
    ↓
Operation completes
    ↓
Context is discarded (request-scoped)
```

---

## Setting Up GraphQL Context in FastAPI

### Basic Setup

The context is injected using the `context_getter` parameter when mounting the GraphQL router:

```python
from strawberry.fastapi import GraphQLRouter
from strawberry import schema

# Define context getter
async def get_graphql_context(request):
    """
    Build GraphQL context for each request.

    The context is created fresh for each GraphQL operation.
    """
    return {
        "request": request,
        "user": get_current_user(request),
        "tenant_id": get_tenant_id(request),
    }

# Mount GraphQL with context
graphql_app = GraphQLRouter(
    schema,
    context_getter=get_graphql_context,
)
app.mount("/graphql", graphql_app)
```

### Using Dependency Injection

In production applications, you typically inject services using a dependency injection container:

```python
from src.core.bootstrap import get_container

async def get_graphql_context(request):
    """Build GraphQL context using DI container."""
    container = get_container()

    return {
        "request": request,
        "doc_repo": container.get("document_repo"),
        "chat_repo": container.get("chat_repo"),
        "vector_store": container.get("vector_store"),
        "llm": container.get("llm"),
        # ... more services
    }
```

---

## Accessing Context in Resolvers

### Using `info.context`

Strawberry GraphQL provides the `info` parameter to all resolvers, which contains the context:

```python
@strawberry.type
class Query:
    @strawberry.field
    def documents(self, info, limit: int = 20) -> List[DocumentType]:
        """
        Query documents with pagination.

        Args:
            info: GraphQL execution info (contains context)
            limit: Max results to return
        """
        # Access context
        request = info.context.get("request")
        tenant_id = info.context.get("tenant_id")
        doc_repo = info.context.get("doc_repo")

        # Use injected services
        documents = doc_repo.list_documents(
            tenant_id=tenant_id,
            limit=limit,
        )

        return [DocumentType.from_entity(d) for d in documents]
```

### Type Hints for Context

To improve type safety, define a `Context` type and use type hints:

```python
from typing import Any, Dict, Optional
from strawberry.types import Info

# Define context type
class GraphQLContext:
    """Type-safe GraphQL context."""

    def __init__(
        self,
        request: Request,
        doc_repo: DocumentRepo,
        chat_repo: ChatRepo,
        vector_store: VectorStore,
        llm: LLMPort,
        # ... more services
    ):
        self.request = request
        self.doc_repo = doc_repo
        self.chat_repo = chat_repo
        self.vector_store = vector_store
        self.llm = llm

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphQLContext":
        """Create context from dict."""
        return cls(**data)

# Type hints in resolvers
def get_context(info: Info[None, None]) -> GraphQLContext:
    """Get typed context from info."""
    context_dict = info.context
    return GraphQLContext.from_dict(context_dict)

# Use in resolvers
@strawberry.type
class Query:
    @strawberry.field
    def documents(self, info: Info[None, None], limit: int = 20) -> List[DocumentType]:
        ctx = get_context(info)
        documents = ctx.doc_repo.list_documents(...)
        return documents
```

---

## Best Practices for GraphQL Context

### ✅ DO

1. **Keep Context Request-Scoped**
```python
# GOOD: Fresh context per request
async def get_graphql_context(request):
    container = get_container()
    return {
        "request": request,
        "doc_repo": container.get("document_repo"),
    }
```

2. **Use Lazy Loading for Heavy Resources**
```python
# GOOD: Lazy load expensive connections
class LazyDatabase:
    def __init__(self, get_db):
        self._get_db = get_db
        self._db = None

    @property
    def db(self):
        if self._db is None:
            self._db = self._get_db()
        return self._db

async def get_graphql_context(request):
    from src.core.bootstrap import get_db

    return {
        "db": LazyDatabase(get_db),
    }
```

3. **Validate Context Dependencies**
```python
# GOOD: Validate all services are available
async def get_graphql_context(request):
    container = get_container()

    context = {
        "request": request,
        "doc_repo": container.get("document_repo"),
        "vector_store": container.get("vector_store"),
    }

    # Validate critical dependencies
    missing = [k for k, v in context.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing services: {missing}")

    return context
```

4. **Provide Fallbacks for Optional Services**
```python
# GOOD: Provide default when service unavailable
async def get_graphql_context(request):
    container = get_container()

    llm = container.get("llm")
    if llm is None:
        # Fallback to mock for development
        from unittest.mock import Mock
        llm = Mock()

    return {
        "request": request,
        "llm": llm,
        # ... other services
    }
```

### ❌ DON'T

1. **Don't Store Global State in Context**
```python
# BAD: Global state leaks between requests
_global_cache = {}

async def get_graphql_context(request):
    return {
        "cache": _global_cache,  # Shared across all requests!
    }

# GOOD: Request-scoped state
async def get_graphql_context(request):
    from src.application.services.multi_layer_cache import CacheService

    cache = CacheService(
        memory_cache={},
        redis_cache=get_redis(),
        db_cache=get_db(),
    )

    return {
        "cache": cache,  # Fresh per request
    }
```

2. **Don't Create Heavy Objects Unconditionally**
```python
# BAD: Always creates database connection
async def get_graphql_context(request):
    return {
        "db": create_database_connection(),  # Created even if not used!
    }

# GOOD: Lazy load or provide factory
async def get_graphql_context(request):
    return {
        "db_factory": lambda: create_database_connection(),
    }
```

3. **Don't Assume All Services Exist**
```python
# BAD: Crashes if service missing
async def get_graphql_context(request):
    container = get_container()

    return {
        "webhook_service": container["webhook_service"],  # KeyError!
    }

# GOOD: Use .get() with fallback
async def get_graphql_context(request):
    container = get_container()

    webhook_service = container.get("webhook_service")
    if webhook_service is None:
        webhook_service = NoOpWebhookService()

    return {
        "webhook_service": webhook_service,
    }
```

---

## Common Pitfalls

### Pitfall 1: Circular Imports

**Problem:**
```python
# src/api/v1/graphql.py
from src.core.bootstrap import get_container  # Import bootstrap
from src.api.v1.graphql import schema  # Import graphql

# src/main.py
from src.api.v1.graphql import schema  # CIRCULAR!
```

**Solution: Import in function**
```python
# src/main.py
def get_graphql_context(request):
    # Import inside function to avoid circular import
    from src.core.bootstrap import get_container

    container = get_container()
    return {...}
```

### Pitfall 2: Context Not Available

**Problem:**
```python
# Resolvers that don't receive info parameter
@strawberry.type
class Query:
    @strawberry.field
    def documents(self, limit: int = 20) -> List[DocumentType]:
        # ERROR: No info parameter!
        doc_repo = info.context.get("doc_repo")  # NameError!
```

**Solution: Always include info parameter**
```python
# Good: Include info parameter
@strawberry.type
class Query:
    @strawberry.field
    def documents(self, info, limit: int = 20) -> List[DocumentType]:
        doc_repo = info.context.get("doc_repo")
        return doc_repo.list_documents(limit=limit)
```

### Pitfall 3: Missing Services

**Problem:**
```python
# GraphQL context missing webhook service
async def get_graphql_context(request):
    container = get_container()
    return {
        "doc_repo": container.get("document_repo"),
        # Missing webhook_repo!
    }

# Resolver fails
@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_webhook(self, info, url: str) -> WebhookType:
        webhook_repo = info.context.get("webhook_repo")  # None!
        return webhook_repo.create(url)  # AttributeError!
```

**Solution: Add all needed services**
```python
async def get_graphql_context(request):
    container = get_container()
    return {
        "doc_repo": container.get("document_repo"),
        "webhook_repo": container.get("webhook_repo"),  # Added!
        # ... other services
    }
```

---

## Complete Example: RAG Engine GraphQL Context

```python
"""
GraphQL Context Setup for RAG Engine
================================
Comprehensive context injection with all dependencies.
"""

from typing import Dict, Any
from fastapi import Request

from src.core.bootstrap import get_container
from src.core.config import settings
from src.api.v1.deps import get_tenant_id
from src.application.services.event_manager import get_event_manager

async def get_graphql_context(request: Request) -> Dict[str, Any]:
    """
    Build GraphQL context with all required dependencies.

    Dependencies injected:
    - Request handling: request, tenant_id
    - Database repositories: doc_repo, chat_repo, query_history_repo, db_repo
    - External services: vector_store, llm, file_storage, redis_client
    - Use cases: search_service, ask_hybrid_use_case
    - Event handling: event_manager

    Args:
        request: FastAPI Request object

    Returns:
        Dictionary with all context dependencies

    Raises:
        RuntimeError: If critical services are missing
    """
    # Get dependency container
    container = get_container()

    # Get tenant ID from request
    tenant_id = get_tenant_id(request)

    # Build context
    context = {
        # Request handling
        "request": request,
        "tenant_id": tenant_id,

        # Database repositories
        "doc_repo": container.get("document_repo"),
        "chat_repo": container.get("chat_repo"),
        "query_history_repo": container.get("query_history_repo"),
        "webhook_repo": container.get("webhook_repo"),
        "chunk_repo": container.get("chunk_repo"),
        "db_repo": container.get("db_repo"),

        # External services
        "vector_store": container.get("vector_store"),
        "llm": container.get("llm"),
        "file_storage": container.get("file_storage"),
        "redis_client": container.get("redis_client"),

        # Caching
        "cache": container.get("cache"),

        # Use cases (application services)
        "search_service": container.get("search_documents_use_case"),
        "ask_hybrid_use_case": container.get("ask_hybrid_use_case"),
        "ask_question_use_case": container.get("ask_question_use_case"),

        # Event handling
        "event_manager": get_event_manager(),
    }

    # Validate critical dependencies
    critical_services = ["doc_repo", "vector_store", "llm"]
    missing = [s for s in critical_services if context.get(s) is None]

    if missing:
        raise RuntimeError(
            f"Critical GraphQL context services missing: {missing}. "
            "Check bootstrap.py configuration."
        )

    # Log context build (dev only)
    if settings.debug:
        from src.core.logging import get_logger
        log = get_logger(__name__)
        log.info(
            "graphql_context_built",
            tenant_id=tenant_id,
            services=list(context.keys()),
        )

    return context
```

---

## Testing GraphQL Context

### Unit Tests

```python
import pytest
from unittest.mock import Mock

def test_graphql_context_includes_required_services():
    """Test that all required services are in context."""
    from src.main import get_graphql_context

    # Create mock request
    request = Mock()
    request.headers = {"X-Tenant-ID": "test-tenant"}

    # Get context
    import asyncio
    context = asyncio.run(get_graphql_context(request))

    # Verify critical services
    assert "doc_repo" in context
    assert "vector_store" in context
    assert "llm" in context
    assert "request" in context
    assert "tenant_id" in context

def test_graphql_context_validation():
    """Test that missing critical services raise error."""
    from unittest.mock import patch

    with patch('src.core.bootstrap.get_container') as mock_container:
        # Mock container without doc_repo
        mock_container.return_value.get.side_effect = lambda k: None

        request = Mock()
        request.headers = {"X-Tenant-ID": "test-tenant"}

        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(get_graphql_context(request))

        assert "Missing" in str(exc_info.value)
```

---

## Summary

### Key Takeaways:

1. **GraphQL context** is request-scoped and shared across resolvers
2. **Dependency injection** ensures services are available where needed
3. **Validate dependencies** to catch configuration issues early
4. **Use lazy loading** for expensive or optional services
5. **Type hints** improve type safety and IDE support
6. **Avoid circular imports** by importing in functions

### Best Practices:

- ✅ Build fresh context per request
- ✅ Validate all critical dependencies
- ✅ Provide fallbacks for optional services
- ✅ Import in functions to avoid circular dependencies
- ✅ Use type hints for context
- ✅ Log context building in development

### Anti-Patterns:

- ❌ Don't share global state in context
- ❌ Don't create heavy objects unconditionally
- ❌ Don't assume services exist without validation
- ❌ Don't forget info parameter in resolvers

---

## Additional Resources

- **Strawberry Documentation**: https://strawberry.rocks/docs/guides/contexts
- **FastAPI Dependencies**: https://fastapi.tiangolo.com/tutorial/dependencies/
- **Dependency Injection Patterns**: Martin Fowler's articles on DI
- **GraphQL Best Practices**: graphql.org/learn/best-practices
