# GraphQL Query Resolvers - Complete Implementation Guide
# ==================================================

## 📚 Learning Objectives

By the end of this guide, you will understand:
- GraphQL query resolvers and their role in GraphQL API design
- How to implement database-backed GraphQL resolvers with Strawberry
- Best practices for GraphQL resolver design
- Performance optimization techniques for GraphQL queries
- How to handle authentication and authorization in GraphQL
- Error handling and validation patterns

---
## 📌 Table of Contents

1. [Introduction](#1-introduction)
2. [GraphQL Query Resolvers Explained](#2-graphql-query-resolvers-explained)
3. [Project Structure & Architecture](#3-project-structure--architecture)
4. [Document Query Resolvers](#4-document-query-resolvers)
5. [Chat Session Query Resolvers](#5-chat-session-query-resolvers)
6. [Query History Resolvers](#6-query-history-resolvers)
7. [Performance Optimization](#7-performance-optimization)
8. [Best Practices](#8-best-practices)
9. [Common Pitfalls](#9-common-pitfalls)
10. [Quiz](#10-quiz)
11. [References](#11-references)

---
## 1. Introduction

### 1.1 What is GraphQL?

GraphQL is a query language for APIs that allows clients to request exactly the data they need, no more and no less. Unlike REST APIs with multiple endpoints for different resources, GraphQL uses a single endpoint with a flexible schema.

**Key Benefits:**
- **Precise Data Fetching**: Get only the fields you request
- **Single Endpoint**: All operations through one URL
- **Strongly Typed Schema**: Self-documenting API
- **Real-time Updates**: Built-in subscription support
- **Introspection**: Query the schema itself

### 1.2 What are Query Resolvers?

Query resolvers are functions that **resolve** (fetch) data for each field in your GraphQL schema. When a client executes a query, GraphQL calls the resolver for each field in parallel and assembles the response.

**Resolver Function Signature:**
```python
@strawberry.field
def field_name(self, parent, info, **kwargs) -> ReturnType:
    """
    Resolve this field.

    Args:
        self: The parent object (or None for root Query)
        parent: The parent object's value (for nested resolvers)
        info: GraphQL execution context (contains request info)
        **kwargs: Field arguments from the query

    Returns:
        The resolved value for this field
    """
    pass
```

### 1.3 Why Implement Proper Resolvers?

**Without Proper Resolvers:**
```python
# BAD: Returns empty or dummy data
@strawberry.field
def documents(self) -> List[DocumentType]:
    return []  # No data!
```

**With Proper Resolvers:**
```python
# GOOD: Queries actual database
@strawberry.field
def documents(self, info, limit: int = 20, offset: int = 0) -> List[DocumentType]:
    tenant_id = get_tenant_id(info.context["request"])
    docs = doc_repo.list_documents(tenant_id=tenant_id, limit=limit, offset=offset)
    return [DocumentType(**doc) for doc in docs]
```

---
## 2. GraphQL Query Resolvers Explained

### 2.1 Resolver Execution Flow

When a GraphQL query is executed:

```
Client Query
    │
    ▼
GraphQL Server (Parse & Validate)
    │
    ▼
Query Execution Engine
    │
    ├─→ documents() resolver
    │   ├─→ Fetch from database
    │   └─→ Return List[DocumentType]
    │
    ├─→ document(id) resolver
    │   ├─→ Fetch single document
    │   └─→ Return DocumentType
    │
    └─→ chat_sessions() resolver
        ├─→ Fetch sessions from database
        └─→ Return List[ChatSessionType]
    │
    ▼
Assemble Response
    │
    ▼
Send JSON to Client
```

### 2.2 Resolver Context

The `info` parameter provides access to:
- **Request Information**: HTTP headers, cookies
- **Schema Details**: Field definitions, type info
- **Custom Context**: Injected dependencies (DB, services)

**Example:**
```python
@strawberry.field
def documents(self, info, limit: int = 20) -> List[DocumentType]:
    # Access request context
    request = info.context["request"]

    # Extract tenant ID from headers
    tenant_id = request.headers.get("X-Tenant-ID")

    # Access injected services
    doc_repo = info.context["doc_repo"]

    # Query database
    docs = doc_repo.list_documents(tenant_id=tenant_id, limit=limit)
    return [DocumentType(**doc) for doc in docs]
```

### 2.3 Async Resolvers

Strawberry supports both sync and async resolvers:

```python
# Sync resolver (runs in thread pool)
@strawberry.field
def documents_sync(self) -> List[DocumentType]:
    return doc_repo.list_documents(...)

# Async resolver (coroutine)
@strawberry.field
async def documents_async(self) -> List[DocumentType]:
    return await doc_repo.list_documents_async(...)
```

**When to Use Async:**
- Database queries (async drivers like asyncpg)
- External API calls
- File I/O operations
- CPU-bound tasks (use CPU worker pool)

---
## 3. Project Structure & Architecture

### 3.1 Current GraphQL Setup

```
src/api/v1/
├── graphql.py              # GraphQL schema and resolvers (THIS FILE)
├── deps.py                # Dependency injection (get_tenant_id)
└── routes_health.py       # Health check endpoints

src/application/
├── ports/
│   ├── document_repo.py    # Document repository interface
│   ├── chat_repo.py       # Chat repository interface
│   └── query_history_repo.py # Query history interface
└── use_cases/
    └── ask_question_hybrid.py # RAG query use case
```

### 3.2 GraphQL Schema Structure

```python
# Type Definitions
@strawberry.type
class DocumentType:
    id: strawberry.ID
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    created_at: datetime
    updated_at: Optional[datetime]

@strawberry.type
class ChatSessionType:
    id: strawberry.ID
    title: Optional[str]
    created_at: datetime

@strawberry.type
class QueryHistoryItemType:
    question: str
    answer: str
    sources: List[str]
    timestamp: datetime

# Root Query Type
@strawberry.type
class Query:
    documents(limit: int, offset: int) -> List[DocumentType]
    document(document_id: ID) -> Optional[DocumentType]
    search_documents(query: str, k: int) -> SearchResultType
    chat_sessions(limit: int) -> List[ChatSessionType]
    query_history(limit: int) -> List[QueryHistoryItemType]

# Root Mutation Type
@strawberry.type
class Mutation:
    ask_question(question: str, k: int) -> AnswerType
    upload_document(file: Upload, filename: str) -> DocumentType
    create_chat_session(title: Optional[str]) -> ChatSessionType

# Root Subscription Type
@strawberry.type
class Subscription:
    document_indexed(document_id: ID) -> DocumentType

# Complete Schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)
```

### 3.3 Integration with FastAPI

GraphQL is mounted on FastAPI as a separate route:

```python
from strawberry.fastapi import GraphQLRouter

# Create GraphQL app
graphql_app = GraphQLRouter(schema)

# Mount on FastAPI
app.mount("/graphql", graphql_app)
```

**Access Points:**
- GraphQL Playground: `http://localhost:8000/graphql` (browser)
- GraphQL Endpoint: `POST http://localhost:8000/graphql` (API)
- GraphQL Schema Introspection: `GET http://localhost:8000/graphql`

---
## 4. Document Query Resolvers

### 4.1 List Documents Resolver

**Purpose:** Fetch paginated list of documents for a tenant.

**GraphQL Query:**
```graphql
query ListDocuments($limit: Int, $offset: Int, $status: DocumentStatus) {
  documents(limit: $limit, offset: $offset, status: $status) {
    id
    filename
    content_type
    sizeBytes
    status
    createdAt
    updatedAt
  }
}
```

**Implementation:**
```python
from typing import List, Optional
import strawberry
from datetime import datetime

# Import dependencies
from src.api.v1.deps import get_tenant_id
from src.application.ports.document_repo import DocumentRepoPort

@strawberry.type
class Query:
    @strawberry.field
    def documents(
        self,
        info,
        limit: int = 20,
        offset: int = 0,
        status: Optional[DocumentStatus] = None,
    ) -> List[DocumentType]:
        """
        Query documents with pagination and filtering.

        Args:
            info: GraphQL execution context
            limit: Max results (default: 20, max: 100)
            offset: Pagination offset (default: 0)
            status: Filter by status (optional)

        Returns:
            List of documents

        الاستعلام عن المستندات مع الترحيل والتصفية
        """
        # Validate inputs
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit > 100:
            limit = 100  # Enforce maximum
        if offset < 0:
            raise ValueError("offset must be non-negative")

        # Get tenant ID from request context
        request = info.context["request"]
        tenant_id = get_tenant_id(request)

        # Get document repository from context
        doc_repo: DocumentRepoPort = info.context["doc_repo"]

        # Query database
        from src.application.ports.document_repo import DocumentStatus as DocStatus
        documents = doc_repo.list_documents(
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
        )

        # Convert domain entities to GraphQL types
        result = []
        for doc in documents:
            # Filter by status if provided
            if status and doc.status.value != status.value:
                continue

            result.append(
                DocumentType(
                    id=doc.document_id,
                    filename=doc.filename,
                    content_type=doc.content_type,
                    size_bytes=doc.size_bytes,
                    status=DocumentStatus(doc.status.value),
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                )
            )

        return result
```

**Key Implementation Details:**

1. **Input Validation:**
   - Enforce maximum `limit` (prevents resource exhaustion)
   - Ensure non-negative `offset`
   - Return 400 Bad Request for invalid inputs

2. **Tenant Isolation:**
   - Extract `tenant_id` from request headers
   - Pass to all repository queries
   - Never return documents from other tenants

3. **Status Filtering:**
   - Filter in-memory after query (for simplicity)
   - Production: Add WHERE clause to SQL query

4. **Type Conversion:**
   - Convert domain `DocumentStatus` to GraphQL `DocumentStatus`
   - Map datetime objects to GraphQL types

**Response Example:**
```json
{
  "data": {
    "documents": [
      {
        "id": "doc-123",
        "filename": "research-paper.pdf",
        "contentType": "application/pdf",
        "sizeBytes": 1048576,
        "status": "INDEXED",
        "createdAt": "2026-01-31T12:00:00Z",
        "updatedAt": "2026-01-31T12:05:00Z"
      }
    ]
  }
}
```

### 4.2 Get Single Document Resolver

**Purpose:** Fetch a single document by ID.

**GraphQL Query:**
```graphql
query GetDocument($documentId: ID!) {
  document(documentId: $documentId) {
    id
    filename
    status
    createdAt
  }
}
```

**Implementation:**
```python
@strawberry.field
def document(
    self,
    info,
    document_id: strawberry.ID,
) -> Optional[DocumentType]:
    """
    Get a single document by ID.

    Args:
        info: GraphQL execution context
        document_id: Document ID to fetch

    Returns:
        Document or None if not found

    الحصول على مستند واحد بالمعرف
    """
    # Validate input
    if not document_id:
        raise ValueError("document_id is required")

    # Get tenant ID from request context
    request = info.context["request"]
    tenant_id = get_tenant_id(request)

    # Get document repository from context
    doc_repo: DocumentRepoPort = info.context["doc_repo"]

    # Query database
    doc = doc_repo.find_by_id(document_id=str(document_id))

    # Check if document exists and belongs to tenant
    if not doc:
        return None

    if doc.tenant_id != tenant_id:
        # Return None instead of error (don't leak existence)
        return None

    # Convert to GraphQL type
    return DocumentType(
        id=doc.document_id,
        filename=doc.filename,
        content_type=doc.content_type,
        size_bytes=doc.size_bytes,
        status=DocumentStatus(doc.status.value),
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )
```

**Security Considerations:**

1. **Authorization:**
   - Check document belongs to tenant
   - Return `None` if access denied (don't leak existence)

2. **Input Validation:**
   - Require `document_id` (non-nullable in schema)
   - Validate ID format

3. **Error Handling:**
   - Return `None` for not found (GraphQL best practice)
   - Don't expose internal errors to client

### 4.3 Search Documents Resolver

**Purpose:** Full-text search with faceted results.

**GraphQL Query:**
```graphql
query SearchDocuments($query: String!, $k: Int, $sortBy: QuerySortBy) {
  searchDocuments(query: $query, k: $k, sortBy: $sortBy) {
    results {
      id
      filename
      score
    }
    total
    facets {
      name
      count
    }
  }
}
```

**Implementation:**
```python
@strawberry.field
def search_documents(
    self,
    info,
    query: str,
    k: int = 10,
    sort_by: QuerySortBy = QuerySortBy.CREATED,
    limit: int = 20,
    offset: int = 0,
) -> SearchResultType:
    """
    Search documents with hybrid search.

    Args:
        info: GraphQL execution context
        query: Search query (required)
        k: Number of results (default: 10)
        sort_by: Sort order (default: CREATED)
        limit: Max results (default: 20)
        offset: Pagination offset (default: 0)

    Returns:
        Search results with facets

    البحث عن المستندات بالبحث الهجين
    """
    # Validate inputs
    if not query or not query.strip():
        raise ValueError("query is required")

    if k < 1 or k > 100:
        raise ValueError("k must be between 1 and 100")

    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")

    # Get dependencies from context
    request = info.context["request"]
    tenant_id = get_tenant_id(request)

    search_service = info.context.get("search_service")
    doc_repo = info.context["doc_repo"]

    # Perform search
    if search_service:
        # Use search service (hybrid search)
        results = search_service.search(
            tenant_id=tenant_id,
            query=query,
            k=k,
            sort_by=sort_by.value,
        )

        # Convert to GraphQL types
        graphql_results = [
            DocumentType(
                id=doc["id"],
                filename=doc["filename"],
                content_type=doc["content_type"],
                size_bytes=doc["size_bytes"],
                status=DocumentStatus(doc["status"]),
                created_at=doc["created_at"],
                updated_at=doc.get("updated_at"),
            )
            for doc in results["items"][:limit]
        ]

        # Create facets
        facets = []
        for status, count in results.get("facets", {}).get("status", {}).items():
            facets.append(FacetType(name=status, count=count))

        return SearchResultType(
            results=graphql_results,
            total=results["total"],
            facets=facets,
        )
    else:
        # Fallback: simple document list (no search service)
        docs = doc_repo.list_documents(tenant_id=tenant_id, limit=limit, offset=offset)

        return SearchResultType(
            results=[
                DocumentType(
                    id=doc.document_id,
                    filename=doc.filename,
                    content_type=doc.content_type,
                    size_bytes=doc.size_bytes,
                    status=DocumentStatus(doc.status.value),
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                )
                for doc in docs
            ],
            total=len(docs),
            facets=None,
        )
```

**Faceted Search:**

Facets provide aggregation counts for filtering:
```json
{
  "data": {
    "searchDocuments": {
      "results": [...],
      "total": 42,
      "facets": [
        {"name": "INDEXED", "count": 35},
        {"name": "CREATED", "count": 5},
        {"name": "FAILED", "count": 2}
      ]
    }
  }
}
```

**Facet Calculation:**
- Count documents by status
- Group by content_type
- Calculate size ranges

---
## 5. Chat Session Query Resolvers

### 5.1 List Chat Sessions Resolver

**Purpose:** Fetch all chat sessions for a tenant.

**GraphQL Query:**
```graphql
query ListChatSessions($limit: Int, $offset: Int) {
  chatSessions(limit: $limit, offset: $offset) {
    id
    title
    createdAt
  }
}
```

**Implementation:**
```python
@strawberry.field
def chat_sessions(
    self,
    info,
    limit: int = 20,
    offset: int = 0,
) -> List[ChatSessionType]:
    """
    Query chat sessions with pagination.

    Args:
        info: GraphQL execution context
        limit: Max results (default: 20)
        offset: Pagination offset (default: 0)

    Returns:
        List of chat sessions

    الاستعلام عن جلسات المحادثة مع الترحيل
    """
    # Validate inputs
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit > 100:
        limit = 100
    if offset < 0:
        raise ValueError("offset must be non-negative")

    # Get tenant ID from request context
    request = info.context["request"]
    tenant_id = get_tenant_id(request)

    # Get chat repository from context
    chat_repo: ChatRepoPort = info.context["chat_repo"]

    # Query database
    sessions = chat_repo.list_sessions(
        tenant_id=tenant_id,
        limit=limit,
    )

    # Convert to GraphQL types
    return [
        ChatSessionType(
            id=session.session_id,
            title=session.title,
            created_at=session.created_at,
        )
        for session in sessions
    ]
```

**Response Example:**
```json
{
  "data": {
    "chatSessions": [
      {
        "id": "session-123",
        "title": "Research on RAG",
        "createdAt": "2026-01-31T10:00:00Z"
      }
    ]
  }
}
```

### 5.2 Get Single Chat Session Resolver

**Purpose:** Fetch a single chat session by ID.

**GraphQL Query:**
```graphql
query GetChatSession($sessionId: ID!) {
  chatSession(sessionId: $sessionId) {
    id
    title
    createdAt
  }
}
```

**Implementation:**
```python
@strawberry.field
def chat_session(
    self,
    info,
    session_id: strawberry.ID,
) -> Optional[ChatSessionType]:
    """
    Get a single chat session by ID.

    Args:
        info: GraphQL execution context
        session_id: Session ID to fetch

    Returns:
        Chat session or None if not found

    الحصول على جلسة محادثة واحدة بالمعرف
    """
    # Validate input
    if not session_id:
        raise ValueError("session_id is required")

    # Get tenant ID from request context
    request = info.context["request"]
    tenant_id = get_tenant_id(request)

    # Get chat repository from context
    chat_repo: ChatRepoPort = info.context["chat_repo"]

    # Query database
    session = chat_repo.get_session(
        tenant_id=tenant_id,
        session_id=str(session_id),
    )

    # Check if session exists and belongs to tenant
    if not session:
        return None

    # Convert to GraphQL type
    return ChatSessionType(
        id=session.session_id,
        title=session.title,
        created_at=session.created_at,
    )
```

---
## 6. Query History Resolvers

### 6.1 List Query History Resolver

**Purpose:** Fetch historical questions and answers.

**GraphQL Query:**
```graphql
query QueryHistory($limit: Int, $offset: Int) {
  queryHistory(limit: $limit, offset: $offset) {
    question
    answer
    sources
    timestamp
  }
}
```

**Implementation:**
```python
@strawberry.field
def query_history(
    self,
    info,
    limit: int = 50,
    offset: int = 0,
) -> List[QueryHistoryItemType]:
    """
    Get query history with pagination.

    Args:
        info: GraphQL execution context
        limit: Max results (default: 50)
        offset: Pagination offset (default: 0)

    Returns:
        List of query history items

    الحصول على تاريخ الاستعلامات مع الترحيل
    """
    # Validate inputs
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit > 100:
        limit = 100
    if offset < 0:
        raise ValueError("offset must be non-negative")

    # Get tenant ID from request context
    request = info.context["request"]
    tenant_id = get_tenant_id(request)

    # Get query history repository from context
    query_repo = info.context.get("query_history_repo")

    # Query database (if repository available)
    if query_repo:
        history = query_repo.list_queries(
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
        )

        # Convert to GraphQL types
        return [
            QueryHistoryItemType(
                question=item.question,
                answer=item.answer,
                sources=item.sources,
                timestamp=item.timestamp,
            )
            for item in history
        ]
    else:
        # Return empty list if repository not available
        return []
```

**Response Example:**
```json
{
  "data": {
    "queryHistory": [
      {
        "question": "What is RAG?",
        "answer": "RAG stands for Retrieval-Augmented Generation...",
        "sources": ["chunk-123", "chunk-456"],
        "timestamp": "2026-01-31T12:00:00Z"
      }
    ]
  }
}
```

---
## 7. Performance Optimization

### 7.1 N+1 Query Problem

**Problem:**
```python
@strawberry.field
def documents(self) -> List[DocumentType]:
    docs = doc_repo.list_documents(...)
    return [DocumentType(**doc) for doc in docs]

# Each DocumentType has a resolver for chunks
@strawberry.field
def chunks(self, parent) -> List[ChunkType]:
    # Runs N times for N documents!
    return chunk_repo.get_chunks(parent.id)
```

**Solution - DataLoader:**
```python
from strawberry.dataloader import DataLoader

class ChunkLoader(DataLoader):
    async def batch_load_fn(self, keys):
        # Batch query: SELECT * FROM chunks WHERE doc_id IN (...)
        chunks = chunk_repo.get_chunks_batch(keys)
        return [chunks.get(key) for key in keys]

# Create loader in request context
chunk_loader = ChunkLoader()

@strawberry.field
def chunks(self, info, parent) -> List[ChunkType]:
    return info.context["chunk_loader"].load(parent.id)
```

### 7.2 Caching

**Resolver-Level Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def _get_document_cached(document_id: str) -> dict:
    return doc_repo.find_by_id(document_id)

@strawberry.field
def document(self, document_id: strawberry.ID) -> Optional[DocumentType]:
    doc = _get_document_cached(str(document_id))
    return DocumentType(**doc) if doc else None
```

**Cache Invalidation:**
```python
# Clear cache on document update
def update_document(document_id: str, updates: dict):
    doc_repo.update(document_id, updates)
    _get_document_cached.cache_clear()
```

### 7.3 Pagination Best Practices

**Cursor-Based Pagination:**
```python
@strawberry.field
def documents_cursor(self, after: Optional[str], first: int = 20) -> List[DocumentType]:
    """
    Cursor-based pagination (better for large datasets).
    """
    if after:
        # Decode cursor (base64 encoded timestamp or ID)
        cursor_data = decode_cursor(after)
        docs = doc_repo.list_documents_after(cursor_data, limit=first)
    else:
        docs = doc_repo.list_documents(limit=first)

    # Return connection edges
    return [
        EdgeType(
            node=DocumentType(**doc),
            cursor=encode_cursor(doc["created_at"]),
        )
        for doc in docs
    ]
```

**Offset-Based Pagination:**
```python
@strawberry.field
def documents_offset(self, offset: int = 0, limit: int = 20) -> List[DocumentType]:
    """
    Offset-based pagination (simpler but slower for large offsets).
    """
    docs = doc_repo.list_documents(offset=offset, limit=limit)
    return [DocumentType(**doc) for doc in docs]
```

**When to Use:**
- **Cursor**: Real-time feeds, infinite scroll
- **Offset**: Admin panels, paginated tables

---
## 8. Best Practices

### 8.1 Input Validation

**DO:** Validate all inputs
```python
@strawberry.field
def documents(self, limit: int = 20, offset: int = 0) -> List[DocumentType]:
    # Enforce reasonable limits
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit > 100:
        limit = 100  # Clamp to maximum

    if offset < 0:
        raise ValueError("offset must be non-negative")

    # ... rest of implementation
```

**DON'T:** Trust client input
```python
@strawberry.field
def documents(self, limit: int, offset: int) -> List[DocumentType]:
    # BAD: Client could request 1,000,000 documents!
    docs = doc_repo.list_documents(limit=limit, offset=offset)
    return [DocumentType(**doc) for doc in docs]
```

### 8.2 Error Handling

**DO:** Return None for not found
```python
@strawberry.field
def document(self, document_id: strawberry.ID) -> Optional[DocumentType]:
    doc = doc_repo.find_by_id(document_id)
    return DocumentType(**doc) if doc else None
```

**DON'T:** Throw exceptions for not found
```python
@strawberry.field
def document(self, document_id: strawberry.ID) -> DocumentType:
    doc = doc_repo.find_by_id(document_id)
    if not doc:
        raise ValueError("Document not found")  # BAD
    return DocumentType(**doc)
```

### 8.3 Authentication & Authorization

**DO:** Check tenant ownership
```python
@strawberry.field
def document(self, info, document_id: strawberry.ID) -> Optional[DocumentType]:
    tenant_id = get_tenant_id(info.context["request"])
    doc = doc_repo.find_by_id(document_id)

    # Verify ownership
    if doc and doc.tenant_id != tenant_id:
        return None  # Don't leak existence

    return DocumentType(**doc) if doc else None
```

**DON'T:** Skip authorization checks
```python
@strawberry.field
def document(self, document_id: strawberry.ID) -> DocumentType:
    # BAD: Anyone can access any document!
    doc = doc_repo.find_by_id(document_id)
    return DocumentType(**doc)
```

### 8.4 Logging & Observability

**DO:** Log resolver calls
```python
from src.core.logging import get_logger

log = get_logger(__name__)

@strawberry.field
def documents(self, info, limit: int, offset: int) -> List[DocumentType]:
    tenant_id = get_tenant_id(info.context["request"])

    log.info(
        "graphql_query_documents",
        tenant_id=tenant_id,
        limit=limit,
        offset=offset,
    )

    docs = doc_repo.list_documents(tenant_id=tenant_id, limit=limit, offset=offset)
    return [DocumentType(**doc) for doc in docs]
```

### 8.5 Type Safety

**DO:** Use proper type hints
```python
from typing import List, Optional
from datetime import datetime

@strawberry.field
def documents(
    self,
    limit: int = 20,
    offset: int = 0,
) -> List[DocumentType]:
    # ...
    return result
```

---
## 9. Common Pitfalls

### 9.1 N+1 Query Problem

**Symptom:**
- Query runs 1 query for list + N queries for nested fields
- Slow performance with many items

**Solution:** Use DataLoader or batch queries

### 9.2 Overfetching

**Symptom:**
- Querying more fields than needed
- Wasted database queries

**Solution:** Let clients select fields

### 9.3 Missing Authorization

**Symptom:**
- Users can access other tenants' data
- Security vulnerability

**Solution:** Always verify tenant ownership

### 9.4 No Input Validation

**Symptom:**
- Clients can request unlimited data
- DoS vulnerability

**Solution:** Enforce limits on all parameters

### 9.5 Ignoring Errors

**Symptom:**
- Silent failures
- Debugging difficult

**Solution:** Log errors and return appropriate responses

---
## 10. Quiz

### Question 1
What is the purpose of GraphQL query resolvers?
- [ ] A) Define the GraphQL schema
- [ ] B) Fetch data for each field in the schema
- [ ] C) Validate GraphQL queries
- [ ] D) Generate documentation

**Answer:** B - Resolvers fetch data for each field.

---

### Question 2
How do you pass the tenant ID to a GraphQL resolver?
- [ ] A) Global variable
- [ ] B) Request context (info.context["request"])
- [ ] C) Environment variable
- [ ] D) Query parameter

**Answer:** B - Extract from request context.

---

### Question 3
What should you return when a resource is not found?
- [ ] A) Raise ValueError
- [ ] B) Return None or empty list
- [ ] C) Return 404 error
- [ ] D] Return null

**Answer:** B - Return None (GraphQL best practice).

---

### Question 4
How do you prevent N+1 query problems?
- [ ] A) Limit query size
- [ ] B) Use DataLoader or batch queries
- [ ] C] Cache all queries
- [ ] D] Use async resolvers

**Answer:** B - Use DataLoader for batch fetching.

---

### Question 5
What is cursor-based pagination?
- [ ] A) Using OFFSET/LIMIT
- [ ] B) Using a pointer (cursor) to the last item
- [ ] C) Using page numbers
- [ ] D] Using random sampling

**Answer:** B - Cursor points to the last fetched item.

---
## 11. References

### Official Documentation
- **Strawberry GraphQL:** https://strawberry.rocks/docs
- **GraphQL Spec:** https://spec.graphql.org/
- **GraphQL Best Practices:** https://graphql.best practices/

### Related Resources
- **DataLoader:** https://github.com/graphql/dataloader
- **Apollo Federation:** https://www.apollographql.com/docs/federation/
- **GraphQL Subscriptions:** https://www.apollographql.com/docs/graphql-subscriptions/

### Code Examples
- This repository: `src/api/v1/graphql.py`
- Document repository: `src/application/ports/document_repo.py`
- Chat repository: `src/application/ports/chat_repo.py`
- Related notebook: `notebooks/learning/02-api/graphql-queries.ipynb`

---
## 🇸🇦 ترجمة عربية / Arabic Translation

### 1. مقدمة

#### 1.1 ما هو GraphQL؟

GraphQL هي لغة استعلام لواجهات برمجة التطبيقات (APIs) تتيح للعملاء طلب البيانات التي يحتاجونها بدقة، لا أكثر ولا أقل. على عكس واجهات برمجة التطبيقات REST التي تحتوي على نقاط نهاية متعددة للموارد المختلفة، يستخدم GraphQL نقطة نهاية واحدة مع مخطط مرن.

**المزايا الرئيسية:**
- **جلب البيانات بدقة**: احصل فقط على الحقول التي تطلبها
- **نقطة نهاية واحدة**: جميع العمليات عبر عنوان URL واحد
- **مخطط مكتوب بقوة**: API ذاتية التوثيق
- **التحديثات في الوقت الفعلي**: دعم الاشتراكات المدمجة
- **التأمل في الذات**: الاستعلام عن المخطط نفسه

#### 1.2 ما هي محلات الاستعلام (Query Resolvers)؟

محلات الاستعلام هي وظائف تحل (تجلب) البيانات لكل حقل في مخطط GraphQL الخاص بك. عندما ينفذ العميل استعلامًا، يستدعي GraphQL محلل كل حقل بالتوازي ويجمع الاستجابة.

### 2. محلات استعلام GraphQL موضحة

#### 2.1 تدفق تنفيذ المحلل

عند تنفيذ استعلام GraphQL:

```
استعلام العميل
    │
    ▼
خادم GraphQL (التحليل والتحقق)
    │
    ▼
محرك تنفيذ الاستعلام
    │
    ├─→ محلل المستندات()
    │   ├─→ جلب من قاعدة البيانات
    │   └─→ إرجاع قائمة[نوع المستند]
    │
    └─→ تجميع الاستجابة
    │
    ▼
إرسال JSON إلى العميل
```

### 4. محللات استعلام المستندات

#### 4.1 محلل قائمة المستندات

**الغرض:** جلب قائمة مقسمة من المستندات للمستأجر.

**استعلام GraphQL:**
```graphql
query ListDocuments($limit: Int, $offset: Int) {
  documents(limit: $limit, offset: $offset) {
    id
    filename
    status
    createdAt
  }
}
```

**التنفيذ:**
```python
@strawberry.field
def documents(
    self,
    info,
    limit: int = 20,
    offset: int = 0,
) -> List[DocumentType]:
    """
    الاستعلام عن المستندات مع الترحيل.

    Args:
        info: سياق تنفيذ GraphQL
        limit: أقصى عدد من النتائج
        offset: إزاحة الترحيل

    Returns:
        قائمة المستندات
    """
    # التحقق من المدخلات
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit > 100:
        limit = 100

    # الحصول على معرف المستأجر من سياق الطلب
    request = info.context["request"]
    tenant_id = get_tenant_id(request)

    # الحصول على مستودع المستندات من السياق
    doc_repo = info.context["doc_repo"]

    # الاستعلام عن قاعدة البيانات
    documents = doc_repo.list_documents(tenant_id=tenant_id, limit=limit, offset=offset)

    # تحويل كيانات المجال إلى أنواع GraphQL
    return [DocumentType(**doc) for doc in documents]
```

### 8. أفضل الممارسات

#### 8.1 التحقق من صحة المدخلات

**افعل:** تحقق من جميع المدخلات
```python
@strawberry.field
def documents(self, limit: int = 20, offset: int = 0) -> List[DocumentType]:
    # فرض حدود معقولة
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit > 100:
        limit = 100
```

**لا تفعل:** ثق في مدخلات العميل
```python
@strawberry.field
def documents(self, limit: int, offset: int) -> List[DocumentType]:
    # سيء: العميل يمكن أن يطلب 1,000,000 مستند!
    docs = doc_repo.list_documents(limit=limit, offset=offset)
```

#### 8.2 المصادقة والتفويض

**افعل:** تحقق من ملكية المستأجر
```python
@strawberry.field
def document(self, info, document_id: strawberry.ID) -> Optional[DocumentType]:
    tenant_id = get_tenant_id(info.context["request"])
    doc = doc_repo.find_by_id(document_id)

    # التحقق من الملكية
    if doc and doc.tenant_id != tenant_id:
        return None

    return DocumentType(**doc) if doc else None
```

**لا تفعل:** تخطي عمليات التحقق من التفويض
```python
@strawberry.field
def document(self, document_id: strawberry.ID) -> DocumentType:
    # سيء: أي شخص يمكنه الوصول إلى أي مستند!
    doc = doc_repo.find_by_id(document_id)
    return DocumentType(**doc)
```

---
## 📝 Summary

In this comprehensive guide, we covered:

1. **GraphQL Query Resolvers** - Functions that fetch data for each field
2. **Document Resolvers** - List documents, get single document, search
3. **Chat Session Resolvers** - List sessions, get single session
4. **Query History Resolvers** - Fetch historical questions and answers
5. **Performance Optimization** - DataLoader, caching, pagination
6. **Best Practices** - Validation, error handling, authorization
7. **Common Pitfalls** - N+1 queries, overfetching, security

## 🚀 Next Steps

1. Read the companion Jupyter notebook: `notebooks/learning/02-api/graphql-queries.ipynb`
2. Implement the resolver code in `src/api/v1/graphql.py`
3. Test resolvers using GraphQL Playground at `http://localhost:8000/graphql`
4. Proceed to Phase 1.2: GraphQL Ask Question Resolver

---

**Document Version:** 1.0
**Last Updated:** 2026-01-31
**Author:** AI-Mastery-2026
