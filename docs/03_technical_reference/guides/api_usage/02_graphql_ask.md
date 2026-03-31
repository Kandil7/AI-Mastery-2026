# GraphQL Ask Question Resolver - Complete Implementation Guide
# ==========================================================

## 📚 Learning Objectives

By the end of this guide, you will understand:
- GraphQL mutations and how they differ from queries
- How to integrate GraphQL mutations with use cases
- Ask question resolver implementation with RAG pipeline
- Error handling and validation in mutations
- Performance considerations for LLM operations
- Best practices for async mutations

---
## 📌 Table of Contents

1. [Introduction](#1-introduction)
2. [GraphQL Mutations Explained](#2-graphql-mutations-explained)
3. [Ask Question Architecture](#3-ask-question-architecture)
4. [Ask Question Resolver Implementation](#4-ask-question-resolver-implementation)
5. [Error Handling & Validation](#5-error-handling--validation)
6. [Performance Optimization](#6-performance-optimization)
7. [Best Practices](#7-best-practices)
8. [Common Pitfalls](#8-common-pitfalls)
9. [Quiz](#9-quiz)
10. [References](#10-references)

---
## 1. Introduction

### 1.1 What are GraphQL Mutations?

Mutations are GraphQL operations that **modify data** on the server. While queries fetch data (read-only), mutations perform write operations.

**Key Differences from Queries:**
- **Side Effects**: Mutations have side effects (database writes, API calls)
- **Sequential Execution**: Mutations execute sequentially (not parallel like queries)
- **Return Data**: Mutations return the modified data (not just success/failure)
- **Naming Convention**: Mutations typically use verb names (create, update, delete)

### 1.2 Why Implement Ask Question as a Mutation?

The `ask_question` operation is technically a mutation because it:
1. **Creates** a new query history record
2. **Updates** metrics (token usage, latency)
3. **Calls external LLM API** (side effect)
4. **Generates** a new answer (creation)

**Alternative Approach:**
- As a **query**: If you want to keep it read-only and handle side effects separately
- As a **mutation**: When side effects are part of the operation (our approach)

### 1.3 The Ask Question Flow

```
Client Request
    │
    ▼
GraphQL Mutation: ask_question(question, k, document_id)
    │
    ▼
GraphQL Resolver: AskQuestionMutation.ask_question()
    │
    ├─→ Input Validation
    │   - Validate question not empty
    │   - Validate k in range
    │   - Validate document_id if provided
    │
    ├─→ Tenant Extraction
    │   - Extract tenant_id from request context
    │
    ├─→ Use Case: AskQuestionHybridUseCase
    │   ├─→ Query Expansion (optional)
    │   ├─→ Embed Question
    │   ├─→ Vector Search (Qdrant)
    │   ├─→ Keyword Search (Postgres FTS)
    │   ├─→ RRF Fusion
    │   ├─→ Reranking (Cross-Encoder)
    │   ├─→ Prompt Building
    │   ├─→ LLM Generation (OpenAI)
    │   └─→ Privacy Guard
    │
    ├─→ Save to Query History
    │
    ├─→ Metrics Collection
    │   - API request latency
    │   - Token usage
    │   - Retrieval score
    │
    └─→ Return AnswerType
        │
        ▼
GraphQL Response
    │
    ▼
Send JSON to Client
```

---
## 2. GraphQL Mutations Explained

### 2.1 Mutation Definition Pattern

```python
@strawberry.type
class Mutation:
    @strawberry.mutation
    def mutation_name(
        self,
        info,
        required_arg: str,
        optional_arg: Optional[str] = None,
    ) -> ReturnType:
        """
        Mutation description.

        Args:
            info: GraphQL execution context
            required_arg: Required argument
            optional_arg: Optional argument

        Returns:
            The result of the mutation
        """
        # 1. Validate inputs
        # 2. Perform operation
        # 3. Return result
        pass
```

### 2.2 Async Mutations

Mutations can be async for I/O-bound operations:

```python
@strawberry.mutation
async def async_mutation(
    self,
    info,
    question: str,
) -> AnswerType:
    """
    Async mutation for long-running operations.
    """
    # Call async use case
    result = await ask_use_case.execute_async(
        question=question,
        tenant_id=get_tenant_id(info.context),
    )
    return AnswerType(**result)
```

### 2.3 Error Handling in Mutations

```python
@strawberry.mutation
def mutation_with_errors(
    self,
    input_data: str,
) -> ResultType:
    """
    Mutation with proper error handling.
    """
    try:
        # Perform operation
        result = perform_operation(input_data)
        return ResultType(success=True, data=result)

    except ValidationError as e:
        # Input validation error
        raise ValueError(f"Invalid input: {e}")

    except DatabaseError as e:
        # Database operation error
        raise RuntimeError(f"Database error: {e}")

    except Exception as e:
        # Unexpected error
        # Log the error
        log.error("Unexpected error", error=str(e))
        # Return error response
        return ResultType(success=False, error=str(e))
```

---
## 3. Ask Question Architecture

### 3.1 AskQuestionHybridUseCase

The ask question use case is the core of the RAG pipeline:

**Dependencies:**
- `cached_embeddings`: Embed question into vectors (cached)
- `vector_store`: Vector similarity search (Qdrant)
- `keyword_store`: Full-text search (Postgres FTS)
- `chunk_text_reader`: Hydrate chunk texts from database
- `reranker`: Re-rank results with cross-encoder
- `llm`: Generate answer with context
- `query_expansion_service`: Expand query with LLM (optional)
- `self_critique`: Self-RAG with critique (optional)
- `router`: Semantic routing (optional)
- `privacy`: Privacy guard for PII (optional)

### 3.2 AskHybridRequest

```python
@dataclass
class AskHybridRequest:
    """Request data for hybrid RAG question."""

    tenant_id: str
    question: str

    # Optional: restrict to single document (ChatPDF mode)
    document_id: str | None = None

    # Retrieval parameters
    k_vec: int = 30  # Top-K for vector search
    k_kw: int = 30  # Top-K for keyword search
    fused_limit: int = 40  # Max after fusion
    rerank_top_n: int = 8  # Final top results after reranking

    # Advanced features
    expand_query: bool = False  # Use LLM to expand query
```

### 3.3 Answer

```python
@dataclass
class Answer:
    """RAG-generated answer with metadata."""

    text: str
    sources: List[str]
    retrieval_k: int
    embed_ms: int | None
    search_ms: int | None
    llm_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
```

---
## 4. Ask Question Resolver Implementation

### 4.1 GraphQL Answer Type

```python
@strawberry.type
class AnswerType:
    """Answer GraphQL type with sources."""

    text: str
    sources: List[str]
    retrieval_k: int
    embed_ms: Optional[int]
    search_ms: Optional[int]
    llm_ms: Optional[int]
```

### 4.2 Ask Question Mutation

**GraphQL Mutation:**
```graphql
mutation AskQuestion($question: String!, $k: Int, $documentId: ID) {
  askQuestion(question: $question, k: $k, documentId: $documentId) {
    text
    sources
    retrievalK
    embedMs
    searchMs
    llmMs
  }
}
```

**Implementation:**
```python
@strawberry.type
class Mutation:
    @strawberry.mutation
    def ask_question(
        self,
        info,
        question: str,
        k: int = 5,
        document_id: Optional[strawberry.ID] = None,
    ) -> AnswerType:
        """
        Ask a question using GraphQL mutation.

        Args:
            info: GraphQL execution context
            question: Question to ask (required)
            k: Number of chunks to retrieve (default: 5)
            document_id: Optional document ID for chat mode

        Returns:
            Answer with text and sources

        طرح سؤال باستخدام GraphQL
        """
        # 1. Validate inputs
        if not question or not question.strip():
            raise ValueError("question is required and cannot be empty")

        if k < 1 or k > 100:
            raise ValueError("k must be between 1 and 100")

        question = question.strip()

        if len(question) > 2000:
            raise ValueError("question too long (max 2000 characters)")

        # 2. Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id
        request = info.context.get("request")
        if not request:
            raise RuntimeError("Request context not available")

        tenant_id = get_tenant_id(request)

        # 3. Get use case from context
        ask_use_case = info.context.get("ask_hybrid_use_case")
        if not ask_use_case:
            raise RuntimeError("Ask use case not available")

        # 4. Execute ask question use case
        from src.application.use_cases.ask_question_hybrid import AskHybridRequest
        request_data = AskHybridRequest(
            tenant_id=tenant_id,
            question=question,
            document_id=str(document_id) if document_id else None,
            rerank_top_n=k,
        )

        result = ask_use_case.execute(request_data)

        # 5. Convert to GraphQL type
        return AnswerType(
            text=result.text,
            sources=result.sources,
            retrieval_k=result.retrieval_k,
            embed_ms=result.embed_ms,
            search_ms=result.search_ms,
            llm_ms=result.llm_ms,
        )
```

**Key Implementation Details:**

1. **Input Validation:**
   - Validate `question` not empty
   - Validate `k` in range [1, 100]
   - Validate `question` length <= 2000 characters

2. **Tenant Extraction:**
   - Extract `tenant_id` from request headers
   - Pass to use case for multi-tenant isolation

3. **Use Case Integration:**
   - Get `ask_hybrid_use_case` from context
   - Create `AskHybridRequest` with all parameters
   - Execute the RAG pipeline

4. **Type Conversion:**
   - Convert domain `Answer` to GraphQL `AnswerType`
   - Map all fields correctly

**Response Example:**
```json
{
  "data": {
    "askQuestion": {
      "text": "RAG stands for Retrieval-Augmented Generation...",
      "sources": ["chunk-123", "chunk-456", "chunk-789", "chunk-012", "chunk-345"],
      "retrievalK": 5,
      "embedMs": 150,
      "searchMs": 200,
      "llmMs": 1200
    }
  }
}
```

### 4.3 Handling Errors

```python
@strawberry.mutation
def ask_question(
    self,
    info,
    question: str,
    k: int = 5,
    document_id: Optional[strawberry.ID] = None,
) -> AnswerType:
    """
    Ask a question with proper error handling.
    """
    try:
        # Validate and execute
        # ... (implementation from above)

    except ValueError as e:
        # Input validation error
        raise ValueError(str(e))

    except Exception as e:
        # Unexpected error
        from src.core.logging import get_logger
        log = get_logger(__name__)
        log.error(
            "ask_question_error",
            tenant_id=tenant_id,
            question=question[:100],  # Truncate for logging
            error=str(e),
        )
        # Return error response
        raise RuntimeError(f"Failed to ask question: {e}")
```

---
## 5. Error Handling & Validation

### 5.1 Input Validation Rules

| Field | Validation | Error Message |
|--------|-------------|---------------|
| `question` | Required, not empty | "question is required and cannot be empty" |
| `question` | Max 2000 characters | "question too long (max 2000 characters)" |
| `k` | Between 1 and 100 | "k must be between 1 and 100" |
| `document_id` | Valid UUID format (if provided) | "Invalid document ID format" |

### 5.2 Error Response Format

**GraphQL Error Response:**
```json
{
  "data": null,
  "errors": [
    {
      "message": "question is required and cannot be empty",
      "locations": [
        {
          "line": 2,
          "column": 3
        }
      ],
      "path": ["askQuestion"]
    }
  ]
}
```

### 5.3 Business Logic Errors

```python
# Example: Document not found
if document_id:
    doc = doc_repo.find_by_id(str(document_id))
    if not doc:
        raise ValueError(f"Document {document_id} not found")

# Example: Rate limit exceeded
if rate_limiter.is_limited(tenant_id):
    raise ValueError("Rate limit exceeded. Please try again later.")

# Example: LLM API error
try:
    answer = llm.generate(prompt)
except OpenAIError as e:
    raise RuntimeError(f"LLM API error: {e}")
```

---
## 6. Performance Optimization

### 6.1 Caching Question Results

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def _get_cached_answer(question: str, k: int) -> Answer:
    """Cache answers for repeated questions."""
    return ask_use_case.execute(AskHybridRequest(...))

@strawberry.mutation
def ask_question(
    self,
    info,
    question: str,
    k: int = 5,
) -> AnswerType:
    """Ask question with caching."""
    # Check cache first
    cache_key = f"{tenant_id}:{question}:{k}"
    cached = _get_cached_answer(question, k)
    if cached:
        return AnswerType(**cached)

    # Execute and cache result
    result = ask_use_case.execute(...)
    _get_cached_answer.cache_clear()  # Clear if needed
    return AnswerType(**result)
```

### 6.2 Streaming Responses

For long answers, consider streaming:

```python
@strawberry.mutation
async def ask_question_stream(
    self,
    info,
    question: str,
) -> AsyncGenerator[str, None]:
    """
    Stream answer chunks as they're generated.

    Yields:
        Answer text chunks
    """
    result = await ask_use_case.execute_async(...)

    # Stream token by token
    for token in result.text.split():
        yield token
```

### 6.3 Async Execution

```python
@strawberry.mutation
async def ask_question_async(
    self,
    info,
    question: str,
    k: int = 5,
) -> AnswerType:
    """
    Async mutation for non-blocking execution.
    """
    request = info.context["request"]
    tenant_id = get_tenant_id(request)

    # Execute async
    result = await ask_use_case.execute_async(
        AskHybridRequest(
            tenant_id=tenant_id,
            question=question,
            rerank_top_n=k,
        )
    )

    return AnswerType(**result)
```

---
## 7. Best Practices

### 7.1 Use Case Integration

**DO:** Delegate business logic to use cases
```python
@strawberry.mutation
def ask_question(self, info, question: str) -> AnswerType:
    use_case = info.context["ask_hybrid_use_case"]
    result = use_case.execute(AskHybridRequest(...))
    return AnswerType(**result)
```

**DON'T:** Implement business logic in resolvers
```python
@strawberry.mutation
def ask_question(self, info, question: str) -> AnswerType:
    # BAD: Business logic in resolver!
    embeddings = embed(question)
    vector_results = qdrant.search(embeddings)
    # ... more logic ...
    return AnswerType(...)
```

### 7.2 Error Handling

**DO:** Validate inputs before execution
```python
if not question or not question.strip():
    raise ValueError("question is required")
```

**DON'T:** Skip validation
```python
@strawberry.mutation
def ask_question(self, info, question: str) -> AnswerType:
    # BAD: No validation!
    result = use_case.execute(AskHybridRequest(question=question))
```

### 7.3 Context Management

**DO:** Extract tenant_id from request context
```python
tenant_id = get_tenant_id(info.context["request"])
```

**DON'T:** Hardcode tenant ID
```python
@strawberry.mutation
def ask_question(self, info, question: str) -> AnswerType:
    # BAD: No tenant isolation!
    tenant_id = "fixed-tenant-id"
```

### 7.4 Metrics Collection

**DO:** Track metrics for observability
```python
from src.core.observability import API_REQUEST_LATENCY, TOKEN_USAGE

@strawberry.mutation
def ask_question(self, info, question: str) -> AnswerType:
    start_time = time.time()

    # Execute
    result = use_case.execute(...)

    # Record metrics
    API_REQUEST_LATENCY.labels(
        endpoint="ask_question",
        status="success",
    ).observe(time.time() - start_time)

    TOKEN_USAGE.labels(
        model="gpt-4",
        type="completion",
    ).inc(result.completion_tokens or 0)

    return AnswerType(**result)
```

---
## 8. Common Pitfalls

### 8.1 Blocking Operations

**Problem:** Blocking I/O in mutations causes slow responses.

**Solution:** Use async mutations and await I/O operations.

### 8.2 No Error Handling

**Problem:** Unhandled exceptions cause server errors.

**Solution:** Wrap in try-catch and return appropriate errors.

### 8.3 Missing Tenant Isolation

**Problem:** Users can access other tenants' data.

**Solution:** Always extract and validate tenant_id.

### 8.4 No Input Validation

**Problem:** Invalid inputs cause runtime errors.

**Solution:** Validate all inputs before processing.

### 8.5 Leaking Internal Errors

**Problem:** Internal errors exposed to clients.

**Solution:** Log errors, return user-friendly messages.

---
## 9. Quiz

### Question 1
What is the primary difference between GraphQL queries and mutations?
- [ ] A) Queries execute in parallel, mutations execute sequentially
- [ ] B) Mutations are for reading, queries are for writing
- [ ] C) Queries are optional, mutations are required
- [ ] D] Mutations have side effects, queries don't

**Answer:** D - Mutations have side effects.

---

### Question 2
How do you handle validation errors in GraphQL mutations?
- [ ] A) Return error field in response
- [ ] B) Raise ValueError with error message
- [ ] C) Return null result
- [ ] D] Log error and return success

**Answer:** B - Raise ValueError with error message.

---

### Question 3
What should the ask_question mutation do besides generating an answer?
- [ ] A) Only generate the answer
- [ ] B) Generate answer and save to query history
- [ ] C) Generate answer, save history, and collect metrics
- [ ] D] Only return sources

**Answer:** C - Generate answer, save history, collect metrics.

---

### Question 4
How do you optimize performance for repeated questions?
- [ ] A) Always generate new answer
- [ ] B) Cache answers by question and k
- [ ] C) Use smaller k value
- [ ] D] Stream responses

**Answer:** B - Cache answers for repeated questions.

---

### Question 5
Why is ask_question a mutation instead of a query?
- [ ] A) It requires authentication
- [ ] B) It has side effects (creates history, calls LLM)
- [ ] C) It returns complex data
- [ ] D) It uses external services

**Answer:** B - It has side effects (creates history, calls LLM).

---
## 10. References

### Official Documentation
- **Strawberry Mutations:** https://strawberry.rocks/docs/mutations
- **GraphQL Spec (Mutations):** https://spec.graphql.org/draft/#sec-Mutation
- **GraphQL Best Practices:** https://graphql.best practices/

### Related Resources
- **RAG Architecture:** Related docs in this repository
- **Use Case Pattern:** `src/application/use_cases/ask_question_hybrid.py`
- **Related notebook:** `notebooks/learning/02-api/graphql-ask.ipynb`

---
## 🇸🇦 ترجمة عربية / Arabic Translation

### 1. مقدمة

#### 1.1 ما هي طفرات GraphQL (Mutations)؟

الطفرات (Mutations) هي عمليات GraphQL تقوم بتعديل البيانات على الخادم. في حين تسترجع الاستعلامات (queries) البيانات (للقراءة فقط)، تقوم الطفرات بعمليات الكتابة.

**الاختلافات الرئيسية عن الاستعلامات:**
- **الآثار الجانبية**: للطفرات آثار جانبية (كتابات في قاعدة البيانات، استدعاءات API)
- **التنفيذ المتسلسل**: تُنفذ الطفرات بشكل متسلسل (ليس متوازيًا مثل الاستعلامات)
- **إرجاع البيانات**: تُرجع الطفرات البيانات المعدلة (ليس فقط النجاح/الفشل)
- **اتفاقيات التسمية**: تستخدم الطفرات أسماء الأفعال (إنشاء، تحديث، حذف)

### 4. تنفيذ محلل طرح السؤال

#### 4.2 طفرة طرح السؤال

**طفرة GraphQL:**
```graphql
mutation AskQuestion($question: String!, $k: Int, $documentId: ID) {
  askQuestion(question: $question, k: $k, documentId: $documentId) {
    text
    sources
    retrievalK
    embedMs
    searchMs
    llmMs
  }
}
```

**التنفيذ:**
```python
@strawberry.mutation
def ask_question(
    self,
    info,
    question: str,
    k: int = 5,
    document_id: Optional[strawberry.ID] = None,
) -> AnswerType:
    """
    طرح سؤال باستخدام طفرة GraphQL.

    Args:
        info: سياق تنفيذ GraphQL
        question: السؤال المطلوب (مطلوب)
        k: عدد المقاطع لاسترجاعها (الافتراضي: 5)
        document_id: معرف المستند الاختياري لوضع المحادثة

    Returns:
        إجابة بالنص والمصادر
    """
    # 1. التحقق من صحة المدخلات
    if not question or not question.strip():
        raise ValueError("question is required and cannot be empty")

    if k < 1 or k > 100:
        raise ValueError("k must be between 1 and 100")

    # 2. استخراج معرف المستأجر من سياق الطلب
    tenant_id = get_tenant_id(info.context["request"])

    # 3. الحصول على حالة الاستخدام من السياق
    ask_use_case = info.context["ask_hybrid_use_case"]

    # 4. تنفيذ حالة الاستخدام
    result = ask_use_case.execute(
        AskHybridRequest(
            tenant_id=tenant_id,
            question=question,
            document_id=str(document_id) if document_id else None,
            rerank_top_n=k,
        )
    )

    # 5. التحويل إلى نوع GraphQL
    return AnswerType(
        text=result.text,
        sources=result.sources,
        retrieval_k=result.retrieval_k,
        embed_ms=result.embed_ms,
        search_ms=result.search_ms,
        llm_ms=result.llm_ms,
    )
```

### 7. أفضل الممارسات

#### 7.1 تكامل حالة الاستخدام

**افعل:** فوّض منطق الأعمال إلى حالات الاستخدام
```python
@strawberry.mutation
def ask_question(self, info, question: str) -> AnswerType:
    use_case = info.context["ask_hybrid_use_case"]
    result = use_case.execute(AskHybridRequest(...))
    return AnswerType(**result)
```

**لا تفعل:** نفذ منطق الأعمال في المحلات
```python
@strawberry.mutation
def ask_question(self, info, question: str) -> AnswerType:
    # سيء: منطق الأعمال في المحلل!
    embeddings = embed(question)
    vector_results = qdrant.search(embeddings)
```

#### 7.2 معالجة الأخطاء

**افعل:** تحقق من صحة المدخلات قبل التنفيذ
```python
if not question or not question.strip():
    raise ValueError("question is required")
```

**لا تفعل:** تخطي التحقق من الصحة
```python
@strawberry.mutation
def ask_question(self, info, question: str) -> AnswerType:
    # سيء: لا توجد التحقق من الصحة!
    result = use_case.execute(AskHybridRequest(question=question))
```

---
## 📝 Summary

In this comprehensive guide, we covered:

1. **GraphQL Mutations** - Operations that modify data
2. **Ask Question Architecture** - Integration with RAG pipeline
3. **Resolver Implementation** - Complete ask_question mutation
4. **Error Handling** - Validation and error responses
5. **Performance Optimization** - Caching, streaming, async
6. **Best Practices** - Use case integration, context management
7. **Common Pitfalls** - Blocking, no error handling, missing isolation

## 🚀 Next Steps

1. Read the companion Jupyter notebook: `notebooks/learning/02-api/graphql-ask.ipynb`
2. Implement the resolver code in `src/api/v1/graphql.py`
3. Test resolvers using GraphQL Playground
4. Proceed to Phase 1.3: GraphQL Chat Session Resolvers

---

**Document Version:** 1.0
**Last Updated:** 2026-01-31
**Author:** AI-Mastery-2026
