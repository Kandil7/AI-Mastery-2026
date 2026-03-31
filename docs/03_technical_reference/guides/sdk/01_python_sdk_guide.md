# Python SDK Guide
# ==================
# Python SDK for RAG Engine
# دليل Python SDK لـ RAG Engine

## Overview / نظرة عامة

The Python SDK provides a convenient interface for interacting with the RAG Engine REST API. It handles authentication, pagination, and response parsing for you.

توفر حزمة Python SDK واجهة مريحة للتفاعل مع API RAG Engine.

## Installation / التثبيت

```bash
pip install rag-engine-client
```

Or install from source:

```bash
git clone https://github.com/your-org/rag-engine-mini.git
cd rag-engine-mini
pip install -e .
```

## Quick Start / البدء السريع

```python
from rag_engine import RAGEngineClient

# Initialize client
client = RAGEngineClient(
    api_url="https://api.ragengine.com/v1",
    api_key="your-api-key",
    tenant_id="your-tenant-id",
)

# Upload a document
with open("document.pdf", "rb") as f:
    result = client.upload_document(
        filename="document.pdf",
        content=f.read(),
        content_type="application/pdf",
    )
    print(f"Document uploaded: {result['document_id']}")

# Ask a question
answer = client.ask_question(
    question="What is RAG?",
    k=5,
)
print(f"Answer: {answer['text']}")
print(f"Sources: {answer['sources']}")

# Get chat sessions
sessions = client.list_chat_sessions()
for session in sessions:
    print(f"Session: {session['title']}")
```

## Authentication / المصادقة

### API Key Authentication / مصادقة مفتاح API

```python
from rag_engine import RAGEngineClient

client = RAGEngineClient(
    api_url="https://api.ragengine.com/v1",
    api_key="sk-xxxxxxxxxxxxx",
)
```

### OAuth Authentication / مصادقة OAuth

```python
from rag_engine import RAGEngineClient

client = RAGEngineClient(
    api_url="https://api.ragengine.com/v1",
    oauth_token="ya29.a0AfH6xxxxx",
)
```

## Documents / المستندات

### Upload Document / رفع مستند

```python
# Upload from file
with open("document.pdf", "rb") as f:
    result = client.upload_document(
        filename="document.pdf",
        content=f.read(),
        content_type="application/pdf",
    )
    print(f"Document ID: {result['document_id']}")
    print(f"Status: {result['status']}")  # 'queued', 'processing', 'indexed', 'failed'

# Upload from bytes
content = b"Document content..."
result = client.upload_document(
    filename="document.txt",
    content=content,
    content_type="text/plain",
)
```

### List Documents / قائمة المستندات

```python
# List all documents
documents = client.list_documents(
    limit=20,
    offset=0,
)

for doc in documents:
    print(f"ID: {doc['id']}")
    print(f"Filename: {doc['filename']}")
    print(f"Status: {doc['status']}")
    print(f"Created: {doc['created_at']}")
    print("-" * 40)

# Filter by status
indexed_docs = client.list_documents(
    status="indexed",
    limit=50,
)
```

### Get Document / الحصول على مستند

```python
# Get document by ID
document = client.get_document(document_id="doc-123")

print(f"Filename: {document['filename']}")
print(f"Size: {document['size_bytes']}")
print(f"Status: {document['status']}")
```

### Delete Document / حذف مستند

```python
success = client.delete_document(document_id="doc-123")
if success:
    print("Document deleted successfully")
```

### Bulk Operations / العمليات بالجملة

```python
# Bulk upload
files = [
    {"filename": "doc1.pdf", "content": b"...", "content_type": "application/pdf"},
    {"filename": "doc2.pdf", "content": b"...", "content_type": "application/pdf"},
]
results = client.bulk_upload_documents(files)

print(f"Success: {results['success_count']}")
print(f"Failures: {results['failure_count']}")

# Bulk delete
success = client.bulk_delete_documents([
    "doc-123",
    "doc-456",
    "doc-789",
])
```

## Search / البحث

### Search Documents / البحث عن المستندات

```python
# Full-text search
results = client.search_documents(
    query="machine learning",
    k=10,
    sort_by="created",  # 'created', 'updated', 'filename', 'size'
)

print(f"Total results: {results['total']}")

for doc in results['results']:
    print(f"Document: {doc['filename']}")
    print(f"Score: {doc['score']}")

# With faceted search
results = client.search_documents(
    query="PDF",
    k=10,
    filters={
        "status": "indexed",
        "content_type": "application/pdf",
    },
)
```

### Auto-Suggest / الاقتراح التلقائي

```python
# Get search suggestions
suggestions = client.get_search_suggestions(
    query="vector",
    limit=5,
    types=["document", "query"],  # 'document', 'query', 'topic'
)

for suggestion in suggestions:
    print(f"Suggestion: {suggestion['text']}")
    print(f"Type: {suggestion['type']}")
    print(f"Relevance: {suggestion['relevance_score']}")
```

## Chat / المحادثة

### Ask Question / طرح سؤال

```python
# Ask a question
answer = client.ask_question(
    question="What is retrieval-augmented generation?",
    k=5,
)

print(f"Answer: {answer['text']}")
print(f"Sources: {answer['sources']}")
print(f"Retrieval K: {answer['retrieval_k']}")

# With performance metrics
answer = client.ask_question(
    question="How does RAG work?",
    k=10,
)

if 'embed_ms' in answer:
    print(f"Embedding time: {answer['embed_ms']}ms")
if 'search_ms' in answer:
    print(f"Search time: {answer['search_ms']}ms")
if 'llm_ms' in answer:
    print(f"LLM time: {answer['llm_ms']}ms")
```

### Create Chat Session / إنشاء جلسة محادثة

```python
# Create new session
session = client.create_chat_session(
    title="RAG Architecture Discussion",
)

print(f"Session ID: {session['id']}")
print(f"Title: {session['title']}")
```

### List Chat Sessions / قائمة جلسات المحادثة

```python
# List sessions
sessions = client.list_chat_sessions(
    limit=20,
    offset=0,
)

for session in sessions:
    print(f"Session: {session['title']}")
    print(f"Created: {session['created_at']}")
    print(f"ID: {session['id']}")
```

### Get Chat Session / الحصول على جلسة محادثة

```python
# Get session details
session = client.get_chat_session(session_id="session-123")

print(f"Title: {session['title']}")
print(f"Turns: {len(session.get('turns', []))}")
```

### Stream Answer / تدفق الإجابة

```python
# Stream answers in real-time
for chunk in client.ask_question_stream(
    question="Explain vector databases",
    k=5,
):
    print(chunk, end="", flush=True)
print()
```

## Exports / التصدير

### Export Documents / تصدير المستندات

```python
# Export to PDF
pdf_content = client.export_documents(
    format="pdf",
    document_ids=["doc-123", "doc-456"],
    title="My Documents",
)
with open("documents.pdf", "wb") as f:
    f.write(pdf_content)

# Export to CSV
csv_content = client.export_documents(
    format="csv",
    document_ids=["doc-123", "doc-456"],
)
with open("documents.csv", "wb") as f:
    f.write(csv_content)

# Export to JSON
json_content = client.export_documents(
    format="json",
    document_ids=["doc-123", "doc-456"],
)
with open("documents.json", "wb") as f:
    f.write(json_content)
```

### Export Chat Sessions / تصدير جلسات المحادثة

```python
# Export chat to Markdown
md_content = client.export_chat_sessions(
    format="markdown",
    session_ids=["session-123"],
)
with open("chat.md", "wb") as f:
    f.write(md_content)
```

## A/B Testing / اختبار A/B

### Get Experiments / الحصول على التجارب

```python
# List experiments
experiments = client.list_experiments(
    status="active",  # 'active', 'paused', 'completed'
    limit=50,
)

for exp in experiments:
    print(f"Experiment: {exp['name']}")
    print(f"Status: {exp['status']}")
    print(f"Variants: {len(exp['variants'])}")
```

### Assign Variant / تعيين نسخة

```python
# Assign user to variant
assignment = client.assign_variant(
    experiment_id="exp-123",
    user_id="user-456",
)

print(f"Assigned variant: {assignment['variant_name']}")
print(f"Config: {assignment['variant_config']}")
```

### Record Conversion / تسجيل تحويل

```python
# Record conversion event
client.record_conversion(
    experiment_id="exp-123",
    variant_id="variant-A",
    success=True,
    value=10.0,  # Optional conversion value
)
```

## GraphQL Client / عميل GraphQL

```python
from rag_engine import GraphQLClient

# Initialize GraphQL client
client = GraphQLClient(
    api_url="https://api.ragengine.com/graphql",
    api_key="your-api-key",
)

# Execute GraphQL query
query = """
    query {
        documents(limit: 20) {
            id
            filename
            status
            sizeBytes
        }
    }
"""

result = client.execute(query)
documents = result['data']['documents']

# Execute mutation
mutation = """
    mutation($question: String!) {
        askQuestion(question: $question, k: 5) {
            text
            sources
        }
    }
"""

variables = {"question": "What is RAG?"}
result = client.execute(mutation, variables=variables)

answer = result['data']['askQuestion']
print(f"Answer: {answer['text']}")
```

## Error Handling / معالجة الأخطاء

```python
from rag_engine import RAGEngineClient, RAGEngineError

client = RAGEngineClient(
    api_url="https://api.ragengine.com/v1",
    api_key="your-api-key",
)

try:
    document = client.get_document(document_id="doc-123")
except RAGEngineError as e:
    print(f"Error: {e.message}")
    print(f"Status Code: {e.status_code}")
    print(f"Request ID: {e.request_id}")

# Retry with exponential backoff
from rag_engine import retry

@retry(max_attempts=3, backoff_factor=2)
def get_document_with_retry(doc_id):
    return client.get_document(doc_id)

document = get_document_with_retry("doc-123")
```

## Advanced Usage / الاستخدام المتقدم

### Custom Headers / رؤوس مخصصة

```python
client = RAGEngineClient(
    api_url="https://api.ragengine.com/v1",
    api_key="your-api-key",
    headers={
        "X-Custom-Header": "value",
        "X-Request-ID": "custom-id",
    },
)
```

### Timeout Configuration / تكوين المهلة الزمنية

```python
client = RAGEngineClient(
    api_url="https://api.ragengine.com/v1",
    api_key="your-api-key",
    timeout=30,  # seconds
    connect_timeout=10,  # seconds
)
```

### Webhooks / خطافات الويب

```python
from rag_engine import WebhookClient

webhook_client = WebhookClient(
    webhook_url="https://your-site.com/webhooks",
    secret="your-webhook-secret",
)

# Verify webhook signature
def handle_webhook(request_data, signature):
    if webhook_client.verify_signature(request_data, signature):
        print("Webhook verified")
        # Process webhook
    else:
        print("Invalid webhook signature")

# Register webhook
client.register_webhook(
    events=["document.indexed", "chat.updated"],
    url="https://your-site.com/webhooks",
)
```

## Best Practices / أفضل الممارسات

1. **Use API Keys for Authentication** - More secure than OAuth for service accounts
2. **Implement Retry Logic** - Network failures are common; use exponential backoff
3. **Cache Results** - Reduce API calls by caching frequently accessed data
4. **Use Streaming** - For long responses, use streaming endpoints
5. **Handle Errors Gracefully** - Always wrap API calls in try/except blocks
6. **Rate Limit Awareness** - Respect API rate limits; implement backoff
7. **Tenant Isolation** - Always specify tenant_id for multi-tenant security

## Examples / أمثلة

### Complete RAG Pipeline / أنبوب RAG كامل

```python
from rag_engine import RAGEngineClient

client = RAGEngineClient(
    api_url="https://api.ragengine.com/v1",
    api_key="your-api-key",
)

# Step 1: Upload document
with open("manual.pdf", "rb") as f:
    doc = client.upload_document(
        filename="manual.pdf",
        content=f.read(),
        content_type="application/pdf",
    )

# Step 2: Wait for indexing
import time
time.sleep(5)

# Step 3: Ask questions
questions = [
    "What is the main topic?",
    "How do I use feature X?",
    "What are the limitations?",
]

for question in questions:
    answer = client.ask_question(question=question, k=5)
    print(f"Q: {question}")
    print(f"A: {answer['text']}\n")
```

### Batch Processing / المعالجة بالجملة

```python
import os

# Upload all PDFs in directory
pdf_files = [f for f in os.listdir("./documents") if f.endswith(".pdf")]

for filename in pdf_files:
    with open(f"./documents/{filename}", "rb") as f:
        result = client.upload_document(
            filename=filename,
            content=f.read(),
            content_type="application/pdf",
        )
        print(f"Uploaded: {filename} -> {result['document_id']}")
```

## Support / الدعم

- **Documentation:** https://docs.ragengine.com
- **GitHub Issues:** https://github.com/your-org/rag-engine-mini/issues
- **Email:** support@ragengine.com

---

**Python SDK Version:** 1.0.0
**Last Updated:** 2026-01-31
