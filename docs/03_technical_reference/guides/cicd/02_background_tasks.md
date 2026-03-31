# Background Tasks with Celery
# مهام الخلفية مع Celery

## Table of Contents / جدول المحتويات

1. [Introduction / مقدمة](#introduction)
2. [Why Background Tasks? / لماذا مهام الخلفية؟](#why-background-tasks)
3. [Celery Architecture / معمارية Celery](#celery-architecture)
4. [Task Design Patterns / أنماط تصميم المهام](#task-design-patterns)
5. [Implemented Tasks / المهام المُنفذة](#implemented-tasks)
6. [Best Practices / أفضل الممارسات](#best-practices)
7. [Common Pitfalls / الأخطاء الشائعة](#common-pitfalls)
8. [Monitoring and Observability / المراقبة والقابلية للملاحظة](#monitoring)
9. [Summary / الملخص](#summary)

---

## Introduction / مقدمة

### What are Background Tasks? / ما هي مهام الخلفية؟

Background tasks are operations that run independently of the main request-response cycle. They allow your application to:

- Process long-running operations without blocking HTTP requests
- Handle computationally intensive work asynchronously
- Schedule work to run at specific times or intervals
- Scale task processing independently of web servers

**Background tasks** are essential for modern web applications that need to handle operations like:
- Document indexing and embedding generation
- Email sending
- Report generation
- Data export/import
- Image/video processing

### مهام الخلفية هي عمليات تعمل بشكل مستقل عن دورة الطلب والاستجابة الرئيسية. تتيح لتطبيقك:

- معالجة العمليات طويلة الأمد دون حظر طلبات HTTP
- التعامل مع الأعباء الحسابية بشكل غير متزامن
- جدولة العمل لتشغيله في أوقات محددة أو على فترات منتظمة
- توسيع معالجة المهام بشكل مستقل عن خوادم الويب

**مهمة الخلفية** ضرورية لتطبيقات الويب الحديثة التي تحتاج إلى التعامل مع عمليات مثل:
- فهرسة المستندات وتوليد التضمينات
- إرسال البريد الإلكتروني
- إنشاء التقارير
- تصدير/استيراد البيانات
- معالجة الصور والفيديو

---

## Why Background Tasks? / لماذا مهام الخلفية؟

### The Request-Response Problem / مشكلة الطلب والاستجابة

Traditional HTTP requests follow a synchronous pattern:

```python
# Synchronous processing (blocks the request)
@app.post("/upload")
async def upload_document(file: UploadFile):
    # This blocks the HTTP connection until complete
    text = extract_text(file)
    chunks = chunk_text(text)
    embeddings = generate_embeddings(chunks)
    save_to_vector_store(chunks, embeddings)
    return {"status": "done"}
```

**Problems with synchronous processing:**
- Timeout errors (HTTP clients timeout after 30-60 seconds)
- Poor user experience (spinning loaders)
- Server resource exhaustion (blocked threads)
- No retry mechanism on failures

### مشكلة الطلب والاستجابة

تتبع طلبات HTTP التقليدية نمطًا متزامنًا:

**مشاكل المعالجة المتزامنة:**
- أخطاء انتهاء المهلة (العملاء ينتهي بهم الأمر بعد 30-60 ثانية)
- تجربة مستخدم سيئة (أشرطة تحميل تدور)
- استنزاف موارد الخادم (سلاسل محظورة)
- عدم وجود آلية إعادة المحاولة عند الفشل

### The Solution: Asynchronous Task Processing / الحل: معالجة المهام غير المتزامنة

```python
# Asynchronous processing (non-blocking)
@app.post("/upload")
async def upload_document(file: UploadFile):
    document_id = save_file(file)
    # Queue the task and return immediately
    index_document.delay(document_id=document_id)
    return {"status": "queued", "document_id": document_id}
```

**Benefits of asynchronous processing:**
- Instant response to users
- Scalable task processing (add more workers)
- Automatic retry on failure
- Better resource utilization
- Progress tracking and monitoring

### مزايا المعالجة غير المتزامنة:
- استجابة فورية للمستخدمين
- معالجة المهام القابلة للتوسع (إضافة المزيد من العاملين)
- إعادة محاولة تلقائية عند الفشل
- استغلال أفضل للموارد
- تتبع التقدم والمراقبة

---

## Celery Architecture / معمارية Celery

### Components / المكونات

Celery consists of three main components:

1. **Broker / الوسيط**: A message broker that holds task messages
   - Redis (most common for simple setups)
   - RabbitMQ (production-grade, more features)
   - Amazon SQS, Azure Service Bus (cloud-native)

2. **Worker / العامل**: Processes tasks from the queue
   - Runs as separate processes
   - Can be scaled horizontally
   - Handles task execution and retries

3. **Backend / الخلفية**: Stores task results (optional)
   - Redis, Memcached, or database
   - Used for tracking task status
   - Can retrieve task results later

### Celery يتكون من ثلاثة مكونات رئيسية:

1. **الوسيط**: وسيط رسائل يحمل رسائل المهام
   - Redis (الأكثر شيوعًا للإعدادات البسيطة)
   - RabbitMQ (لإنتاج، ميزات أكثر)
   - Amazon SQS، Azure Service Bus (سحابية)

2. **العامل**: يعالج المهام من الطابور
   - يعمل كعمليات منفصلة
   - يمكن توسيعها أفقيًا
   - يتعامل مع تنفيذ المهام وإعادة المحاولة

3. **الخلفية**: تخزن نتائج المهام (اختياري)
   - Redis أو Memcached أو قاعدة بيانات
   - تُستخدم لتتبع حالة المهام
   - يمكن استرجاع نتائج المهام لاحقًا

### Task Flow / تدفق المهام

```
┌─────────────┐      ┌─────────┐      ┌─────────┐      ┌──────────┐
│  FastAPI    │ ───> │  Broker │ ───> │ Worker  │ ───> │ Backend  │
│  Request    │      │ (Redis) │      │ Process │      │ (Redis)  │
└─────────────┘      └─────────┘      └─────────┘      └──────────┘
      │                   │                  │
      │ delay()           │ pull()           │ store_result()
      │                   │                  │
      └───────────────────┴──────────────────┴────────────┘
```

### Message Flow / تدفق الرسائل

1. **Task Submission / تقديم المهمة**:
   ```python
   from src.workers.tasks import index_document
   index_document.delay(document_id="doc123")
   # Or with explicit args
   index_document.apply_async(args=["doc123"], countdown=10)
   ```

2. **Broker Storage / تخزين الوسيط**:
   - Task serialized to JSON/MsgPack
   - Stored in Redis queue (list data structure)
   - Key format: `celery` (default queue name)

3. **Worker Processing / معالجة العامل**:
   - Worker polls broker continuously
   - Deserializes task message
   - Executes task function with provided arguments
   - Handles exceptions and retries

4. **Result Storage / تخزين النتيجة**:
   - Task result stored in backend (if configured)
   - Status tracked: `PENDING` → `STARTED` → `SUCCESS`/`FAILURE`
   - Can be retrieved by task ID

### تدفق الرسائل

1. **تقديم المهمة**:
   - تُسلسل المهمة إلى JSON/MsgPack
   - تُخزن في طابور Redis (بنية قائمة)
   - تنسيق المفتاح: `celery` (اسم الطابور الافتراضي)

2. **معالجة العامل**:
   - يستقصي العامل الوسيط باستمرار
   - يُفك تسلسل رسالة المهمة
   - ينفذ وظيفة المهمة بالحجج المقدمة
   - يتعامل مع الاستثناءات وإعادة المحاولة

3. **تخزين النتيجة**:
   - تُخزن نتيجة المهمة في الخلفية (إذا تم تكوينها)
   - تُتتبع الحالة: `PENDING` → `STARTED` → `SUCCESS`/`FAILURE`
   - يمكن استرجاعها بمعرف المهمة

---

## Task Design Patterns / أنماط تصميم المهام

### 1. Idempotent Tasks / المهام المعرفية

**Definition / التعريف**: A task can be run multiple times with the same inputs without causing unintended side effects.

**Why important? / لماذا مهمة؟**:
- Automatic retries on failure
- Duplicate task messages (network issues)
- Manual re-queuing

**Example / مثال**:
```python
# BAD: Non-idempotent (causes duplicates on retry)
@celery_app.task
def create_user(email: str):
    user = db.add(email=email)  # Fails silently on duplicate

# GOOD: Idempotent (checks before creating)
@celery_app.task
def create_user(email: str):
    if not db.exists(email):
        user = db.add(email=email)
        return user.id
    else:
        return db.get(email).id
```

**التعريف**: مهمة يمكن تشغيلها عدة مرات بنفس المدخلات دون التسبب في آثار جانبية غير مقصودة.

**لماذا مهمة؟**:
- إعادة المحاولة التلقائية عند الفشل
- رسائل مهام مكررة (مشاكل الشبكة)
- إعادة الطابور يدويًا

### 2. Atomic Operations / العمليات الذرية

**Definition / التعريف**: Operations that either complete entirely or fail completely, never leaving partial state.

**Pattern / النمط**:
```python
@celery_app.task
def process_document(document_id: str):
    # Use database transactions for atomicity
    with db.transaction():
        # Step 1: Mark as processing
        doc = db.get(document_id)
        doc.status = "processing"

        # Step 2: Process
        result = expensive_operation(doc)

        # Step 3: Update status
        doc.status = "completed"
        doc.result = result

    # If any step fails, entire transaction rolls back
```

**التعريف**: عمليات تكتمل بالكامل أو تفشل تمامًا، ولا تترك حالة جزئية أبدًا.

### 3. Task Dependencies / تبعيات المهام

Celery supports chaining and grouping tasks:

```python
from celery import chain, chord, group

# Chain: Execute tasks sequentially
result = chain(
    extract_text.s(document_id),
    generate_chunks.s(),
    create_embeddings.s()
).apply_async()

# Group: Execute tasks in parallel
result = group(
    index_document.s(doc_id1),
    index_document.s(doc_id2),
    index_document.s(doc_id3)
).apply_async()

# Chord: Group + callback
callback = summarize_results.s()
result = chord(
    [index_document.s(doc_id) for doc_id in doc_ids],
    callback
).apply_async()
```

يدعم Celery ربط وتجميع المهام:

### 4. Progress Tracking / تتبع التقدم

For long-running tasks, provide progress updates:

```python
@celery_app.task(bind=True)
def long_running_task(self, total_items: int):
    for i, item in enumerate(items):
        process_item(item)

        # Update progress
        progress = (i + 1) / total_items * 100
        self.update_state(
            state="PROGRESS",
            meta={"current": i + 1, "total": total_items, "percent": progress}
        )

    return {"status": "completed"}
```

Retrieve progress:
```python
task = long_running_task.AsyncResult(task_id)
if task.state == "PROGRESS":
    print(f"Progress: {task.info['percent']}%")
```

للمهام طويلة الأمد، قدم تحديثات التقدم:

---

## Implemented Tasks / المهام المُنفذة

### 1. Bulk Upload Documents / رفع المستندات بالجملة

**Purpose / الغرض**: Upload multiple documents efficiently without blocking the API.

**File / الملف**: `src/workers/tasks.py` - `bulk_upload_documents`

**Key Features / الميزات الرئيسية**:
- Sequential processing (maintains order)
- Individual error handling per document
- Queues indexing tasks automatically
- Metrics tracking with Prometheus
- Bilingual error messages

**Code Example / مثال الكود**:
```python
@celery_app.task(
    name="bulk_upload_documents",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def bulk_upload_documents(
    self,
    *,
    tenant_id: str,
    files: List[Dict[str, Any]],
) -> dict:
    """
    Process multiple document uploads in bulk.

    Args:
        tenant_id: Tenant/user ID
        files: List of file dicts with keys:
            - filename: Original filename
            - content_type: MIME type
            - content: File bytes

    Returns:
        Dict with success/failure counts and results
    """
    container = get_container()
    file_store = container["file_store"]
    document_repo = container["document_repo"]

    results = []
    success_count = 0
    failure_count = 0

    for file_info in files:
        try:
            # Store file
            stored = await file_store.save_upload(...)

            # Create document record
            document_id = document_repo.create_document(...)

            # Queue indexing task
            index_document.delay(document_id=document_id.value)

            results.append({"filename": file_info["filename"], "status": "queued"})
            success_count += 1

        except Exception as e:
            results.append({
                "filename": file_info.get("filename", "unknown"),
                "status": "failed",
                "error": str(e),
            })
            failure_count += 1

    return {
        "total": len(files),
        "success": success_count,
        "failures": failure_count,
        "results": results,
    }
```

**الغرض**: رفع مستندات متعددة بكفاءة دون حظر API.

**استدعاء المهمة**:
```python
from src.workers.tasks import bulk_upload_documents

task = bulk_upload_documents.delay(
    tenant_id="tenant123",
    files=[
        {"filename": "doc1.pdf", "content_type": "application/pdf", "content": b"..."},
        {"filename": "doc2.pdf", "content_type": "application/pdf", "content": b"..."},
    ]
)

# Check status later
result = task.get(timeout=300)
```

### 2. Bulk Delete Documents / حذف المستندات بالجملة

**Purpose / الغرض**: Delete multiple documents with cascade cleanup.

**File / الملف**: `src/workers/tasks.py` - `bulk_delete_documents`

**Key Features / الميزات الرئيسية**:
- Cascade deletion (chunks, vectors, records)
- Per-document error handling
- Atomic operations per document
- Metrics tracking

**Cascade Deletion / الحذف المتسلسل**:
```
Document Record
    ├── File (storage)
    ├── Chunks (database)
    │   └── Embeddings (cache)
    └── Vectors (Qdrant)
```

**Code Pattern / نمط الكود**:
```python
@celery_app.task
def bulk_delete_documents(*, tenant_id: str, document_ids: List[str]):
    container = get_container()
    document_repo = container["document_repo"]
    vector_store = container["vector_store"]
    chunk_repo = container["chunk_repo"]

    for doc_id_str in document_ids:
        try:
            doc_id = DocumentId(doc_id_str)

            # Delete from vector store (Qdrant)
            vector_store.delete_points(tenant_id=tenant, document_id=doc_id)

            # Delete chunks from database
            chunk_repo.delete_by_document(tenant_id=tenant, document_id=doc_id)

            # Delete document record (cascades to file if needed)
            document_repo.delete_document(tenant_id=tenant, document_id=doc_id)

        except Exception as e:
            # Continue with next document
            logger.error(f"Delete failed for {doc_id_str}", error=str(e))
```

**الغرض**: حذف مستندات متعددة مع تنظيف متسلسل.

### 3. PDF Merge / دمج PDF

**Purpose / الغرض**: Merge multiple PDF documents into a single file.

**File / الملف**: `src/workers/tasks.py` - `merge_pdfs`

**Key Features / الميزات الرئيسية**:
- Page-by-page merging algorithm
- Supports creating new or updating existing documents
- Automatic indexing after merge
- Storage integration

**Algorithm / الخوارزمية**:
```python
@celery_app.task
def merge_pdfs(
    *,
    tenant_id: str,
    source_document_ids: List[str],
    merged_filename: str,
    target_document_id: str | None = None,
):
    # 1. Retrieve source documents
    source_docs = []
    for doc_id_str in source_document_ids:
        stored_file = document_reader.get_stored_file(...)
        pdf_reader = PyPDF2.PdfReader(stored_file.path)
        source_docs.append({"reader": pdf_reader, "filename": stored_file.filename})

    # 2. Create merged PDF (page-by-page)
    merged_writer = PyPDF2.PdfWriter()
    for source in source_docs:
        for page_num in range(len(source["reader"].pages)):
            merged_writer.add_page(source["reader"].pages[page_num])

    # 3. Write to bytes
    merged_bytes = BytesIO()
    merged_writer.write(merged_bytes)
    merged_content = merged_bytes.getvalue()

    # 4. Store and create/update document
    stored = await file_store.save_upload(...)
    file_hash = hashlib.sha256(merged_content).hexdigest()

    if target_document_id:
        # Update existing
        document_repo.update_document(...)
    else:
        # Create new
        new_id = document_repo.create_document(...)

    # 5. Queue indexing task
    index_document.delay(document_id=merged_document_id)

    return {"merged_document_id": merged_document_id, "source_count": len(source_document_ids)}
```

**الغرض**: دمج مستندات PDF متعددة في ملف واحد.

### 4. Chat Title Generation / توليد عناوين المحادثات

**Purpose / الغرض**: Auto-generate descriptive titles for chat sessions.

**File / الملف**: `src/workers/tasks.py` - `generate_chat_title`

**Key Features / الميزات الرئيسية**:
- LLM-powered title generation
- Uses first 3 turns for context
- Title length limited to 50 chars
- Updates session record in database

**Prompt Design / تصميم المطالبة**:
```python
prompt = f"""Generate a concise title (max 50 characters) for this chat session:

{context}

Guidelines:
- Use the main topic discussed
- Keep it short and descriptive
- Focus on what the user was asking about
- Do NOT include the word "Chat" or "Session"
- Provide ONLY the title, nothing else

Title:"""
```

**الغرض**: توليد عناوين وصفية تلقائيًا لجلسات المحادثة.

### 5. Chat Session Summary / ملخص جلسة المحادثة

**Purpose / الغرض**: Generate comprehensive summaries of completed chat sessions.

**File / الملف**: `src/workers/tasks.py` - `summarize_chat_session`

**Key Features / الميزات الرئيسية**:
- LLM-powered summarization
- Extracts main topics (max 5)
- Sentiment analysis (positive/neutral/negative)
- Updates session record with summary, topics, sentiment

**Output Format / تنسيق الإخراج**:
```python
{
    "summary": "Discussion of RAG architecture components",
    "topics": ["RAG", "Vector databases", "Embeddings"],
    "sentiment": "positive",
    "question_count": 12,
    "status": "success"
}
```

**الغرض**: إنشاء ملخصات شاملة لجلسات المحادثة المكتملة.

---

## Best Practices / أفضل الممارسات

### 1. Task Signature / توقيع المهمة

**Always use keyword arguments / استخدم دائمًا الحجج المسمية**:
```python
# GOOD
@celery_app.task
def process_document(*, tenant_id: str, document_id: str):
    ...

# BAD (order-dependent, error-prone)
@celery_app.task
def process_document(tenant_id: str, document_id: str):
    ...
```

### 2. Error Handling / معالجة الأخطاء

**Use `bind=True` for task instance / استخدم `bind=True` لمثيل المهمة**:
```python
@celery_app.task(bind=True)
def my_task(self, arg1):
    try:
        result = expensive_operation(arg1)
    except Exception as e:
        # Log and retry
        logger.error("Task failed", error=str(e))
        raise self.retry(exc=e, countdown=60)
```

### 3. Retry Strategy / استراتيجية إعادة المحاولة

**Configure retries appropriately / تكوين عمليات إعادة المحاولة بشكل مناسب**:
```python
@celery_app.task(
    autoretry_for=(ConnectionError, TimeoutError),  # Only retry on these
    retry_backoff=True,  # Exponential backoff
    retry_kwargs={"max_retries": 3},
    retry_jitter=True,  # Randomize to avoid thundering herd
)
def network_task():
    ...
```

### 4. Timeouts / المهلات

**Set timeouts to prevent hanging tasks / اضبط مهلات لمنع مهام معلقة**:
```python
@app.task(time_limit=600)  # Hard limit (kills task)
@app.task.soft_time_limit=550)  # Soft limit (raises exception)
def long_task():
    ...
```

### 5. Task Naming / تسمية المهام

**Use descriptive, unique names / استخدم أسماء وصفية وفريدة**:
```python
@celery_app.task(name="index_document_v2")  # Explicit name
def index_document():
    ...
```

### 6. Resource Cleanup / تنظيف الموارد

**Ensure proper cleanup / تأكد من التنظيف الصحيح**:
```python
@app.task
def process_with_file(file_path: str):
    try:
        with open(file_path, 'rb') as f:
            data = process(f.read())
        return data
    finally:
        # Ensure cleanup even on failure
        if os.path.exists(file_path):
            os.remove(file_path)
```

---

## Common Pitfalls / الأخطاء الشائعة

### 1. Blocking Operations / العمليات الحاجبة

**Problem / المشكلة**: Synchronous I/O blocks the worker process.

**Solution / الحل**: Use async/await or non-blocking libraries:
```python
# BAD
@app.task
def slow_task():
    time.sleep(10)  # Blocks worker for 10 seconds

# GOOD
@app.task
def async_task():
    asyncio.run(async_operation())  # Non-blocking
```

### 2. Large Arguments / الحجج الكبيرة

**Problem / المشكلة**: Large arguments slow down serialization and broker operations.

**Solution / الحل**: Pass IDs, not data:
```python
# BAD (large payload)
@app.task
def process_large_data(data: bytes):  # 100MB payload
    ...

# GOOD (small payload)
@app.task
def process_by_id(document_id: str):
    data = db.load(document_id)  # Load when needed
    ...
```

### 3. Database Session Issues / مشاكل جلسة قاعدة البيانات

**Problem / المشكلة**: Reusing database sessions across task executions.

**Solution / الحل**: Create new session per task:
```python
@app.task
def database_task():
    # Create new session for this task
    with SessionLocal() as db:
        result = db.query(...)
    # Session closed automatically
```

### 4. Missing Result Backend / عدم وجود خلفية النتائج

**Problem / المشكلة**: Can't track task status or retrieve results.

**Solution / الحل**: Configure result backend:
```python
# celery_app.py
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1',  # Add this!
)
```

### 5. Infinite Loops / الحلقات اللانهائية

**Problem / المشكلة**: Task never completes, blocking worker.

**Solution / الحل**: Always use iteration limits:
```python
# BAD
@app.task
def infinite_task():
    while True:
        process(item)  # Never stops

# GOOD
@app.task
def limited_task(max_items: int = 1000):
    for i, item in enumerate(items):
        if i >= max_items:
            break
        process(item)
```

---

## Monitoring and Observability / المراقبة والقابلية للملاحظة

### Prometheus Metrics / مقاييس Prometheus

The RAG engine integrates Celery task metrics with Prometheus:

```python
from src.core.observability import CELERY_TASK_COUNT, CELERY_TASK_DURATION

@celery_app.task
def monitored_task():
    start_time = time.time()
    try:
        result = do_work()
        CELERY_TASK_COUNT.labels(task="monitored_task", status="success").inc()
        CELERY_TASK_DURATION.labels(task="monitored_task").observe(time.time() - start_time)
        return result
    except Exception as e:
        CELERY_TASK_COUNT.labels(task="monitored_task", status="failure").inc()
        CELERY_TASK_DURATION.labels(task="monitored_task").observe(time.time() - start_time)
        raise
```

**Metrics exposed / المقاييس المكشوفة**:
- `celery_task_count_total` - Total task executions by task name and status
- `celery_task_duration_seconds` - Task execution time histogram

### Structured Logging / التسجيل المهيكل

```python
import structlog

log = structlog.get_logger()

@celery_app.task
def logged_task(document_id: str):
    log.info("task_started", document_id=document_id)
    try:
        result = process(document_id)
        log.info("task_completed", document_id=document_id, result_count=len(result))
        return result
    except Exception as e:
        log.error("task_failed", document_id=document_id, error=str(e))
        raise
```

### Flower Monitoring / مراقبة Flower

Flower is a web-based tool for monitoring and administrating Celery clusters:

```bash
pip install flower
celery -A src.workers.celery_app flower --port=5555
```

**Features / الميزات**:
- Real-time task monitoring
- Worker status and statistics
- Task inspection (arguments, results, traceback)
- Task management (revoke, kill)
- Broker monitoring

**ميزات**:
- مراقبة المهام في الوقت الفعلي
- حالة وإحصائيات العامل
- فحص المهام (الحجج، النتائج، التتبع)
- إدارة المهام (إلغاء، قتل)
- مراقبة الوسيط

### Monitoring / المراقبة

---

## Summary / الملخص

### Key Takeaways / النقاط الرئيسية

1. **Background tasks are essential** for modern web applications
   - Non-blocking user experience
   - Scalable processing
   - Resilience through retries

2. **Celery provides a robust solution** for task queuing
   - Flexible broker choices (Redis, RabbitMQ)
   - Task chaining and grouping
   - Built-in retry mechanisms

3. **Design patterns matter**:
   - Idempotent tasks
   - Atomic operations
   - Progress tracking
   - Error handling

4. **Implementation highlights**:
   - Bulk operations (upload, delete)
   - PDF merging
   - Chat enhancements (title, summary)
   - Comprehensive monitoring

5. **Best practices**:
   - Use keyword arguments
   - Configure retries appropriately
   - Set timeouts
   - Clean up resources
   - Monitor with Prometheus and Flower

### النقاط الرئيسية

1. **مهام الخلفية ضرورية** لتطبيقات الويب الحديثة
   - تجربة مستخدم غير حظر
   - معالجة قابلة للتوسع
   - المرونة من خلال إعادة المحاولة

2. **يوفر Celery حلاً قويًا** لصف المهام
   - اختيارات وسيط مرنة (Redis و RabbitMQ)
   - ربط وتجميع المهام
   - آليات إعادة محاولة مدمجة

3. **أنماط التصميم مهمة**:
   - مهام معرفية
   - عمليات ذرية
   - تتبع التقدم
   - معالجة الأخطاء

4. **أبرز التنفيذ**:
   - العمليات بالجملة (رفع، حذف)
   - دمج PDF
   - تحسينات المحادثة (العنوان، الملخص)
   - مراقبة شاملة

5. **أفضل الممارسات**:
   - استخدم الحجج المسمية
   - تكوين عمليات إعادة المحاولة بشكل مناسب
   - اضبط المهلات
   - نظف الموارد
   - راقب باستخدام Prometheus و Flower

---

## Further Reading / قراءة إضافية

- [Celery Documentation](https://docs.celeryq.dev/)
- [Redis as Celery Broker](https://docs.celeryq.dev/en/stable/userguide/configuration.html#broker-settings)
- [Flower Monitoring Tool](https://flower.readthedocs.io/)
- [Best Practices for Long-Running Tasks](https://docs.celeryq.dev/en/stable/userguide/tasks.html#best-practices)

---

## Arabic Summary / ملخص بالعربية

هذا الدليل يغطي مفهوم مهام الخلفية وتنفيذها باستخدام Celery في محرك RAG. تشمل المواضيع الرئيسية:

1. **لماذا مهام الخلفية**: لمعالجة العمليات طويلة الأمد دون حظر طلبات HTTP
2. **معمارية Celery**: الوسيط (Redis)، العامل، والخلفية
3. **المهام المُنفذة**: رفع وحذف المستندات بالجملة، دمج PDF، توليد عناوين المحادثات، تلخيص الجلسات
4. **أنماط التصميم**: المهام المعرفية، العمليات الذرية، تبعيات المهام، تتبع التقدم
5. **أفضل الممارسات**: توقيع المهام، معالجة الأخطاء، استراتيجية إعادة المحاولة، تنظيف الموارد
6. **المراقبة**: مقاييس Prometheus، التسجيل المهيكل، Flower

تمثل هذه المهام أساس معالجة البيانات غير المتزامنة في نظام RAG، مما يسمح بالعمليات القابلة للتوسع والموثوقية العالية.
