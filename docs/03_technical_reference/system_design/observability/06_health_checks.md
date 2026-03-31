# Health Checks and Observability
# فحوصات الصحة والقابلية للملاحظة

## Table of Contents / جدول المحتويات

1. [Introduction / مقدمة](#introduction)
2. [Why Health Checks? / لماذا فحوصات الصحة؟](#why-health-checks)
3. [Types of Health Checks / أنواع فحوصات الصحة](#types-of-health-checks)
4. [Kubernetes Integration / التكامل مع Kubernetes](#kubernetes-integration)
5. [Implementation Details / تفاصيل التنفيذ](#implementation-details)
6. [Best Practices / أفضل الممارسات](#best-practices)
7. [Common Pitfalls / الأخطاء الشائعة](#common-pitfalls)
8. [Summary / الملخص](#summary)

---

## Introduction / مقدمة

### What are Health Checks? / ما هي فحوصات الصحة؟

Health checks are endpoints that external systems use to verify that your application and its dependencies are functioning correctly. They serve as a contract between your application and orchestration systems like Kubernetes, load balancers, and monitoring tools.

**Key characteristics of good health checks:**
- Fast response times (typically < 100ms for liveness, < 1s for readiness)
- Clear status indicators (ok/degraded/error)
- Minimal resource usage
- Idempotent (can be called repeatedly without side effects)

**فحوصات الصحة** هي نقاط نهاية تستخدمها الأنظمة الخارجية للتحقق من أن تطبيقك وتبعياته تعمل بشكل صحيح. تعمل كعقد بين تطبيقك وأنظمة التنسيق مثل Kubernetes وأدوات المراقبة.

**الخصائص الرئيسية لفحوصات الصحة الجيدة:**
- أوقات استجابة سريعة (عادةً < 100ms للنشاط، < 1 ثانية للجاهزية)
- مؤشرات حالة واضحة (ok/degraded/error)
- استخدام موارد ضئيل
- معرفية (يمكن استدعاؤها بشكل متكرر دون آثار جانبية)

---

## Why Health Checks? / لماذا فحوصات الصحة؟

### 1. Container Orchestration / تنسيق الحاويات

In Kubernetes, health checks are essential for:

**Liveness Probes / فحوصات النشاط**:
- Restart containers that are deadlocked or stuck
- Prevent zombie containers from consuming resources
- Ensure the application is actually running

```yaml
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 3
```

**Readiness Probes / فحوصات الجاهزية**:
- Control traffic routing to pods
- Prevent requests from reaching unready containers
- Graceful startup of dependent services

```yaml
# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 2
```

في Kubernetes، فحوصات الصحة ضرورية لـ:

---

### 2. Load Balancing / موازنة الحمل

Health checks allow load balancers to:

- Remove unhealthy instances from rotation
- Distribute traffic only to healthy instances
- Prevent cascading failures
- Enable graceful degradation

**Example / مثال**:
```
┌─────────────┐
│ Load Balancer│
└──────┬──────┘
       │
       ├─── Instance 1 (❌ Unhealthy) → No traffic
       ├─── Instance 2 (✅ Healthy)  → Traffic
       └─── Instance 3 (✅ Healthy)  → Traffic
```

تسمح فحوصات الصحة لأجهزة موازنة الحمل بـ:

---

### 3. Monitoring and Alerting / المراقبة والتنبيه

Health checks provide a baseline for:

- Synthetic monitoring (external monitoring tools ping endpoints)
- Alerting on service degradation
- SLA/SLO compliance tracking
- Incident response workflows

**Example monitoring integration / مثال تكامل المراقبة**:
```python
# Prometheus alert on health check failures
- alert: ServiceUnhealthy
  expr: up{job="rag-engine"} == 0
  for: 2m
  annotations:
    summary: "Service {{ $labels.instance }} is down"
```

توفر فحوصات الصحة أساسًا لـ:

---

### 4. Observability / القابلية للملاحظة

Health checks contribute to observability by providing:

- Dependency status tracking
- Latency measurements
- Error rate monitoring
- Capacity planning data

**What can we observe? / ماذا يمكننا ملاحظته؟**
- Database connection pool status
- Redis connection health
- Qdrant cluster status
- LLM API availability
- File storage connectivity

تساهم فحوصات الصحة في القابلية للملاحظة من خلال توفير:

---

## Types of Health Checks / أنواع فحوصات الصحة

### 1. Liveness Check / فحص النشاط

**Purpose / الغرض**: Determine if the container needs to be restarted.

**Characteristics / الخصائص**:
- **Fast**: Must return within 1-2 seconds
- **Lightweight**: Minimal resource usage
- **Local**: No external dependencies
- **Binary**: OK or error (no partial states)

**When to restart? / متى يتم إعادة التشغيل؟**
- Deadlock conditions
- Infinite loops
- Memory leaks
- Unrecoverable errors

**Example / مثال**:
```python
@router.get("/health")
def health_check() -> dict:
    """
    Basic health check endpoint.

    Fast, lightweight check that verifies:
    - Application is running
    - Basic config loaded
    - No fatal errors

    Should return < 100ms
    """
    return {
        "status": "ok",
        "env": settings.env,
        "app_name": settings.app_name,
    }
```

**الغرض**: تحديد ما إذا كانت الحاوية تحتاج إلى إعادة تشغيل.

### 2. Readiness Check / فحص الجاهزية

**Purpose / الغرض**: Determine if the container can accept traffic.

**Characteristics / الخصائص**:
- **Dependency-aware**: Checks external dependencies
- **Reasonable timeout**: Can take up to 1-2 seconds
- **Partial states**: Can return degraded status
- **Graceful degradation**: Some services down is OK

**What to check? / ماذا يجب فحصه؟**
- Database connection (essential)
- Redis connection (can be degraded)
- Qdrant connection (essential for RAG)
- File storage (can be degraded)

**Example / مثال**:
```python
@router.get("/health/ready")
def readiness_check() -> dict:
    """
    Readiness check for Kubernetes/load balancers.

    Checks critical dependencies:
    - PostgreSQL database
    - Redis cache
    - Qdrant vector store

    Returns "ready": True if all critical checks pass.
    """
    checks = {
        "database": _check_postgres_connection(),
        "redis": _check_redis_connection(),
        "qdrant": _check_qdrant_connection(),
    }

    # Ready if all critical checks are not in error state
    ready = all(check["status"] != "error" for check in checks.values())

    return {
        "ready": ready,
        "checks": checks,
    }
```

**الغرض**: تحديد ما إذا كانت الحاوية يمكنها قبول حركة المرور.

### 3. Deep Health Check / فحص الصحة العميق

**Purpose / الغرض**: Comprehensive check of all system dependencies.

**Characteristics / الخصائص**:
- **Comprehensive**: All dependencies checked
- **Detailed**: Returns latency and error messages
- **Slower**: Can take 2-5 seconds
- **Not for orchestration**: For monitoring/admin only

**What to check? / ماذا يجب فحصه؟**
- All dependency connections
- Connection latency
- Actual data operations (read/write)
- Service-specific functionality

**Example / مثال**:
```python
@router.get("/health/deep")
def deep_health_check() -> dict:
    """
    Deep health check for all system dependencies.

    Includes:
    - PostgreSQL database (with latency)
    - Redis cache (with latency)
    - Qdrant vector store (with latency)
    - LLM provider (with generation test)
    - File storage (with read/write test)

    For monitoring and debugging, not orchestration.
    """
    checks = {
        "database": _check_postgres_connection(),
        "redis": _check_redis_connection(),
        "qdrant": _check_qdrant_connection(),
        "llm": _check_llm_connection(),
        "file_storage": _check_file_storage(),
    }

    # Overall status based on check results
    statuses = [check["status"] for check in checks.values()]

    if "error" in statuses:
        overall_status = "error"
    elif "degraded" in statuses:
        overall_status = "degraded"
    else:
        overall_status = "ok"

    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": time.time(),
    }
```

**الغرض**: فحص شامل لجميع تبعيات النظام.

---

## Kubernetes Integration / التكامل مع Kubernetes

### Liveness Probe Configuration / تكوين فحص النشاط

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-engine
spec:
  template:
    spec:
      containers:
      - name: rag-engine
        image: rag-engine:latest
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30  # Give app time to start
          periodSeconds: 10      # Check every 10 seconds
          timeoutSeconds: 5       # Fail if no response in 5s
          failureThreshold: 3    # Restart after 3 failures
```

**Configuration notes / ملاحظات التكوين**:
- `initialDelaySeconds`: How long to wait before first check
- `periodSeconds`: How often to check
- `timeoutSeconds`: How long to wait for response
- `failureThreshold`: How many failures before restart

### Readiness Probe Configuration / تكوين فحص الجاهزية

```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5   # Check sooner than liveness
  periodSeconds: 5        # Check more frequently
  timeoutSeconds: 3        # Shorter timeout
  failureThreshold: 2      # Mark unready after 2 failures
```

**Key differences / الاختلافات الرئيسية**:
- Faster to become ready (shorter delay)
- More frequent checks
- Shorter timeout (fails fast)
- Lower failure threshold (conservative)

### Startup Probe (Optional) / فحص البدء (اختياري)

```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 0
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 30  # 5s * 30 = 150s max startup time
```

**When to use startup probes? / متى تستخدم فحوصات البدء؟**
- Applications with long startup times (> 30s)
- Applications that load large models/data
- Prevents liveness probes from killing slow-starting containers

**متى تستخدم فحوصات البدء؟**
- التطبيقات ذات أوقات بدء طويلة (> 30 ثانية)
- التطبيقات التي تحمل نماذج/بيانات كبيرة
- تمنع فحوصات النشاط من قتل الحاويات البطيئة في البدء

---

## Implementation Details / تفاصيل التنفيذ

### Database Health Check / فحص صحة قاعدة البيانات

```python
def _check_postgres_connection() -> Dict[str, Any]:
    """
    Check PostgreSQL database connection with latency measurement.

    Returns:
        Dict with status (ok/degraded/error), latency_ms, and message

    فحص اتصال قاعدة بيانات PostgreSQL
    """
    try:
        start_time = time.time()
        engine = create_engine(settings.database_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = (time.time() - start_time) * 1000

        # Status based on latency
        status = "ok" if latency_ms < 100 else "degraded"
        return {
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "message": "Database connection successful",
        }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": None,
            "message": f"Database connection failed: {str(e)}",
        }
```

**Key points / النقاط الرئيسية**:
- Uses `pool_pre_ping` to verify connections
- Measures actual query execution time
- Returns degraded status if slow but working
- Returns error if connection fails

**النقاط الرئيسية**:
- يستخدم `pool_pre_ping` للتحقق من الاتصالات
- يقيس وقت تنفيذ الاستعلام الفعلي
- يُرجع حالة degraded إذا كان بطيئًا لكنه يعمل
- يُرجع error إذا فشل الاتصال

### Redis Health Check / فحص صحة Redis

```python
def _check_redis_connection() -> Dict[str, Any]:
    """
    Check Redis connection with latency measurement.

    فحص اتصال Redis
    """
    try:
        start_time = time.time()
        client = redis.from_url(settings.redis_url, socket_timeout=2)
        client.ping()
        latency_ms = (time.time() - start_time) * 1000

        status = "ok" if latency_ms < 50 else "degraded"
        return {
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "message": "Redis connection successful",
        }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": None,
            "message": f"Redis connection failed: {str(e)}",
        }
```

**Key points / النقاط الرئيسية**:
- Uses `PING` command (fastest check)
- Sets socket timeout to avoid hanging
- Stricter latency threshold (< 50ms)
- Connection pooling handled by redis-py

### Qdrant Health Check / فحص صحة Qdrant

```python
def _check_qdrant_connection() -> Dict[str, Any]:
    """
    Check Qdrant vector database connection.

    فحص اتصال قاعدة بيانات المتجهات Qdrant
    """
    try:
        start_time = time.time()
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=2,
        )
        client.get_collections()
        latency_ms = (time.time() - start_time) * 1000

        status = "ok" if latency_ms < 100 else "degraded"
        return {
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "message": "Qdrant connection successful",
        }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": None,
            "message": f"Qdrant connection failed: {str(e)}",
        }
```

**Key points / النقاط الرئيسية**:
- Calls `get_collections()` to verify API access
- Sets timeout to avoid hanging
- Verifies authentication with API key
- Can distinguish network vs. authentication errors

### LLM Health Check / فحص صحة LLM

```python
def _check_llm_connection() -> Dict[str, Any]:
    """
    Check LLM provider availability by performing a simple generation test.

    فحص توفر مزود نموذج اللغة
    """
    try:
        from src.core.bootstrap import get_container

        start_time = time.time()
        container = get_container()
        llm = container["llm"]

        # Simple test generation
        result = llm.generate("Test", max_tokens=5, timeout=5)
        latency_ms = (time.time() - start_time) * 1000

        if result and len(result) > 0:
            status = "ok" if latency_ms < 2000 else "degraded"
            return {
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "message": f"LLM connection successful ({settings.llm_backend})",
            }
        else:
            return {
                "status": "error",
                "latency_ms": round(latency_ms, 2),
                "message": "LLM returned empty response",
            }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": None,
            "message": f"LLM connection failed ({settings.llm_backend}): {str(e)}",
        }
```

**Key points / النقاط الرئيسية**:
- Performs actual generation (not just connection check)
- Uses very short prompt and max_tokens
- Sets timeout to prevent hanging
- Distinguishes between connection and generation errors

### File Storage Health Check / فحص صحة التخزين

```python
def _check_file_storage() -> Dict[str, Any]:
    """
    Check file storage backend connectivity with write/read test.

    فحص تخزين الملفات
    """
    try:
        import uuid
        import os

        start_time = time.time()
        file_store = create_file_store(settings)
        test_filename = f"health_check_{uuid.uuid4().hex[:16]}.txt"
        test_content = b"health_check_test"

        # Write test
        async def write_read_test():
            stored = await file_store.save_upload(
                tenant_id="health_check",
                upload_filename=test_filename,
                content_type="text/plain",
                data=test_content,
            )
            # Read test
            with open(stored.path, "rb") as f:
                content = f.read()
            # Cleanup
            os.remove(stored.path)
            return content

        content = asyncio.run(write_read_test())
        latency_ms = (time.time() - start_time) * 1000

        if content == test_content:
            status = "ok" if latency_ms < 500 else "degraded"
            return {
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "message": f"File storage successful ({settings.filestore_backend})",
            }
        else:
            return {
                "status": "error",
                "latency_ms": round(latency_ms, 2),
                "message": "File storage read/write mismatch",
            }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": None,
            "message": f"File storage failed ({settings.filestore_backend}): {str(e)}",
        }
```

**Key points / النقاط الرئيسية**:
- Performs actual write/read operations
- Cleans up test file after check
- Uses unique filename to avoid collisions
- Works with any storage backend (S3, GCS, Azure, local)

---

## Best Practices / أفضل الممارسات

### 1. Use Appropriate Endpoints / استخدم نقاط النهاية المناسبة

| Check Type | Endpoint | Use For |
|-----------|----------|---------|
| Liveness | `/health` | Container restart decisions |
| Readiness | `/health/ready` | Traffic routing decisions |
| Deep | `/health/deep` | Monitoring and debugging |

| نوع الفحص | نقطة النهاية | تستخدم لـ |
|-----------|--------------|----------|
| النشاط | `/health` | قرارات إعادة تشغيل الحاوية |
| الجاهزية | `/health/ready` | قرارات توجيه الحركة |
| عميق | `/health/deep` | المراقبة والتصحيح |

### 2. Set Appropriate Thresholds / اضبط العتبات المناسبة

```python
# Latency thresholds based on dependency
DATABASE_THRESHOLD_MS = 100    # Database can be slower
REDIS_THRESHOLD_MS = 50       # Cache should be fast
QDRANT_THRESHOLD_MS = 100     # Vector DB moderate speed
LLM_THRESHOLD_MS = 2000       # LLM can be slow
STORAGE_THRESHOLD_MS = 500    # Storage moderate speed
```

### 3. Graceful Degradation / التدهور المشرف

Allow partial functionality:

```python
# Example: Ready even if file storage is degraded
ready = all(
    check["status"] != "error"
    for check in ["database", "redis", "qdrant"]
)
# File storage being degraded is acceptable
```

اسمح بالوظائف الجزئية:

### 4. Minimal State / حالة ضئيلة

Don't maintain state between health checks:

```python
# BAD: Maintains state
class HealthChecker:
    def __init__(self):
        self._last_check = None
        self._cached_status = None

# GOOD: Stateless
def health_check():
    # Always check fresh
    return _check_dependencies()
```

لا تحافظ على الحالة بين فحوصات الصحة:

### 5. Timeouts and Circuit Breaking / المهلات وقواطع الدائرة

Set timeouts and implement circuit breaking:

```python
import asyncio
from functools import wraps

def with_timeout(timeout_sec: int):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout_sec)
            except asyncio.TimeoutError:
                return {"status": "error", "message": "Timeout"}
        return wrapper
    return decorator

@with_timeout(timeout_sec=2)
async def check_slow_dependency():
    return await expensive_operation()
```

اضبط المهلات ونفذ قواطع الدائرة:

### 6. Caching for Expensive Checks / التخزين المؤقت للفحوصات المكلفة

Cache expensive check results for short periods:

```python
from functools import lru_cache
import time

@lru_cache(maxsize=32)
def _cached_llm_check(timestamp: int) -> Dict[str, Any]:
    """Cache LLM check for 30 seconds"""
    return _check_llm_connection()

def llm_check() -> Dict[str, Any]:
    """LLM check with 30-second cache"""
    cache_key = int(time.time() // 30)
    return _cached_llm_check(cache_key)
```

خزن مؤقتًا نتائج الفحوصات المكلفة لفترات قصيرة:

---

## Common Pitfalls / الأخطاء الشائعة

### 1. Too Aggressive Checks / فحوصات عدوانية جدًا

**Problem / المشكلة**: Checking too frequently overwhelms dependencies.

**Solution / الحل**: Use appropriate intervals:
```yaml
# BAD: Too frequent
periodSeconds: 1

# GOOD: Appropriate interval
periodSeconds: 10  # Every 10 seconds
```

### 2. Cascading Failures / الفشل المتسلسل

**Problem / المشكلة**: Health check failure causes cascade.

**Example / مثال**:
```
LLM API down → Health check fails → All pods restarted → Database overwhelmed → More failures
```

**Solution / الحل**: Separate liveness from readiness:
- Liveness: Only restart if app is truly dead
- Readiness: Can handle some dependencies being down

### 3. False Positives / إيجابيات كاذبة

**Problem / المشكلة**: Checks pass but service is broken.

**Example / مثال**:
```python
# BAD: Just returns OK without checking
def health_check():
    return {"status": "ok"}
```

**Solution / الحل**: Actually test functionality:
```python
# GOOD: Tests actual functionality
def health_check():
    if not can_connect_to_db():
        return {"status": "error", "message": "Database unavailable"}
    return {"status": "ok"}
```

### 4. Blocking Checks / فحوصات حاجبة

**Problem / المشكلة**: Checks block request handling.

**Solution / الحل**: Use timeouts and async operations:
```python
# BAD: Blocks indefinitely
def health_check():
    result = slow_operation()  # May hang
    return {"status": "ok" if result else "error"}

# GOOD: Uses timeout
def health_check():
    try:
        result = asyncio.wait_for(slow_operation(), timeout=2)
        return {"status": "ok" if result else "error"}
    except asyncio.TimeoutError:
        return {"status": "degraded", "message": "Timeout"}
```

### 5. Missing Context / سياق مفقود

**Problem / المشكلة**: Check fails without explanation.

**Solution / الحل**: Provide detailed error messages:
```python
# BAD
return {"status": "error"}

# GOOD
return {
    "status": "error",
    "message": "Database connection failed: connection refused",
    "latency_ms": None,
    "error_details": str(e),
}
```

---

## Summary / الملخص

### Key Takeaways / النقاط الرئيسية

1. **Health checks are critical** for production systems
   - Container orchestration
   - Load balancing
   - Monitoring and alerting
   - Observability

2. **Three types of checks**:
   - Liveness: Is the container alive?
   - Readiness: Can the container handle traffic?
   - Deep: Are all dependencies healthy?

3. **Implementation highlights**:
   - PostgreSQL: Connection test with latency measurement
   - Redis: PING command with fast response
   - Qdrant: Collection access verification
   - LLM: Generation test to verify API
   - Storage: Write/read test with cleanup

4. **Best practices**:
   - Use appropriate endpoints for each use case
   - Set sensible latency thresholds
   - Allow graceful degradation
   - Maintain minimal state
   - Implement timeouts and circuit breaking
   - Cache expensive checks

5. **Kubernetes integration**:
   - Liveness probes for restarts
   - Readiness probes for traffic routing
   - Startup probes for slow-starting containers
   - Proper configuration of delays, periods, and thresholds

### النقاط الرئيسية

1. **فحوصات الصحة حرجة** للأنظمة الإنتاجية
   - تنسيق الحاويات
   - موازنة الحمل
   - المراقبة والتنبيه
   - القابلية للملاحظة

2. **ثلاثة أنواع من الفحوصات**:
   - النشاط: هل الحاوية حية؟
   - الجاهزية: هل يمكن للحاوية التعامل مع الحركة؟
   - عميق: هل جميع التبعيات صحية؟

3. **أبرز التنفيذ**:
   - PostgreSQL: اختبار اتصال مع قياس زمن الانتقال
   - Redis: أمر PING مع استجابة سريعة
   - Qdrant: التحقق من الوصول إلى المجموعات
   - LLM: اختبار التوليد للتحقق من API
   - التخزين: اختبار كتابة/قراءة مع التنظيف

4. **أفضل الممارسات**:
   - استخدم نقاط النهاية المناسبة لكل حالة استخدام
   - اضبط عتبات زمن انتقال معقولة
   - اسمح بالتدهور المشرف
   - حافظ على حالة ضئيلة
   - نفذ المهلات وقواطع الدائرة
   - خزن مؤقتًا الفحوصات المكلفة

5. **التكامل مع Kubernetes**:
   - فحوصات النشاط لإعادة التشغيل
   - فحوصات الجاهزية لتوجيه الحركة
   - فحوصات البدء للحاويات البطيئة
   - تكوين مناسب للتأخيرات والفترات والعتبات

---

## Further Reading / قراءة إضافية

- [Kubernetes Probes Documentation](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [Health Check Best Practices](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#container-probes)
- [Monitoring Patterns](https://sre.google/workbook/alerting-on-slos/)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)

---

## Arabic Summary / ملخص بالعربية

هذا الدليل يغطي فحوصات الصحة والقابلية للملاحظة في محرك RAG. تشمل المواضيع الرئيسية:

1. **ما هي فحوصات الصحة**: نقاط نهاية للتحقق من أن التطبيق وتبعياته تعمل بشكل صحيح
2. **أنواع الفحوصات**: النشاط (liveness)، الجاهزية (readiness)، والعميق (deep)
3. **تكامل Kubernetes**: استخدام الفحوصات لتنسيق الحاويات
4. **تفاصيل التنفيذ**: فحوصات لـ PostgreSQL و Redis و Qdrant و LLM و التخزين
5. **أفضل الممارسات**: نقاط نهاية مناسبة، عتبات معقولة، تدهور مشرف، مهلات
6. **الأخطاء الشائعة**: فحوصات عدوانية جدًا، فشل متسلسل، إيجابيات كاذبة

تمثل فحوصات الصحة جزءًا أساسيًا من قابلية المراقبة والصحة التشغيلية للنظام.
