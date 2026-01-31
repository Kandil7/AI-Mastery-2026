# Webhooks Delivery - Complete Implementation Guide
# ===========================================

## 📚 Learning Objectives

By the end of this guide, you will understand:
- Webhooks architecture and event-driven systems
- Retry strategies with exponential backoff
- Async event delivery patterns
- HMAC signature verification for security
- Webhook repository patterns
- Best practices for reliable webhook delivery

---
## 📌 Table of Contents

1. [Introduction](#1-introduction)
2. [Webhook Architecture](#2-webhook-architecture)
3. [Webhook Delivery with Retries](#3-webhook-delivery-with-retries)
4. [Event Triggering System](#4-event-triggering-system)
5. [HMAC Signature Verification](#5-hmac-signature-verification)
6. [Webhook Repository](#6-webhook-repository)
7. [Error Handling](#7-error-handling)
8. [Performance Considerations](#8-performance-considerations)
9. [Best Practices](#9-best-practices)
10. [Quiz](#10-quiz)
11. [References](#11-references)

---
## 1. Introduction

### 1.1 What are Webhooks?

Webhooks are HTTP callbacks that enable event-driven communication between systems. When an event occurs in your application, you send an HTTP POST to a configured URL with event data.

**Key Characteristics:**
- **Event-Driven**: Triggered by specific events (document.upload, chat.session.created, etc.)
- **Push-Based**: Server pushes data to client (no polling)
- **Asynchronous**: Non-blocking event delivery
- **Retryable**: Failed deliveries can be retried

### 1.2 Why Use Webhooks?

**Use Cases:**
1. **Real-Time Notifications**: Notify external systems immediately when events occur
2. **Third-Party Integrations**: Allow external services to react to your events
3. **Automation**: Trigger external workflows based on your events
4. **Audit Trails**: Send events to external logging/analytics systems
5. **Custom Actions**: Execute custom logic when specific events occur

**Examples:**
- Send notification to Slack when document is uploaded
- Trigger CI/CD pipeline when document is indexed
- Update external CRM when chat session is created
- Send metrics to external monitoring system
- Trigger webhook for compliance reporting

### 1.3 Webhook Flow

```
Event Occurs
    │
    ▼
Event Publisher (RAG Engine)
    │
    ├─→ Find matching webhooks
    │   - Filter by user_id
    │   - Filter by event type
    │   - Check active status
    │
    ├─→ Prepare payload
    │   - Add event type
    │   - Add timestamp
    │   - Add event data
    │   - Generate HMAC signature
    │
    ├─→ Queue delivery task
    │   - Use Celery/Redis queue
    │   - Non-blocking async delivery
    │
    └─→ Deliver Webhook (Worker)
        │
        ├─→ HTTP POST to webhook URL
        │   - Retry on failure
        │   - Exponential backoff
        │   - Max 3-5 retries
        │
        ├─→ Verify response
        │   - Check status code (2xx = success)
        │   - Validate response body if needed
        │
        ├─→ Update webhook status
        │   - Update last_triggered_at
        │   - Update last_error (if failed)
        │
        └─→ Log delivery
```

---
## 2. Webhook Architecture

### 2.1 Components

**Webhook Manager** (`src/application/services/webhooks.py`):
- `WebhookManager`: Main service for managing webhooks
- `Webhook`: Dataclass representing webhook configuration
- `WebhookEvent`: Enum of event types
- `WebhookPayload`: Payload sent to webhook URL

**Webhook Repository** (`src/adapters/persistence/postgres/repo_webhooks.py`):
- `WebhookRepoPort`: Interface for webhook persistence
- `PostgresWebhookRepo`: PostgreSQL implementation

**Task Queue** (`src/workers/tasks.py`):
- `deliver_webhook_task`: Celery task for async delivery
- Retry configuration: exponential backoff, max retries

### 2.2 Webhook Data Model

```python
@dataclass
class Webhook:
    """Webhook configuration."""
    id: str
    user_id: str
    url: str
    events: List[str]
    secret: str  # HMAC secret for verification
    active: bool = True
    created_at: str
    last_triggered_at: Optional[str] = None

@dataclass
class WebhookEvent:
    """Webhook event type."""
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_INDEXED = "document.indexed"
    DOCUMENT_FAILED = "document.failed"
    CHAT_TURN_CREATED = "chat_turn.created"
    CHAT_SESSION_SUMMARIZED = "chat_session.summarized"
    ERROR_OCCURRED = "error.occurred"

@dataclass
class WebhookPayload:
    """Payload sent to webhook URL."""
    event: str
    timestamp: str
    data: Dict[str, Any]
    signature: str  # HMAC signature
```

---
## 3. Webhook Delivery with Retries

### 3.1 Delivery Function

**Implementation:**
```python
import asyncio
import aiohttp
import hmac
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

log = logging.getLogger(__name__)

class WebhookManager:
    """Manage webhooks and deliver events."""

    def __init__(self, webhook_repo, http_client):
        """
        Initialize webhook manager.

        Args:
            webhook_repo: Webhook repository
            http_client: HTTP client for delivery
        """
        self._repo = webhook_repo
        self._http = http_client

    async def deliver_webhook(
        self,
        webhook: Webhook,
        payload: WebhookPayload,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> bool:
        """
        Deliver webhook with retry logic.

        Args:
            webhook: Webhook configuration
            payload: Payload to send
            max_retries: Maximum retry attempts (default: 3)
            retry_delay: Initial delay in seconds (default: 1.0)

        Returns:
            True if delivery succeeded, False otherwise
        """
        retry_count = 0
        delay = retry_delay

        while retry_count <= max_retries:
            try:
                # Send HTTP POST
                async with self._http.post(
                    webhook.url,
                    json=payload.__dict__,
                    headers={
                        "Content-Type": "application/json",
                        "X-Webhook-Signature": payload.signature,
                        "User-Agent": "RAG-Engine/1.0",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    # Check status code
                    if 200 <= response.status < 300:
                        log.info(
                            "webhook_delivered",
                            webhook_id=webhook.id,
                            event=payload.event,
                            retry_count=retry_count,
                        )
                        return True
                    else:
                        # Non-2xx status, retry
                        raise aiohttp.ClientResponseError(
                            status=response.status,
                            message=f"HTTP {response.status}"
                        )

            except asyncio.TimeoutError:
                log.warning(
                    "webhook_timeout",
                    webhook_id=webhook.id,
                    event=payload.event,
                    retry_count=retry_count,
                )
            except aiohttp.ClientError as e:
                log.error(
                    "webhook_delivery_error",
                    webhook_id=webhook.id,
                    event=payload.event,
                    retry_count=retry_count,
                    error=str(e),
                )

            # Increment retry count
            retry_count += 1

            # Exponential backoff
            if retry_count <= max_retries:
                delay = delay * (2 ** (retry_count - 1))
                log.info(
                    "webhook_retry",
                    webhook_id=webhook.id,
                    event=payload.event,
                    retry_count=retry_count,
                    delay=delay,
                )
                await asyncio.sleep(delay)

        # All retries failed
        log.error(
            "webhook_failed",
            webhook_id=webhook.id,
            event=payload.event,
            max_retries=max_retries,
        )
        return False
```

### 3.2 Retry Strategy

**Exponential Backoff:**
- Start with 1 second delay
- Double delay each retry (1s → 2s → 4s → 8s)
- Max retries: 3 (typical) to 5 (high importance)

**Formula:**
```
delay = initial_delay * (2 ^ (retry_count - 1))
```

**Example:**
```
Retry 1: delay = 1s
Retry 2: delay = 2s
Retry 3: delay = 4s
```

### 3.3 Jitter (Randomization)

**Add jitter to avoid thundering herd:**
```python
import random

def get_jitter_delay(base_delay: float, jitter_factor: float = 0.1) -> float:
    """
    Add random jitter to delay.

    Args:
        base_delay: Base delay in seconds
        jitter_factor: Jitter percentage (0.1 = 10%)

    Returns:
        Delayed delay with jitter
    """
    jitter = base_delay * jitter_factor
    return base_delay + random.uniform(-jitter, jitter)

# Usage
delay = get_jitter_delay(2.0)  # 2.0s ± 10% = 1.8s to 2.2s
```

---
## 4. Event Triggering System

### 4.1 Trigger Function

**Implementation:**
```python
class WebhookManager:
    # ... (previous methods) ...

    async def trigger_event(
        self,
        user_id: str,
        event_type: str,
        event_data: Dict[str, Any],
    ) -> List[bool]:
        """
        Trigger event to all matching webhooks.

        Args:
            user_id: User ID for filtering
            event_type: Event type (e.g., "document.uploaded")
            event_data: Event-specific data

        Returns:
            List of delivery results (True=success, False=failed)
        """
        # 1. Find matching webhooks
        webhooks = self._repo.find_by_event(user_id, event_type)

        if not webhooks:
            log.debug(
                "no_webhooks",
                user_id=user_id,
                event=event_type,
            )
            return []

        # 2. Prepare payload for each webhook
        payloads = []
        for webhook in webhooks:
            if not webhook.active:
                log.debug("webhook_inactive", webhook_id=webhook.id)
                continue

            payload = WebhookPayload(
                event=event_type,
                timestamp=datetime.utcnow().isoformat(),
                data=event_data,
                signature=self._generate_signature(
                    webhook.secret,
                    event_data,
                ),
            )
            payloads.append(payload)

        # 3. Deliver all webhooks
        results = []
        for webhook, payload in zip(webhooks, payloads):
            success = await self.deliver_webhook(webhook, payload)

            # Update webhook status
            self._repo.update_last_triggered(
                webhook.id,
                triggered_at=datetime.utcnow(),
            )

            # Log delivery
            log.info(
                "webhook_triggered",
                webhook_id=webhook.id,
                event=event_type,
                success=success,
            )

            results.append(success)

        return results

    def _generate_signature(
        self,
        secret: str,
        data: Dict[str, Any],
    ) -> str:
        """
        Generate HMAC signature.

        Args:
            secret: Webhook secret key
            data: Data to sign

        Returns:
            Hex-encoded HMAC signature
        """
        payload_str = json.dumps(data, sort_keys=True)
        signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature
```

### 4.2 Triggering from Use Cases

**Example: Document Upload**
```python
class UploadDocumentUseCase:
    def __init__(self, file_store, document_repo, webhook_manager):
        self._webhook_manager = webhook_manager
        # ... other dependencies ...

    def execute(self, request):
        # Upload document
        document_id = self._document_repo.create_document(...)

        # Trigger webhook
        event_data = {
            "document_id": document_id,
            "filename": request.filename,
            "size_bytes": request.size_bytes,
        }

        await self._webhook_manager.trigger_event(
            user_id=request.tenant_id,
            event_type=WebhookEvent.DOCUMENT_UPLOADED,
            event_data=event_data,
        )

        return document_id
```

---
## 5. HMAC Signature Verification

### 5.1 Signature Generation

**Algorithm:**
1. Serialize payload to JSON (sorted keys)
2. Encode payload string
3. Compute HMAC-SHA256 using webhook secret
4. Return hex-encoded signature

**Implementation:**
```python
def generate_signature(secret: str, payload: Dict[str, Any]) -> str:
    """
    Generate HMAC-SHA256 signature.

    Args:
        secret: Webhook secret
        payload: Payload data

    Returns:
        Hex-encoded signature
    """
    import hmac
    import hashlib
    import json

    payload_str = json.dumps(payload, sort_keys=True)
    signature = hmac.new(
        secret.encode(),
        payload_str.encode(),
        hashlib.sha256,
    ).hexdigest()
    return signature
```

### 5.2 Signature Verification

**Implementation:**
```python
def verify_signature(
    secret: str,
    payload: Dict[str, Any],
    received_signature: str,
) -> bool:
    """
    Verify HMAC signature.

    Args:
        secret: Webhook secret
        payload: Received payload
        received_signature: Received signature

    Returns:
        True if signature matches, False otherwise
    """
    # Compute expected signature
    expected_signature = generate_signature(secret, payload)

    # Constant-time comparison
    from hmac import compare_digest
    return compare_digest(expected_signature.encode(), received_signature.encode())
```

### 5.3 Signature Header

**HTTP Headers:**
```http
POST /webhook HTTP/1.1
Host: webhook.example.com
Content-Type: application/json
X-Webhook-Signature: a1b2c3d4e5f6...
User-Agent: RAG-Engine/1.0
```

**Code:**
```python
headers = {
    "Content-Type": "application/json",
    "X-Webhook-Signature": signature,
    "User-Agent": "RAG-Engine/1.0",
}
```

---
## 6. Webhook Repository

### 6.1 Repository Interface

**Implementation:**
```python
from typing import Protocol, Optional
from dataclasses import dataclass

@dataclass
class Webhook:
    """Webhook configuration."""
    id: str
    user_id: str
    url: str
    events: List[str]
    secret: str
    active: bool = True
    created_at: str
    last_triggered_at: Optional[str] = None

class WebhookRepoPort(Protocol):
    """Port for webhook persistence."""

    def create_webhook(
        self,
        user_id: str,
        url: str,
        events: List[str],
        secret: str,
    ) -> str:
        """
        Create new webhook.

        Args:
            user_id: User ID
            url: Webhook URL
            events: Subscribed events
            secret: HMAC secret

        Returns:
            New webhook ID
        """
        ...

    def find_by_event(
        self,
        user_id: str,
        event_type: str,
    ) -> List[Webhook]:
        """
        Find webhooks subscribed to event.

        Args:
            user_id: User ID
            event_type: Event type

        Returns:
            List of matching webhooks
        """
        ...

    def update_last_triggered(
        self,
        webhook_id: str,
        triggered_at: datetime,
    ) -> None:
        """
        Update last triggered timestamp.

        Args:
            webhook_id: Webhook ID
            triggered_at: Trigger timestamp
        """
        ...

    def delete_webhook(
        self,
        webhook_id: str,
        user_id: str,
    ) -> bool:
        """
        Delete webhook.

        Args:
            webhook_id: Webhook ID
            user_id: User ID (for authorization)

        Returns:
            True if deleted
        """
        ...
```

### 6.2 PostgreSQL Implementation

```python
from sqlalchemy import text
from datetime import datetime

class PostgresWebhookRepo(WebhookRepoPort):
    """PostgreSQL implementation of webhook repository."""

    def __init__(self, db):
        self._db = db

    def create_webhook(
        self,
        user_id: str,
        url: str,
        events: List[str],
        secret: str,
    ) -> str:
        """Create new webhook."""
        webhook_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        query = text("""
            INSERT INTO webhooks (id, user_id, url, events, secret, active, created_at)
            VALUES (:id, :user_id, :url, :events, :secret, TRUE, :created_at)
            RETURNING id
        """)

        result = self._db.execute(
            query,
            {
                "id": webhook_id,
                "user_id": user_id,
                "url": url,
                "events": json.dumps(events),
                "secret": secret,
                "created_at": created_at,
            }
        )

        return webhook_id

    def find_by_event(
        self,
        user_id: str,
        event_type: str,
    ) -> List[Webhook]:
        """Find webhooks subscribed to event."""
        query = text("""
            SELECT id, user_id, url, events, secret, active, created_at, last_triggered_at
            FROM webhooks
            WHERE user_id = :user_id
              AND active = TRUE
              AND :event_type = ANY(events)
        """)

        rows = self._db.execute(query, {"user_id": user_id, "event_type": event_type}).fetchall()

        return [
            Webhook(
                id=row["id"],
                user_id=row["user_id"],
                url=row["url"],
                events=json.loads(row["events"]),
                secret=row["secret"],
                active=row["active"],
                created_at=row["created_at"],
                last_triggered_at=row["last_triggered_at"],
            )
            for row in rows
        ]

    def update_last_triggered(
        self,
        webhook_id: str,
        triggered_at: datetime,
    ) -> None:
        """Update last triggered timestamp."""
        query = text("""
            UPDATE webhooks
            SET last_triggered_at = :triggered_at
            WHERE id = :webhook_id
        """)

        self._db.execute(query, {"webhook_id": webhook_id, "triggered_at": triggered_at})

    def delete_webhook(
        self,
        webhook_id: str,
        user_id: str,
    ) -> bool:
        """Delete webhook."""
        query = text("""
            DELETE FROM webhooks
            WHERE id = :webhook_id AND user_id = :user_id
            RETURNING id
        """)

        result = self._db.execute(
            query,
            {"webhook_id": webhook_id, "user_id": user_id}
        ).fetchone()

        return result is not None
```

---
## 7. Error Handling

### 7.1 Webhook Errors

| Error Type | Cause | Handling |
|------------|--------|----------|
| Timeout | Webhook URL unresponsive | Retry with exponential backoff |
| 4xx | Client error (bad URL, bad secret) | Log and fail (no retry) |
| 5xx | Server error | Retry with exponential backoff |
| Signature Invalid | Secret mismatch | Return 401 Unauthorized |
| Rate Limited | Too many requests | Retry after delay |
| JSON Parse Error | Invalid response | Log and fail |

### 7.2 Error Response

**Return 503 on failure:**
```python
if not success:
    raise HTTPException(
        status_code=503,
        detail="Webhook delivery failed",
    )
```

---
## 8. Performance Considerations

### 8.1 Async Delivery

**Use async HTTP client:**
```python
import aiohttp

async with aiohttp.ClientSession() as session:
    async with session.post(url, json=payload) as response:
        # Handle response
        pass
```

### 8.2 Connection Pooling

**Reuse HTTP connections:**
```python
async with aiohttp.TCPConnector(limit=100) as connector:
    async with aiohttp.ClientSession(connector=connector) as session:
        # Reuse connections
        pass
```

### 8.3 Batch Delivery

**Deliver webhooks in parallel:**
```python
async def deliver_all(webhooks, payloads):
    """Deliver all webhooks in parallel."""
    tasks = []
    for webhook, payload in zip(webhooks, payloads):
        task = asyncio.create_task(deliver_webhook(webhook, payload))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

---
## 9. Best Practices

### 9.1 Security

**DO:**
- Use HTTPS for webhook URLs
- Generate secure secrets (32+ chars)
- Verify HMAC signatures
- Add rate limiting per webhook
- Log all delivery attempts

**DON'T:**
- Send secrets in payload
- Skip signature verification
- Allow unverified HTTP
- Ignore webhook errors

### 9.2 Reliability

**DO:**
- Retry on transient failures (timeouts, 5xx)
- Use exponential backoff
- Add jitter to avoid thundering herd
- Queue webhooks for async delivery
- Monitor webhook health

**DON'T:**
- Give up after first failure
- Retry synchronously
- Ignore failed webhooks
- Skip webhook status updates

### 9.3 Observability

**DO:**
- Log all webhook deliveries
- Track success/failure rates
- Alert on high failure rates
- Monitor webhook latency
- Store delivery history

**DON'T:**
- Skip logging
- Ignore errors
- Hide delivery failures
- Lack metrics

---
## 10. Quiz

### Question 1
What is exponential backoff?
- [ ] A) Fixed delay between retries
- [ ] B) Delay doubles each retry
- [ ] C) Random delay between retries
- [ ] D] No delay between retries

**Answer:** B - Delay doubles each retry.

---

### Question 2
Why use HMAC signatures?
- [ ] A) Encrypt webhook payload
- [ ] B) Verify webhook authenticity
- [ ] C) Compress webhook payload
- [ ] D) Add metadata to webhook

**Answer:** B - Verify webhook authenticity.

---

### Question 3
What is jitter in retry logic?
- [ ] A) Fixed delay
- [ ] B) Random variation in delay
- [ ] C) No delay
- [ ] D) Exponential delay

**Answer:** B - Random variation in delay.

---

### Question 4
When should you NOT retry webhook delivery?
- [ ] A) Timeout (4xx client errors)
- [ ] B) Server error (5xx)
- [ ] C) Network timeout
- [ ] D) Invalid secret (signature mismatch)

**Answer:** D - Invalid secret (don't retry auth errors).

---

### Question 5
How do you avoid thundering herd?
- [ ] A) Sequential delivery
- [ ] B) Parallel delivery with jitter
- [ ] C) Single webhook
- [ ] D] Fixed retry delay

**Answer:** B - Parallel delivery with jitter.

---
## 11. References

### Official Documentation
- **Webhook Best Practices:** https://webhooks.best practices/
- **HMAC:** https://tools.ietf.org/html/rfc2104
- **aiohttp:** https://docs.aiohttp.org/

### Related Resources
- This file: `src/application/services/webhooks.py`
- Task queue: `src/workers/tasks.py`
- Related notebook: `notebooks/learning/observability/webhooks-delivery.ipynb`

---
## 🇸🇦 ترجمة عربية / Arabic Translation

### 1. مقدمة

#### 1.1 ما هي الـ Webhooks؟

الـ Webhooks هي عمليات استدعاء HTTP تتيح الاتصال المدفوع بالأحداث بين الأنظمة. عندما يحدث حدث في تطبيقك، ترسل طلب HTTP POST إلى عنوان URL مهي ببيانات الحدث.

**الخصائص الرئيسية:**
- **مدفوعة بالأحداث**: يتم تشغيلها بواسطة أحداث محددة (document.upload، chat.session.created، إلخ)
- **القائمة على الدفع**: الخادم يدفع البيانات إلى العميل (لا يوجد استقصاء)
- **غير متزامنة**: تسليم الأحداث غير الحاجبة
- **قابلة لإعادة المحاولة**: عمليات التسليم الفاشلة يمكن إعادة محاولتها

### 3. تسليم Webhook مع إعادة المحاولة

#### 3.1 وظيفة التسليم

**التنفيذ:**
```python
class WebhookManager:
    async def deliver_webhook(
        self,
        webhook: Webhook,
        payload: WebhookPayload,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> bool:
        """
        تسليم webhook مع منطق إعادة المحاولة.

        Args:
            webhook: تكوين webhook
            payload: الحمولة المراد إرسالها
            max_retries: أقصى عدد محاولات إعادة المحاولة (الافتراضي: 3)
            retry_delay: التأخير الأولي بالثواني (الافتراضي: 1.0)

        Returns:
            True if التسليم نجح، False خلاف ذلك
        """
        retry_count = 0
        delay = retry_delay

        while retry_count <= max_retries:
            try:
                # إرسال HTTP POST
                async with self._http.post(webhook.url, json=payload) as response:
                    # التحقق من رمز الحالة
                    if 200 <= response.status < 300:
                        return True
                    else:
                        # رمز حالة غير 2xx، إعادة المحاولة
                        raise aiohttp.ClientResponseError(...)

            except aiohttp.ClientError as e:
                # تسجيل الخطأ

            # زيادة عداد إعادة المحاولة
            retry_count += 1

            # التراجع الأسي للمؤخّر
            if retry_count <= max_retries:
                delay = delay * (2 ** (retry_count - 1))
                await asyncio.sleep(delay)

        # فشلت جميع محاولات إعادة المحاولة
        return False
```

### 9. أفضل الممارسات

#### 9.1 الأمان

**افعل:**
- استخدم HTTPS لعناوين webhook
- إنشاء أسرار آمنة (32+ حرفًا)
- تحقق من توقيعات HMAC
- أضف حد المعدل لكل webhook
- سجّل جميع محاولات التسليم

**لا تفعل:**
- أرسل الأسرار في الحمولة
- تخطي التحقق من التوقيع
- السماح بـ HTTP غير الموثق
- تجاهل أخطاء webhook

---
## 📝 Summary

In this comprehensive guide, we covered:

1. **Webhook Architecture** - Event-driven systems
2. **Delivery with Retries** - Exponential backoff, jitter
3. **Event Triggering** - Filter and trigger webhooks
4. **HMAC Verification** - Signature generation and verification
5. **Repository Pattern** - PostgreSQL implementation
6. **Error Handling** - Proper error responses
7. **Performance** - Async delivery, connection pooling
8. **Best Practices** - Security, reliability, observability

## 🚀 Next Steps

1. Read companion Jupyter notebook
2. Implement webhook service code
3. Add webhook API endpoints
4. Proceed to Phase 3: Document Storage

---

**Document Version:** 1.0
**Last Updated:** 2026-01-31
**Author:** AI-Mastery-2026
