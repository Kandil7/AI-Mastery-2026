"""
Webhooks System
================
Service for managing webhooks and event notifications.

نظام إدارة الـ Webhooks والإشعارات
"""

import asyncio
import aiohttp
import hmac
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class Webhook:
    """Webhook configuration.

    تكوين الـ Webhook
    """

    id: str
    user_id: str
    url: str
    events: List[str]
    secret: str
    active: bool = True
    created_at: str
    last_triggered_at: Optional[str] = None


@dataclass
class WebhookEvent:
    """Webhook event type.

    أنواع أحداث الـ Webhook
    """

    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_INDEXED = "document.indexed"
    DOCUMENT_FAILED = "document.failed"
    CHAT_TURN_CREATED = "chat_turn.created"
    CHAT_SESSION_SUMMARIZED = "chat_session.summarized"
    ERROR_OCCURRED = "error.occurred"


@dataclass
class WebhookPayload:
    """Payload sent to webhook URL.

    حمولة مرسلة إلى عنوان الـ Webhook
    """

    event: str
    timestamp: str
    data: Dict[str, Any]
    signature: str


class WebhookManager:
    """
    Manage webhooks and deliver events.

    يدير الـ Webhooks ويسلم الأحداث
    """

    def __init__(self, webhook_repo, http_client):
        """
        Initialize webhook manager.

        Args:
            webhook_repo: Webhook repository
            http_client: HTTP client for delivery
        """
        self._repo = webhook_repo
        self._http = http_client

    async def create_webhook(
        self,
        user_id: str,
        url: str,
        events: List[str],
        secret: str | None = None,
    ) -> str:
        """
        Create a new webhook.

        Args:
            user_id: User ID
            url: Webhook URL
            events: Events to subscribe to
            secret: Optional HMAC secret (generate if None)

        Returns:
            New webhook ID

        إنشاء webhook جديد
        """
        import uuid

        webhook_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        # Generate secret if not provided
        if not secret:
            import secrets

            secret = secrets.token_hex(32)

        log.info(
            "webhook_created",
            webhook_id=webhook_id,
            user_id=user_id,
            events=events,
        )

        self._repo.create_webhook(
            user_id=user_id,
            url=url,
            events=events,
            secret=secret,
        )

        return webhook_id

    def delete_webhook(
        self,
        webhook_id: str,
        user_id: str,
    ) -> bool:
        """
        Delete a webhook.

        Args:
            webhook_id: Webhook ID
            user_id: User ID (for authorization)

        Returns:
            True if deleted, False if not found

        حذف webhook
        """
        log.info(
            "webhook_deleted",
            webhook_id=webhook_id,
            user_id=user_id,
        )

        return self._repo.delete_webhook(webhook_id, user_id)

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

        تشغيل حدث لجميع الـ Webhooks المتطابقة
        """
        # Find matching webhooks
        webhooks = self._repo.find_by_event(user_id, event_type)

        if not webhooks:
            log.debug(
                "no_webhooks",
                user_id=user_id,
                event=event_type,
            )
            return []

        # Prepare payload for each webhook
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
                    event_type,
                    event_data,
                ),
            )
            payloads.append(payload)

        # Deliver all webhooks
        results = []
        for webhook, payload in zip(webhooks, payloads):
            success = await self.deliver_webhook(webhook, payload)

            # Update webhook status
            self._repo.update_last_triggered(
                webhook.id,
                triggered_at=datetime.utcnow().isoformat(),
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

        تسليم webhook مع منطق إعادة المحاولة
        """
        retry_count = 0
        delay = retry_delay

        while retry_count <= max_retries:
            try:
                # Send HTTP POST
                async with self._http.post(
                    webhook.url,
                    json={
                        "event": payload.event,
                        "timestamp": payload.timestamp,
                        "data": payload.data,
                    },
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
                            status_code=response.status,
                        )
                        return True
                    else:
                        # Non-2xx status, retry
                        log.warning(
                            "webhook_retry",
                            webhook_id=webhook.id,
                            event=payload.event,
                            retry_count=retry_count,
                            status_code=response.status,
                        )
                        raise aiohttp.ClientResponseError(
                            status=response.status, message=f"HTTP {response.status}"
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

            # Exponential backoff with jitter
            if retry_count <= max_retries:
                import random

                jitter_factor = 0.1
                jitter = delay * jitter_factor
                actual_delay = delay * (2 ** (retry_count - 1)) + random.uniform(-jitter, jitter)

                log.info(
                    "webhook_retry_delay",
                    webhook_id=webhook.id,
                    event=payload.event,
                    retry_count=retry_count,
                    delay=actual_delay,
                )

                await asyncio.sleep(actual_delay)

        # All retries failed
        log.error(
            "webhook_failed",
            webhook_id=webhook.id,
            event=payload.event,
            max_retries=max_retries,
        )
        return False

    def _generate_signature(
        self,
        secret: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> str:
        """
        Generate HMAC signature.

        Args:
            secret: Webhook secret
            event_type: Event type
            data: Event data

        Returns:
            Hex-encoded HMAC signature

        توليد توقيع HMAC
        """
        payload_str = json.dumps(
            {
                "event": event_type,
                "data": data,
            },
            sort_keys=True,
        )

        signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256,
        ).hexdigest()

        return signature

    @staticmethod
    def verify_signature(
        secret: str,
        received_payload: Dict[str, Any],
        received_signature: str,
    ) -> bool:
        """
        Verify HMAC signature.

        Args:
            secret: Webhook secret
            received_payload: Payload data
            received_signature: Received signature

        Returns:
            True if signature matches, False otherwise

        التحقق من صحة توقيع HMAC
        """
        # Compute expected signature
        payload_str = json.dumps(received_payload, sort_keys=True)
        expected_signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256,
        ).hexdigest()

        # Constant-time comparison
        from hmac import compare_digest

        return compare_digest(expected_signature.encode(), received_signature.encode())
