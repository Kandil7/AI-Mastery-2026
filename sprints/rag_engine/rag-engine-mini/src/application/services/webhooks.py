"""
Webhooks System
=================
Service for managing webhooks and event notifications.

نظام إدارة الـ Webhooks والإشعارات
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging
import hmac
import hashlib
import json

log = logging.getLogger(__name__)


@dataclass
class Webhook:
    """Webhook configuration.

    تكوين الـ Webhook
    """

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
    signature: str  # HMAC signature


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

    def create_webhook(
        self,
        user_id: str,
        url: str,
        events: List[str],
        secret: str | None = None,
    ) -> Webhook:
        """
        Create a new webhook.

        Args:
            user_id: User ID
            url: Webhook URL
            events: Events to subscribe to
            secret: Optional HMAC secret (generate if None)

        Returns:
            Created webhook

        إنشاء webhook جديد
        """
        import uuid

        # Generate secret if not provided
        if secret is None:
            secret = self._generate_secret()

        webhook = Webhook(
            id=str(uuid.uuid4()),
            user_id=user_id,
            url=url,
            events=events,
            secret=secret,
            active=True,
            created_at=datetime.utcnow().isoformat(),
            last_triggered_at=None,
        )

        # Validate webhook URL
        if not self._validate_webhook_url(url):
            raise ValueError("Invalid webhook URL")

        # Validate events
        invalid_events = [e for e in events if e not in self._get_all_events()]
        if invalid_events:
            raise ValueError(f"Invalid events: {invalid_events}")

        # Store webhook
        self._repo.create(webhook)

        log.info("Webhook created", webhook_id=webhook.id, url=url, events=events)

        return webhook

    def get_webhooks(self, user_id: str) -> List[Webhook]:
        """Get all webhooks for a user."""
        return self._repo.find_by_user_id(user_id)

    def delete_webhook(self, webhook_id: str, user_id: str) -> bool:
        """
        Delete a webhook.

        Args:
            webhook_id: Webhook ID
            user_id: User ID for authorization

        Returns:
            True if deleted

        حذف webhook
        """
        webhook = self._repo.find_by_id(webhook_id)

        if not webhook:
            raise ValueError(f"Webhook {webhook_id} not found")

        if webhook.user_id != user_id:
            raise PermissionError("Access denied: Webhook belongs to another user")

        self._repo.delete(webhook_id)

        log.info("Webhook deleted", webhook_id=webhook_id)
        return True

    def trigger_event(
        self,
        event: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> int:
        """
        Trigger an event to all subscribed webhooks.

        Args:
            event: Event type
            data: Event payload data
            user_id: Optional user ID for filtering

        Returns:
            Number of webhooks triggered

        تشغيل حدث وإرسال للـ Webhooks المشتركة
        """
        # Get webhooks to trigger
        if user_id:
            webhooks = self._repo.find_by_user_id_and_event(user_id, event)
        else:
            webhooks = self._repo.find_by_event(event)

        # Filter active webhooks
        active_webhooks = [w for w in webhooks if w.active and event in w.events]

        if not active_webhooks:
            log.debug("No active webhooks for event", event=event)
            return 0

        # Build payload
        timestamp = datetime.utcnow().isoformat()
        payload = WebhookPayload(
            event=event,
            timestamp=timestamp,
            data=data,
            signature="",  # Will be set per webhook
        )

        # Deliver to each webhook
        success_count = 0
        for webhook in active_webhooks:
            try:
                # Generate signature for this webhook
                payload.signature = self._generate_signature(
                    json.dumps(
                        {"event": event, "timestamp": timestamp, "data": data}, sort_keys=True
                    ),
                    webhook.secret,
                )

                # Send webhook
                response = self._http.post(
                    webhook.url,
                    json=payload.__dict__,
                    headers={
                        "Content-Type": "application/json",
                        "X-RAG-Webhook-Signature": payload.signature,
                        "X-RAG-Webhook-Event": event,
                        "X-RAG-Webhook-Timestamp": timestamp,
                    },
                    timeout=5,  # 5 second timeout
                )

                response.raise_for_status()

                # Update last triggered timestamp
                self._repo.update_last_triggered(webhook.id, timestamp)

                success_count += 1
                log.info("Webhook delivered", webhook_id=webhook.id, event=event)

            except Exception as e:
                log.error(
                    "Webhook delivery failed",
                    webhook_id=webhook.id,
                    event=event,
                    error=str(e),
                )

        return success_count

    def test_webhook(self, webhook_id: str, user_id: str) -> Dict[str, Any]:
        """
        Test a webhook delivery (ping event).

        Args:
            webhook_id: Webhook ID
            user_id: User ID for authorization

        Returns:
            Test result with response info

        اختبار توصيل webhook
        """
        webhook = self._repo.find_by_id(webhook_id)

        if not webhook:
            raise ValueError(f"Webhook {webhook_id} not found")

        if webhook.user_id != user_id:
            raise PermissionError("Access denied: Webhook belongs to another user")

        # Create test payload
        test_data = {
            "test": True,
            "webhook_id": webhook_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Generate signature
        signature = self._generate_signature(
            json.dumps({"event": "ping", "data": test_data}, sort_keys=True),
            webhook.secret,
        )

        # Send test request
        try:
            response = self._http.post(
                webhook.url,
                json={
                    "event": "ping",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": test_data,
                    "signature": signature,
                },
                headers={
                    "Content-Type": "application/json",
                    "X-RAG-Webhook-Signature": signature,
                },
                timeout=10,
            )

            result = {
                "success": response.status_code in [200, 201, 202, 204],
                "status_code": response.status_code,
                "response_text": response.text,
            }

            log.info("Webhook test completed", webhook_id=webhook.id, result=result)
            return result

        except Exception as e:
            log.error("Webhook test failed", webhook_id=webhook_id, error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    def verify_webhook_signature(
        self,
        payload: str | bytes,
        signature: str,
        secret: str,
    ) -> bool:
        """
        Verify webhook signature.

        Args:
            payload: Original payload (string or bytes)
            signature: Received signature
            secret: Webhook secret

        Returns:
            True if signature is valid

        التحقق من توقيع webhook
        """
        expected_signature = self._generate_signature(payload, secret)
        return hmac.compare_digest(expected_signature, signature)

    def _generate_signature(self, payload: str | bytes, secret: str) -> str:
        """Generate HMAC signature."""
        if isinstance(payload, bytes):
            payload_bytes = payload
        else:
            payload_bytes = payload.encode("utf-8")

        hmac_obj = hmac.new(
            secret.encode("utf-8"),
            hashlib.sha256,
        )
        signature = hmac_obj.hexdigest(payload_bytes)

        return signature

    def _generate_secret(self) -> str:
        """Generate random webhook secret."""
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits
        secret = "".join(secrets.choice(alphabet) for _ in range(64))

        return secret

    def _get_all_events(self) -> List[str]:
        """Get all available event types."""
        return [
            WebhookEvent.DOCUMENT_UPLOADED,
            WebhookEvent.DOCUMENT_DELETED,
            WebhookEvent.DOCUMENT_INDEXED,
            WebhookEvent.DOCUMENT_FAILED,
            WebhookEvent.CHAT_TURN_CREATED,
            WebhookEvent.CHAT_SESSION_SUMMARIZED,
            WebhookEvent.ERROR_OCCURRED,
        ]

    def _validate_webhook_url(self, url: str) -> bool:
        """Validate webhook URL."""
        import re

        # Check for HTTP/HTTPS
        if not url.startswith(("http://", "https://")):
            return False

        # Check for localhost in production
        if "localhost" in url or "127.0.0.1" in url:
            return False

        return True


# -----------------------------------------------------------------------------
# Webhook Repository Interface (placeholder)
# -----------------------------------------------------------------------------


class WebhookRepository:
    """
    Interface for webhook persistence.

    واجهة استمرار webhook
    """

    def create(self, webhook: Webhook) -> str:
        """Create a webhook. Returns ID."""
        raise NotImplementedError()

    def find_by_id(self, webhook_id: str) -> Optional[Webhook]:
        """Find webhook by ID."""
        raise NotImplementedError()

    def find_by_user_id(self, user_id: str) -> List[Webhook]:
        """Find all webhooks for user."""
        raise NotImplementedError()

    def find_by_user_id_and_event(
        self,
        user_id: str,
        event: str,
    ) -> List[Webhook]:
        """Find webhooks for user subscribed to event."""
        raise NotImplementedError()

    def find_by_event(self, event: str) -> List[Webhook]:
        """Find all webhooks subscribed to event."""
        raise NotImplementedError()

    def update_last_triggered(self, webhook_id: str, timestamp: str) -> bool:
        """Update last triggered timestamp."""
        raise NotImplementedError()

    def delete(self, webhook_id: str) -> bool:
        """Delete a webhook."""
        raise NotImplementedError()


if __name__ == "__main__":
    from unittest.mock import Mock

    # Test webhook manager
    repo = Mock()
    http = Mock()

    manager = WebhookManager(repo, http)

    # Test webhook creation
    webhook = manager.create_webhook(
        user_id="user-123",
        url="https://example.com/webhook",
        events=[WebhookEvent.DOCUMENT_UPLOADED, WebhookEvent.DOCUMENT_INDEXED],
    )
    print(f"Created webhook: {webhook.id}")

    # Test event triggering
    http.post.return_value.status_code = 200
    manager.trigger_event(
        WebhookEvent.DOCUMENT_UPLOADED,
        {"document_id": "doc-456", "filename": "test.pdf"},
        user_id="user-123",
    )

    # Test signature verification
    signature = manager._generate_signature('{"test": "data"}', "secret123")
    is_valid = manager.verify_webhook_signature('{"test": "data"}', signature, "secret123")
    print(f"Signature valid: {is_valid}")
