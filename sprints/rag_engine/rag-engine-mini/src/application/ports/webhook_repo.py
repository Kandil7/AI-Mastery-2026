"""
Webhook Repository Port
======================
Interface for webhook persistence.

منفذ مستودع الـ Webhooks
"""

from typing import Protocol, List
from datetime import datetime


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
    created_at: datetime
    last_triggered_at: datetime | None = None


class WebhookRepoPort(Protocol):
    """
    Port for webhook persistence.

    منفذ لاستمرارية الـ Webhooks
    """

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

    def list_webhooks(
        self,
        user_id: str,
    ) -> List[Webhook]:
        """
        List all webhooks for user.

        Args:
            user_id: User ID

        Returns:
            List of webhooks
        """
        ...
