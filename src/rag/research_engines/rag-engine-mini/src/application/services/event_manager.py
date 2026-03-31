"""
Event Manager for Real-time Subscriptions
=========================================
Manages event broadcasting for GraphQL subscriptions.

مدير الأحداث للاشتراكات في الوقت الفعلي
"""

import asyncio
import logging
from typing import Callable, Dict, List, Any, Optional

log = logging.getLogger(__name__)


class EventManager:
    """
    Manages real-time event broadcasting for WebSocket subscriptions.

    This is an in-memory implementation. In production, use Redis Pub/Sub
    for multi-instance deployments.

    إدارة أحداث الوقت الفعلي للاشتراكات WebSocket
    """

    def __init__(self):
        """Initialize event manager with empty subscriptions."""
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, event_type: str, callback: Callable[[dict], Any]):
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to (e.g., "document.indexed")
            callback: Async callback function to call when event occurs

        الاشتراك في نوع حدث معين
        """
        async with self._lock:
            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []
            self._subscriptions[event_type].append(callback)
        log.info(f"Subscribed to event: {event_type}")

    async def unsubscribe(self, event_type: str, callback: Callable[[dict], Any]):
        """
        Unsubscribe from an event type.

        Args:
            event_type: Event type to unsubscribe from
            callback: Callback function to remove

        إلغاء الاشتراك من نوع حدث معين
        """
        async with self._lock:
            if event_type in self._subscriptions:
                try:
                    self._subscriptions[event_type].remove(callback)
                    log.info(f"Unsubscribed from event: {event_type}")
                except ValueError:
                    pass  # Callback not in list

    async def publish(self, event_type: str, event_data: dict):
        """
        Publish an event to all subscribers.

        Args:
            event_type: Event type to publish
            event_data: Event payload data

        نشر حدث لجميع المشتركين
        """
        async with self._lock:
            callbacks = self._subscriptions.get(event_type, []).copy()

        # Call all callbacks (don't hold the lock)
        if callbacks:
            log.info(f"Publishing event: {event_type} to {len(callbacks)} subscribers")
            tasks = []
            for callback in callbacks:
                try:
                    task = asyncio.create_task(callback(event_data))
                    tasks.append(task)
                except Exception as e:
                    log.error(f"Error creating task for callback: {e}")

            # Wait for all callbacks to complete (with timeout)
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    log.warning(f"Event publishing timed out for: {event_type}")

    def get_subscriber_count(self, event_type: str) -> int:
        """
        Get the number of subscribers for an event type.

        Args:
            event_type: Event type to query

        Returns:
            Number of subscribers

        الحصول على عدد المشتركين لنوع حدث معين
        """
        return len(self._subscriptions.get(event_type, []))

    def list_event_types(self) -> List[str]:
        """
        Get all registered event types.

        Returns:
            List of event type strings

        الحصول على جميع أنواع الأحداث المسجلة
        """
        return list(self._subscriptions.keys())

    async def clear_subscriptions(self, event_type: Optional[str] = None):
        """
        Clear all subscriptions or subscriptions for a specific event type.

        Args:
            event_type: Optional event type to clear. If None, clears all.

        مسح جميع الاشتراكات أو اشتراكات نوع حدث معين
        """
        async with self._lock:
            if event_type:
                if event_type in self._subscriptions:
                    del self._subscriptions[event_type]
                    log.info(f"Cleared subscriptions for event: {event_type}")
            else:
                self._subscriptions.clear()
                log.info("Cleared all subscriptions")


# Global event manager instance (singleton)
# In production, inject via dependency injection
_event_manager: Optional[EventManager] = None


def get_event_manager() -> EventManager:
    """
    Get the global event manager instance.

    Returns:
        EventManager instance

    الحصول على مثيل مدير الأحداث العالمي
    """
    global _event_manager
    if _event_manager is None:
        _event_manager = EventManager()
    return _event_manager


def set_event_manager(manager: EventManager):
    """
    Set the global event manager instance.

    Args:
        manager: EventManager instance to use

    تعيين مثيل مدير الأحداث العالمي
    """
    global _event_manager
    _event_manager = manager


if __name__ == "__main__":
    # Test event manager
    async def test():
        manager = EventManager()

        received_events = []

        async def callback1(data):
            received_events.append(("callback1", data))

        async def callback2(data):
            received_events.append(("callback2", data))

        # Subscribe
        await manager.subscribe("document.indexed", callback1)
        await manager.subscribe("document.indexed", callback2)

        # Publish
        await manager.publish("document.indexed", {"document_id": "doc-123"})

        # Wait a bit
        await asyncio.sleep(0.1)

        print(f"Received {len(received_events)} events")
        for source, data in received_events:
            print(f"  {source}: {data}")

        # Unsubscribe
        await manager.unsubscribe("document.indexed", callback1)

        # Publish again
        received_events.clear()
        await manager.publish("document.indexed", {"document_id": "doc-456"})

        await asyncio.sleep(0.1)
        print(f"After unsubscribe: {len(received_events)} events")

    asyncio.run(test())
