"""Task queue adapters package."""

from src.adapters.queue.celery_queue import CeleryTaskQueue

__all__ = ["CeleryTaskQueue"]
