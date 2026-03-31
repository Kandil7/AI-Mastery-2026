"""
Celery Monitoring Helpers
=========================
Best-effort worker and queue status for admin endpoints.
"""

from typing import Dict, Any

from src.workers.celery_app import celery_app


def get_worker_status() -> Dict[str, Any]:
    """
    Return summary of Celery workers and queues.
    """
    inspect = celery_app.control.inspect()
    stats = inspect.stats() or {}
    active = inspect.active() or {}
    reserved = inspect.reserved() or {}

    workers_online = len(stats)
    active_count = sum(len(tasks) for tasks in active.values())
    reserved_count = sum(len(tasks) for tasks in reserved.values())
    busy_workers = sum(1 for tasks in active.values() if tasks)
    workers_idle = max(workers_online - busy_workers, 0)

    tasks_processed = 0
    for worker_stats in stats.values():
        total = worker_stats.get("total") if isinstance(worker_stats, dict) else None
        if isinstance(total, dict):
            tasks_processed += sum(total.values())

    return {
        "workers_online": workers_online,
        "workers_processing": active_count,
        "workers_idle": workers_idle,
        "tasks_queue_size": reserved_count,
        "tasks_processed": tasks_processed,
    }
