"""
Celery Application Configuration
=================================
Celery app setup for background task processing.

إعداد تطبيق Celery للمعالجة في الخلفية
"""

from celery import Celery

from src.core.config import settings

celery_app = Celery(
    "rag_engine_mini",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Celery configuration
celery_app.conf.update(
    # Task acknowledgment after completion (safer)
    task_acks_late=True,
    
    # Only fetch one task at a time (for heavy tasks)
    worker_prefetch_multiplier=1,
    
    # Task routing
    task_routes={
        "index_document": {"queue": "indexing"},
    },
    
    # Task time limits
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_time_limit - 60,
    
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
)

# Import tasks to register them
# This is done here to avoid circular imports
celery_app.autodiscover_tasks(["src.workers"])
