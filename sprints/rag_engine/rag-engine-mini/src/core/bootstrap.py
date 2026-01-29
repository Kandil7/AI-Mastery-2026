"""
DI Container / Bootstrap
=========================
Dependency Injection container that wires all adapters to ports.

حاوية حقن التبعيات
"""

from functools import lru_cache

from qdrant_client import QdrantClient

from src.core.config import settings


@lru_cache(maxsize=1)
def get_container() -> dict:
    """
    Build and cache the DI container.
    
    This is the single place where all adapters are instantiated
    and wired to their corresponding ports.
    
    Design Decision: Using a simple dict as container for clarity.
    For larger projects, consider dependency-injector or similar.
    
    هذا هو المكان الوحيد لإنشاء وربط المحولات
    """
    from src.adapters.llm.openai_llm import OpenAILLM
    from src.adapters.embeddings.openai_embeddings import OpenAIEmbeddings
    from src.adapters.vector.qdrant_store import QdrantVectorStore
    from src.adapters.rerank.cross_encoder import CrossEncoderReranker
    from src.adapters.rerank.noop_reranker import NoopReranker
    from src.adapters.cache.redis_cache import RedisCache
    from src.adapters.filestore.local_store import LocalFileStore
    from src.adapters.extraction.default_extractor import DefaultTextExtractor
    from src.adapters.queue.celery_queue import CeleryTaskQueue
    from src.workers.celery_app import celery_app
    
    from src.application.services.embedding_cache import CachedEmbeddings
    from src.application.use_cases.upload_document import UploadDocumentUseCase
    from src.application.use_cases.ask_question_hybrid import AskQuestionHybridUseCase
    
    # Note: In production, these would be real implementations
    # For now, we use placeholder adapters for some ports
    
    # =========================================================================
    # Infrastructure adapters
    # =========================================================================
    
    # Redis cache
    cache = RedisCache(settings.redis_url)
    
    # File store
    file_store = LocalFileStore(
        upload_dir=settings.upload_dir,
        max_mb=settings.max_upload_mb,
    )
    
    # Text extractor
    text_extractor = DefaultTextExtractor()
    
    # Task queue
    task_queue = CeleryTaskQueue(celery_app)
    
    # =========================================================================
    # LLM & Embeddings
    # =========================================================================
    
    # LLM
    llm = OpenAILLM(
        api_key=settings.openai_api_key or "",
        model=settings.openai_chat_model,
    )
    
    # Embeddings
    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key or "",
        model=settings.openai_embed_model,
    )
    
    # Cached embeddings wrapper
    cached_embeddings = CachedEmbeddings(
        embeddings=embeddings,
        cache=cache,
        ttl_seconds=settings.embedding_cache_ttl,
    )
    
    # =========================================================================
    # Vector store
    # =========================================================================
    
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection=settings.qdrant_collection,
        vector_size=settings.embedding_dim,
    )
    
    # =========================================================================
    # Reranker
    # =========================================================================
    
    if settings.rerank_backend == "cross_encoder":
        reranker = CrossEncoderReranker(
            model_name=settings.cross_encoder_model,
            device=settings.cross_encoder_device,
        )
    else:
        reranker = NoopReranker()
    
    # =========================================================================
    # Database repositories
    # =========================================================================
    
    import os
    use_real_db = os.environ.get("USE_REAL_DB", "false").lower() == "true"
    
    if use_real_db:
        # Production: Use real Postgres implementations
        from src.adapters.persistence.postgres import (
            PostgresDocumentRepo,
            PostgresChunkDedupRepo,
            PostgresChunkTextReader,
            PostgresKeywordStore,
        )
        
        document_repo = PostgresDocumentRepo()
        document_idempotency_repo = document_repo  # Same class implements both
        document_reader = document_repo  # Implements get_stored_file
        chunk_dedup_repo = PostgresChunkDedupRepo()
        chunk_text_reader = PostgresChunkTextReader()
        keyword_store = PostgresKeywordStore()
    else:
        # Development: Use in-memory placeholder implementations
        from src.adapters.persistence.placeholder import (
            PlaceholderDocumentRepo,
            PlaceholderDocumentIdempotencyRepo,
            PlaceholderDocumentReader,
            PlaceholderChunkDedupRepo,
            PlaceholderChunkTextReader,
            PlaceholderKeywordStore,
        )
        
        document_repo = PlaceholderDocumentRepo()
        document_idempotency_repo = PlaceholderDocumentIdempotencyRepo()
        document_reader = PlaceholderDocumentReader()
        chunk_dedup_repo = PlaceholderChunkDedupRepo()
        chunk_text_reader = PlaceholderChunkTextReader()
        keyword_store = PlaceholderKeywordStore()
    
    # =========================================================================
    # Use cases
    # =========================================================================
    
    upload_use_case = UploadDocumentUseCase(
        file_store=file_store,
        document_repo=document_repo,
        idempotency_repo=document_idempotency_repo,
        task_queue=task_queue,
    )
    
    ask_hybrid_use_case = AskQuestionHybridUseCase(
        cached_embeddings=cached_embeddings,
        vector_store=vector_store,
        keyword_store=keyword_store,
        chunk_text_reader=chunk_text_reader,
        reranker=reranker,
        llm=llm,
    )
    
    # =========================================================================
    # Build container
    # =========================================================================
    
    return {
        # Infrastructure
        "cache": cache,
        "file_store": file_store,
        "text_extractor": text_extractor,
        "task_queue": task_queue,
        
        # LLM & Embeddings
        "llm": llm,
        "embeddings": embeddings,
        "cached_embeddings": cached_embeddings,
        
        # Vector & Search
        "vector_store": vector_store,
        "reranker": reranker,
        "keyword_store": keyword_store,
        
        # Database repositories
        "document_repo": document_repo,
        "document_idempotency_repo": document_idempotency_repo,
        "document_reader": document_reader,
        "chunk_dedup_repo": chunk_dedup_repo,
        "chunk_text_reader": chunk_text_reader,
        
        # Use cases
        "upload_use_case": upload_use_case,
        "ask_hybrid_use_case": ask_hybrid_use_case,
    }


def reset_container() -> None:
    """Reset the cached container (for testing)."""
    get_container.cache_clear()
