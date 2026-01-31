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
    from src.adapters.rerank.noop_reranker import NoopReranker
    from src.adapters.cache.redis_cache import RedisCache
    from src.adapters.filestore.factory import create_file_store
    from src.adapters.extraction.default_extractor import DefaultTextExtractor
    from src.adapters.queue.celery_queue import CeleryTaskQueue
    from src.workers.celery_app import celery_app
    from src.adapters.persistence.placeholder import PlaceholderUserRepo
    
    from src.application.services.embedding_cache import CachedEmbeddings
    from src.application.use_cases.upload_document import UploadDocumentUseCase
    from src.application.use_cases.ask_question_hybrid import AskQuestionHybridUseCase
    from src.application.use_cases.search_documents import SearchDocumentsUseCase
    from src.application.use_cases.bulk_operations import BulkOperationsUseCase
    from src.application.use_cases.reindex_document import ReindexDocumentUseCase
    
    # Note: In production, these would be real implementations
    # For now, we use placeholder adapters for some ports
    
    # =========================================================================
    # Infrastructure adapters
    # =========================================================================
    
    # Redis cache
    cache = RedisCache(settings.redis_url)
    
    # File store
    file_store = create_file_store(settings)
    
    # Text extractor
    text_extractor = DefaultTextExtractor()
    
    # Task queue
    task_queue = CeleryTaskQueue(celery_app)
    
    # =========================================================================
    # LLM & Embeddings
    # =========================================================================
    
    # LLM Choice
    if settings.llm_backend == "ollama":
        from src.adapters.llm.ollama_llm import OllamaLLM
        llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
        )
    elif settings.llm_backend == "gemini":
        from src.adapters.llm.gemini_llm import GeminiLLM
        llm = GeminiLLM(
            api_key=settings.gemini_api_key or "",
            model_name=settings.gemini_model,
        )
    elif settings.llm_backend == "huggingface":
        from src.adapters.llm.huggingface_llm import HuggingFaceLLM
        llm = HuggingFaceLLM(
            api_key=settings.hf_api_key or "",
            model_name=settings.hf_model,
        )
    else:
        llm = OpenAILLM(
            api_key=settings.openai_api_key or "",
            model=settings.openai_chat_model,
        )
    
    # Embeddings Choice
    if settings.embeddings_backend == "local":
        from src.adapters.embeddings.local_embeddings import LocalEmbeddings
        embeddings = LocalEmbeddings(
            model_name=settings.local_embed_model,
            device=settings.local_embed_device,
        )
    else:
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
        try:
            from src.adapters.rerank.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker(
                model_name=settings.cross_encoder_model,
                device=settings.cross_encoder_device,
            )
        except ModuleNotFoundError:
            reranker = NoopReranker()
    elif settings.rerank_backend == "llm":
        from src.adapters.rerank.llm_reranker import LLMReranker
        reranker = LLMReranker(llm=llm)
    else:
        reranker = NoopReranker()
    
    # =========================================================================
    # Database repositories
    # =========================================================================
    
    if settings.use_real_db:
        # Production: Use real Postgres implementations
        from src.adapters.persistence.postgres import (
            PostgresDocumentRepo,
            PostgresChunkDedupRepo,
            PostgresChunkTextReader,
            PostgresKeywordStore,
            PostgresChatRepo,
            PostgresGraphRepo,
        )
        
        document_repo = PostgresDocumentRepo()
        document_idempotency_repo = document_repo  # Same class implements both
        document_reader = document_repo  # Implements get_stored_file
        chunk_dedup_repo = PostgresChunkDedupRepo()
        chunk_text_reader = PostgresChunkTextReader()
        keyword_store = PostgresKeywordStore()
        chat_repo = PostgresChatRepo()
        graph_repo = PostgresGraphRepo()
        user_repo = PlaceholderUserRepo()
    else:
        # Development: Use in-memory placeholder implementations
        from src.adapters.persistence.placeholder import (
            PlaceholderDocumentRepo,
            PlaceholderDocumentIdempotencyRepo,
            PlaceholderDocumentReader,
            PlaceholderChunkDedupRepo,
            PlaceholderChunkTextReader,
            PlaceholderKeywordStore,
            PlaceholderChatRepo,
            PlaceholderGraphRepo,
        )
        
        document_repo = PlaceholderDocumentRepo()
        document_idempotency_repo = PlaceholderDocumentIdempotencyRepo()
        document_reader = PlaceholderDocumentReader()
        chunk_dedup_repo = PlaceholderChunkDedupRepo()
        chunk_text_reader = PlaceholderChunkTextReader()
        keyword_store = PlaceholderKeywordStore()
        chat_repo = PlaceholderChatRepo()
        graph_repo = PlaceholderGraphRepo()
        user_repo = PlaceholderUserRepo()
    
    # Query Expansion
    from src.application.services.query_expansion import QueryExpansionService
    query_expansion_service = QueryExpansionService(llm=llm)
    
    # Self Critique & Self-RAG
    from src.application.services.self_critique import SelfCritiqueService
    self_critique_service = SelfCritiqueService(llm=llm)

    # Graph Extraction (Stage 3)
    from src.application.services.graph_extractor import GraphExtractorService
    graph_extractor_service = GraphExtractorService(llm=llm)

    # Vision Service (Stage 4)
    from src.application.services.vision_service import VisionService
    vision_service = VisionService(llm=llm)

    # Stage 5: Autonomy & Optimization
    from src.application.services.semantic_router import SemanticRouterService
    from src.application.services.privacy_guard import PrivacyGuardService
    from src.adapters.tools.web_search import TavilySearchAdapter
    
    semantic_router = SemanticRouterService(llm=llm)
    privacy_guard = PrivacyGuardService()
    search_tool = TavilySearchAdapter(api_key=settings.tavily_api_key or "")
    
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
        query_expansion_service=query_expansion_service,
        self_critique=self_critique_service,
        router=semantic_router,
        privacy=privacy_guard,
        search_tool=search_tool,
    )

    search_documents_use_case = SearchDocumentsUseCase(
        document_repo=document_repo,
    )

    bulk_operations_use_case = BulkOperationsUseCase(
        upload_use_case=upload_use_case,
        file_store=file_store,
        document_repo=document_repo,
        task_queue=task_queue,
    )

    reindex_document_use_case = ReindexDocumentUseCase(
        document_repo=document_repo,
        task_queue=task_queue,
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
        "chat_repo": chat_repo,
        "graph_repo": graph_repo,
        "user_repo": user_repo,

        # Services
        "graph_extractor": graph_extractor_service,
        "vision_service": vision_service,
        "router": semantic_router,
        "privacy": privacy_guard,
        
        # Use cases
        "upload_use_case": upload_use_case,
        "ask_hybrid_use_case": ask_hybrid_use_case,
        "search_documents_use_case": search_documents_use_case,
        "bulk_operations_use_case": bulk_operations_use_case,
        "reindex_document_use_case": reindex_document_use_case,
    }


def reset_container() -> None:
    """Reset the cached container (for testing)."""
    get_container.cache_clear()
