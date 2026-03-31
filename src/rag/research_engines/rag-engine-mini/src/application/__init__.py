"""Application layer package - use cases, ports, and services."""

from src.application.ports import (
    LLMPort,
    EmbeddingsPort,
    VectorStorePort,
    KeywordStorePort,
    DocumentRepoPort,
    DocumentIdempotencyPort,
    DocumentReaderPort,
    ChunkRepoPort,
    ChunkTextReaderPort,
    ChatRepoPort,
    CachePort,
    FileStorePort,
    TaskQueuePort,
    RerankerPort,
)

from src.application.use_cases import (
    UploadDocumentUseCase,
    UploadDocumentRequest,
    AskQuestionHybridUseCase,
    AskHybridRequest,
)

from src.application.services import (
    chunk_text_token_aware,
    build_rag_prompt,
    rrf_fusion,
    ScoredChunk,
    CachedEmbeddings,
    hydrate_chunk_texts,
)

__all__ = [
    # Ports
    "LLMPort",
    "EmbeddingsPort",
    "VectorStorePort",
    "KeywordStorePort",
    "DocumentRepoPort",
    "DocumentIdempotencyPort",
    "DocumentReaderPort",
    "ChunkRepoPort",
    "ChunkTextReaderPort",
    "ChatRepoPort",
    "CachePort",
    "FileStorePort",
    "TaskQueuePort",
    "RerankerPort",
    # Use Cases
    "UploadDocumentUseCase",
    "UploadDocumentRequest",
    "AskQuestionHybridUseCase",
    "AskHybridRequest",
    # Services
    "chunk_text_token_aware",
    "build_rag_prompt",
    "rrf_fusion",
    "ScoredChunk",
    "CachedEmbeddings",
    "hydrate_chunk_texts",
]
