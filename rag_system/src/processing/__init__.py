"""
Text Processing Module

Handles text processing operations:
- Chunking (6 strategies)
- Embedding generation
- Arabic text processing
"""

from .advanced_chunker import (
    AdvancedChunker,
    create_chunker,
    ChunkingStrategy,
    Chunk,
    get_recommended_chunking,
)

from .embedding_pipeline import (
    EmbeddingPipeline,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingModel,
    create_embedding_pipeline,
    get_recommended_model,
)

from .arabic_processor import (
    ArabicChunker,
)

# IslamicChunker removed - use AdvancedChunker with strategy="islamic" instead

__all__ = [
    # Chunking
    "AdvancedChunker",
    "create_chunker",
    "ChunkingStrategy",
    "Chunk",
    "get_recommended_chunking",
    "ArabicChunker",
    # "IslamicChunker",  # Removed - use AdvancedChunker instead
    # Embeddings
    "EmbeddingPipeline",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingModel",
    "create_embedding_pipeline",
    "get_recommended_model",
]
