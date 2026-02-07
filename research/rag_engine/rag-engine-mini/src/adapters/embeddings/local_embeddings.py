"""
Local Embeddings Adapter
=========================
Implementation of EmbeddingsPort using SentenceTransformers.

محول التضمينات المحلية باستخدام SentenceTransformers
"""

from typing import Sequence

from sentence_transformers import SentenceTransformer

from src.domain.errors import EmbeddingError


class LocalEmbeddings:
    """
    Local embeddings adapter using SentenceTransformers.
    
    No API costs, works offline, GPU acceleration available.
    
    Recommended models:
    - all-MiniLM-L6-v2 (fast, 384 dim)
    - all-mpnet-base-v2 (better quality, 768 dim)
    - nomic-embed-text-v1 (good quality, 768 dim)
    
    محول التضمينات المحلية - بدون تكلفة API
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        """
        Initialize local embedding model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on (cpu/cuda)
        """
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}") from e
    
    def embed_one(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Local embedding error: {e}") from e
    
    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            embeddings = self._model.encode(list(texts), convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            raise EmbeddingError(f"Local batch embedding error: {e}") from e
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._model.get_sentence_embedding_dimension()
