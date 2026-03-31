"""
Embeddings Port
================
Interface for text embedding providers.

منفذ التضمينات النصية
"""

from typing import Protocol, Sequence


class EmbeddingsPort(Protocol):
    """
    Port for text embeddings generation.
    
    Implementations: OpenAI, SentenceTransformers, etc.
    
    Design Decision: Supporting both single and batch embedding
    for efficiency. Batch calls are much more cost-effective.
    
    قرار التصميم: دعم التضمين المفرد والدفعي للكفاءة
    """
    
    def embed_one(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            EmbeddingError: If embedding fails
        """
        ...
    
    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: Sequence of texts to embed
            
        Returns:
            List of embedding vectors (same order as input)
            
        Raises:
            EmbeddingError: If embedding fails
            
        Note:
            Batch embedding is more efficient and cost-effective.
            Use this when embedding multiple texts (e.g., chunks).
            
            التضمين الدفعي أكثر كفاءة وفعالية من حيث التكلفة
        """
        ...
