"""
OpenAI Embeddings Adapter
==========================
Implementation of EmbeddingsPort for OpenAI API.

محول تضمينات OpenAI
"""

from typing import Sequence

from openai import OpenAI

from src.domain.errors import EmbeddingError


class OpenAIEmbeddings:
    """
    OpenAI Embeddings adapter implementing EmbeddingsPort.
    
    محول تضمينات OpenAI
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
            timeout: Request timeout in seconds
        """
        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model
    
    def embed_one(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: On API errors
        """
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=text,
            )
            return response.data[0].embedding
            
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding error: {e}") from e
    
    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors (same order as input)
            
        Raises:
            EmbeddingError: On API errors
        """
        if not texts:
            return []
        
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=list(texts),
            )
            
            # OpenAI returns in same order as input
            return [d.embedding for d in response.data]
            
        except Exception as e:
            raise EmbeddingError(f"OpenAI batch embedding error: {e}") from e
