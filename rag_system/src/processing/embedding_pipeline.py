"""
Embedding generation pipeline for Arabic text.
"""

import os
import json
import pickle
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from datetime import datetime


class EmbeddingGenerator:
    """Generate embeddings for Arabic text using multilingual models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence transformer model
            batch_size: Batch size for embedding generation
            max_length: Maximum sequence length
            normalize: Whether to normalize embeddings
            cache_dir: Directory to cache embeddings
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._cache = {}

        # Load cache if exists
        if cache_dir and os.path.exists(
            os.path.join(cache_dir, "embeddings_cache.pkl")
        ):
            self._load_cache()

    def _load_cache(self):
        """Load embedding cache from disk."""
        cache_path = os.path.join(self.cache_dir, "embeddings_cache.pkl")
        try:
            with open(cache_path, "rb") as f:
                self._cache = pickle.load(f)
            print(f"Loaded {len(self._cache)} cached embeddings")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            self._cache = {}

    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self.cache_dir:
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, "embeddings_cache.pkl")

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            print(f"Saved {len(self._cache)} embeddings to cache")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
                print(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        cache_key = self._get_cache_key(text)

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate embedding
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        # Cache result
        self._cache[cache_key] = embedding

        return embedding

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings (num_texts x embedding_dim)
        """
        # Check which texts are cached
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key not in self._cache:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

            # Update cache
            for i, text in enumerate(uncached_texts):
                cache_key = self._get_cache_key(text)
                self._cache[cache_key] = embeddings[i]

        # Combine cached and new embeddings
        all_embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        # Fill in cached embeddings
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                all_embeddings[i] = self._cache[cache_key]

        # Fill in new embeddings
        for i, idx in enumerate(uncached_indices):
            all_embeddings[idx] = embeddings[i]

        return all_embeddings

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        Uses same method as embed_text but ensures proper handling.

        Args:
            query: Input query

        Returns:
            Query embedding
        """
        return self.embed_text(query)

    def batch_embed_chunks(
        self, chunks: List[Dict[str, Any]], show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Embed a list of text chunks with their metadata.

        Args:
            chunks: List of chunk dictionaries with 'content' key
            show_progress: Whether to show progress bar

        Returns:
            List of chunks with added 'embedding' key
        """
        if not chunks:
            return []

        # Extract texts
        texts = [chunk["content"] for chunk in chunks]

        # Generate embeddings
        embeddings = self.embed_texts(texts, show_progress=show_progress)

        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()

        return chunks

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache = {}
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, "embeddings_cache.pkl")
            if os.path.exists(cache_path):
                os.remove(cache_path)

    def __del__(self):
        """Save cache on deletion."""
        if self._cache:
            self._save_cache()


class EmbeddingCache:
    """Simple file-based embedding cache for large-scale processing."""

    def __init__(self, cache_dir: str):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Load index
        self.index_file = os.path.join(cache_dir, "index.json")
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load cache index."""
        if os.path.exists(self.index_file):
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"mappings": {}, "stats": {"count": 0, "size_mb": 0}}

    def _save_index(self):
        """Save cache index."""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def get(self, chunk_id: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            Embedding array or None if not cached
        """
        if chunk_id not in self.index["mappings"]:
            return None

        embedding_file = os.path.join(self.cache_dir, self.index["mappings"][chunk_id])

        if not os.path.exists(embedding_file):
            return None

        return np.load(embedding_file)

    def put(self, chunk_id: str, embedding: np.ndarray):
        """
        Store embedding in cache.

        Args:
            chunk_id: Unique chunk identifier
            embedding: Embedding array to cache
        """
        # Generate filename
        filename = f"{chunk_id}.npy"
        filepath = os.path.join(self.cache_dir, filename)

        # Save embedding
        np.save(filepath, embedding)

        # Update index
        self.index["mappings"][chunk_id] = filename
        self.index["stats"]["count"] += 1
        self._save_index()

    def exists(self, chunk_id: str) -> bool:
        """Check if embedding exists in cache."""
        return chunk_id in self.index["mappings"]

    def clear(self):
        """Clear all cached embeddings."""
        import shutil

        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
        self.index = {"mappings": {}, "stats": {"count": 0, "size_mb": 0}}
        self._save_index()


def get_embedding_model(
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device: str = "cpu",
    cache_dir: Optional[str] = None,
) -> EmbeddingGenerator:
    """
    Factory function to get an embedding model.

    Args:
        model_name: Name of the embedding model
        device: Device to use
        cache_dir: Directory for embedding cache

    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
    )
