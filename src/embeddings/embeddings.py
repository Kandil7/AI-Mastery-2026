"""
Embeddings Module - AI-Mastery-2026

This module provides unified interfaces for generating embeddings from text and images.
It includes caching mechanisms for efficient embedding storage and retrieval.

Key Components:
- TextEmbedder: Generate embeddings from text using sentence-transformers
- ImageEmbedder: Generate embeddings from images using CLIP
- MultiModalEmbedder: Combined text and image embeddings
- EmbeddingCache: LRU cache with optional disk persistence

Author: AI-Mastery-2026
License: MIT
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import json
import os
from pathlib import Path
from collections import OrderedDict
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding models.
    
    Attributes:
        model_name: Name of the embedding model to use
        embedding_dim: Dimension of the output embeddings
        batch_size: Batch size for processing multiple inputs
        normalize: Whether to L2 normalize embeddings
        cache_dir: Directory for caching embeddings
        use_gpu: Whether to use GPU acceleration
    
    Example:
        >>> config = EmbeddingConfig(
        ...     model_name="all-MiniLM-L6-v2",
        ...     embedding_dim=384,
        ...     batch_size=32
        ... )
    """
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 32
    normalize: bool = True
    cache_dir: Optional[str] = None
    use_gpu: bool = False
    max_seq_length: int = 512


# =============================================================================
# Embedding Cache
# =============================================================================

class EmbeddingCache:
    """
    LRU Cache for embeddings with optional disk persistence.
    
    This cache stores embeddings keyed by a hash of the input text/image,
    reducing redundant computation for repeated inputs.
    
    Attributes:
        max_size: Maximum number of embeddings to cache in memory
        persist: Whether to persist cache to disk
        cache_path: Path to the cache file on disk
    
    Example:
        >>> cache = EmbeddingCache(max_size=1000, persist=True)
        >>> cache.set("hello world", np.array([0.1, 0.2, 0.3]))
        >>> embedding = cache.get("hello world")
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        persist: bool = False,
        cache_path: Optional[str] = None
    ):
        """
        Initialize the embedding cache.
        
        Args:
            max_size: Maximum number of entries in the cache
            persist: Whether to save cache to disk
            cache_path: Path to save the cache file
        """
        self.max_size = max_size
        self.persist = persist
        self.cache_path = cache_path or ".embedding_cache.pkl"
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0
        
        # Load existing cache if persistence is enabled
        if persist and os.path.exists(self.cache_path):
            self._load_cache()
    
    def _compute_key(self, text: str) -> str:
        """
        Compute a unique hash key for the input text.
        
        Args:
            text: Input text to hash
            
        Returns:
            MD5 hash of the input text
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache.
        
        Args:
            text: Input text to look up
            
        Returns:
            Cached embedding if found, None otherwise
        """
        key = self._compute_key(text)
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Input text (used as key)
            embedding: Embedding vector to cache
        """
        key = self._compute_key(text)
        
        # Remove oldest entry if cache is full
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = embedding
        self._cache.move_to_end(key)
        
        # Persist to disk if enabled
        if self.persist:
            self._save_cache()
    
    def get_batch(self, texts: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Retrieve multiple embeddings from cache.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary mapping texts to their cached embeddings (or None)
        """
        return {text: self.get(text) for text in texts}
    
    def set_batch(self, texts: List[str], embeddings: List[np.ndarray]) -> None:
        """
        Store multiple embeddings in cache.
        
        Args:
            texts: List of input texts
            embeddings: List of embedding vectors
        """
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        if self.persist and os.path.exists(self.cache_path):
            os.remove(self.cache_path)
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(dict(self._cache), f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                self._cache = OrderedDict(data)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate
        }
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, text: str) -> bool:
        return self._compute_key(text) in self._cache


# =============================================================================
# Base Embedder Class
# =============================================================================

class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.
    
    Provides a common interface for text and image embedding models.
    """
    
    @abstractmethod
    def encode(self, inputs: Union[str, List[str]]) -> np.ndarray:
        """
        Encode inputs into embeddings.
        
        Args:
            inputs: Single input or list of inputs
            
        Returns:
            Embedding array of shape (n_inputs, embedding_dim)
        """
        pass
    
    @abstractmethod
    def encode_batch(self, inputs: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode inputs in batches for efficiency.
        
        Args:
            inputs: List of inputs
            batch_size: Size of each batch
            
        Returns:
            Embedding array of shape (n_inputs, embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        pass


# =============================================================================
# Text Embedder
# =============================================================================

class TextEmbedder(BaseEmbedder):
    """
    Text embedding using sentence-transformers.
    
    This class provides a high-level interface for generating text embeddings
    using pre-trained sentence transformer models. It supports batching,
    caching, and GPU acceleration.
    
    Supported Models:
        - all-MiniLM-L6-v2: Fast, 384 dimensions
        - all-mpnet-base-v2: Best quality, 768 dimensions
        - paraphrase-multilingual-MiniLM-L12-v2: Multilingual, 384 dimensions
    
    Example:
        >>> embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
        >>> embeddings = embedder.encode(["Hello world", "AI is amazing"])
        >>> print(embeddings.shape)  # (2, 384)
    
    Attributes:
        model_name: Name of the sentence-transformers model
        config: EmbeddingConfig with model settings
        cache: Optional EmbeddingCache for caching results
    """
    
    # Model dimension mappings
    MODEL_DIMS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "all-distilroberta-v1": 768,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        config: Optional[EmbeddingConfig] = None,
        cache: Optional[EmbeddingCache] = None,
        use_gpu: bool = False
    ):
        """
        Initialize the TextEmbedder.
        
        Args:
            model_name: Name of the sentence-transformers model
            config: Optional EmbeddingConfig
            cache: Optional EmbeddingCache for caching
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.config = config or EmbeddingConfig(model_name=model_name)
        self.cache = cache
        self.use_gpu = use_gpu
        self._model = None
        self._dim = self.MODEL_DIMS.get(model_name, 384)
        
        logger.info(f"Initialized TextEmbedder with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                device = "cuda" if self.use_gpu else "cpu"
                self._model = SentenceTransformer(self.model_name, device=device)
                logger.info(f"Loaded model {self.model_name} on {device}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Using fallback TF-IDF embeddings."
                )
                self._model = "fallback"
    
    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """
        Fallback encoding using TF-IDF when sentence-transformers is unavailable.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            TF-IDF based embeddings
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=self._dim)
        tfidf_matrix = vectorizer.fit_transform(texts).toarray()
        
        # Pad or truncate to match expected dimensions
        if tfidf_matrix.shape[1] < self._dim:
            padding = np.zeros((tfidf_matrix.shape[0], self._dim - tfidf_matrix.shape[1]))
            tfidf_matrix = np.hstack([tfidf_matrix, padding])
        
        return tfidf_matrix.astype(np.float32)
    
    def encode(self, inputs: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            inputs: Single text or list of texts
            
        Returns:
            Embedding array of shape (n_inputs, embedding_dim)
            
        Example:
            >>> embedder = TextEmbedder()
            >>> emb = embedder.encode("Hello world")
            >>> print(emb.shape)  # (1, 384)
        """
        # Handle single string input
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Check cache first
        if self.cache:
            cached = self.cache.get_batch(inputs)
            uncached_texts = [t for t, e in cached.items() if e is None]
            
            if not uncached_texts:
                # All embeddings were cached
                return np.array([cached[t] for t in inputs])
        else:
            uncached_texts = inputs
        
        # Generate embeddings for uncached texts
        self._load_model()
        
        if self._model == "fallback":
            embeddings = self._fallback_encode(uncached_texts)
        else:
            embeddings = self._model.encode(
                uncached_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False
            )
        
        # Update cache
        if self.cache:
            for text, emb in zip(uncached_texts, embeddings):
                self.cache.set(text, emb)
            
            # Combine cached and new embeddings
            result = []
            emb_dict = dict(zip(uncached_texts, embeddings))
            for text in inputs:
                if cached.get(text) is not None:
                    result.append(cached[text])
                else:
                    result.append(emb_dict[text])
            return np.array(result)
        
        return np.array(embeddings)
    
    def encode_batch(
        self,
        inputs: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts in batches for memory efficiency.
        
        Args:
            inputs: List of texts to encode
            batch_size: Size of each batch
            show_progress: Whether to show progress bar
            
        Returns:
            Embedding array of shape (n_inputs, embedding_dim)
        """
        self._load_model()
        
        all_embeddings = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between -1 and 1
            
        Example:
            >>> embedder = TextEmbedder()
            >>> sim = embedder.similarity("cat", "dog")
            >>> print(f"Similarity: {sim:.3f}")
        """
        emb1 = self.encode(text1)[0]
        emb2 = self.encode(text2)[0]
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        return float(dot_product / (norm1 * norm2))
    
    def most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar texts to a query.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return
            
        Returns:
            List of (text, score) tuples sorted by similarity
            
        Example:
            >>> embedder = TextEmbedder()
            >>> results = embedder.most_similar(
            ...     "machine learning",
            ...     ["AI research", "cooking recipes", "deep learning"]
            ... )
        """
        query_emb = self.encode(query)[0]
        candidate_embs = self.encode(candidates)
        
        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb)
        
        # Sort by similarity
        indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(candidates[i], float(similarities[i])) for i in indices]
    
    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        return self._dim


# =============================================================================
# Image Embedder
# =============================================================================

class ImageEmbedder(BaseEmbedder):
    """
    Image embedding using CLIP (Contrastive Language-Image Pre-training).
    
    CLIP learns visual concepts from natural language supervision,
    allowing zero-shot image classification and powerful image embeddings.
    
    Example:
        >>> embedder = ImageEmbedder()
        >>> # From file path
        >>> embeddings = embedder.encode("path/to/image.jpg")
        >>> # From PIL Image
        >>> from PIL import Image
        >>> img = Image.open("path/to/image.jpg")
        >>> embeddings = embedder.encode(img)
    
    Attributes:
        model_name: Name of the CLIP model variant
        config: EmbeddingConfig with model settings
    """
    
    MODEL_DIMS = {
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "openai/clip-vit-base-patch32": 512,
    }
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        config: Optional[EmbeddingConfig] = None,
        use_gpu: bool = False
    ):
        """
        Initialize the ImageEmbedder.
        
        Args:
            model_name: Name of the CLIP model
            config: Optional EmbeddingConfig
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.config = config or EmbeddingConfig(model_name=model_name)
        self.use_gpu = use_gpu
        self._model = None
        self._processor = None
        self._dim = self.MODEL_DIMS.get(model_name, 512)
        
        logger.info(f"Initialized ImageEmbedder with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the CLIP model."""
        if self._model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch
                
                self._model = CLIPModel.from_pretrained(self.model_name)
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                
                if self.use_gpu and torch.cuda.is_available():
                    self._model = self._model.cuda()
                
                logger.info(f"Loaded CLIP model: {self.model_name}")
            except ImportError:
                logger.warning(
                    "transformers not installed. ImageEmbedder not available."
                )
                self._model = "unavailable"
    
    def _load_image(self, image_input: Union[str, Any]) -> Any:
        """
        Load image from file path or return PIL Image.
        
        Args:
            image_input: File path or PIL Image
            
        Returns:
            PIL Image object
        """
        from PIL import Image
        
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        return image_input.convert("RGB")
    
    def encode(self, inputs: Union[str, Any, List]) -> np.ndarray:
        """
        Encode image(s) into embeddings.
        
        Args:
            inputs: Single image (path or PIL) or list of images
            
        Returns:
            Embedding array of shape (n_inputs, embedding_dim)
        """
        import torch
        
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        self._load_model()
        
        if self._model == "unavailable":
            # Return random embeddings as fallback
            return np.random.randn(len(inputs), self._dim).astype(np.float32)
        
        # Load all images
        images = [self._load_image(img) for img in inputs]
        
        # Process images
        processed = self._processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        if self.use_gpu and torch.cuda.is_available():
            processed = {k: v.cuda() for k, v in processed.items()}
        
        # Get embeddings
        with torch.no_grad():
            image_features = self._model.get_image_features(**processed)
        
        embeddings = image_features.cpu().numpy()
        
        # Normalize if configured
        if self.config.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def encode_batch(self, inputs: List, batch_size: int = 16) -> np.ndarray:
        """
        Encode images in batches.
        
        Args:
            inputs: List of images
            batch_size: Size of each batch
            
        Returns:
            Embedding array
        """
        all_embeddings = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        return self._dim


# =============================================================================
# Multi-Modal Embedder
# =============================================================================

class MultiModalEmbedder:
    """
    Multi-modal embeddings combining text and images.
    
    Uses CLIP to project both text and images into the same embedding space,
    enabling cross-modal similarity search.
    
    Example:
        >>> embedder = MultiModalEmbedder()
        >>> text_emb = embedder.encode_text("a photo of a cat")
        >>> image_emb = embedder.encode_image("cat.jpg")
        >>> similarity = np.dot(text_emb[0], image_emb[0])
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        use_gpu: bool = False
    ):
        """
        Initialize the MultiModalEmbedder.
        
        Args:
            model_name: Name of the CLIP model
            use_gpu: Whether to use GPU
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._model = None
        self._processor = None
        self._dim = 512
        
        logger.info(f"Initialized MultiModalEmbedder with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the CLIP model."""
        if self._model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch
                
                self._model = CLIPModel.from_pretrained(self.model_name)
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                
                if self.use_gpu and torch.cuda.is_available():
                    self._model = self._model.cuda()
                    
            except ImportError:
                logger.warning("transformers not installed.")
                self._model = "unavailable"
    
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into CLIP embedding space.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Embedding array
        """
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
        
        self._load_model()
        
        if self._model == "unavailable":
            return np.random.randn(len(texts), self._dim).astype(np.float32)
        
        processed = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        if self.use_gpu:
            processed = {k: v.cuda() for k, v in processed.items()}
        
        with torch.no_grad():
            text_features = self._model.get_text_features(**processed)
        
        embeddings = text_features.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def encode_image(self, images: Union[str, Any, List]) -> np.ndarray:
        """
        Encode image(s) into CLIP embedding space.
        
        Args:
            images: Single image or list of images
            
        Returns:
            Embedding array
        """
        from PIL import Image
        import torch
        
        if not isinstance(images, list):
            images = [images]
        
        self._load_model()
        
        if self._model == "unavailable":
            return np.random.randn(len(images), self._dim).astype(np.float32)
        
        # Load images
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert("RGB"))
            else:
                loaded_images.append(img.convert("RGB"))
        
        processed = self._processor(
            images=loaded_images,
            return_tensors="pt",
            padding=True
        )
        
        if self.use_gpu:
            processed = {k: v.cuda() for k, v in processed.items()}
        
        with torch.no_grad():
            image_features = self._model.get_image_features(**processed)
        
        embeddings = image_features.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def text_image_similarity(self, text: str, image: Union[str, Any]) -> float:
        """
        Compute similarity between text and image.
        
        Args:
            text: Text description
            image: Image path or PIL Image
            
        Returns:
            Similarity score
        """
        text_emb = self.encode_text(text)[0]
        image_emb = self.encode_image(image)[0]
        
        return float(np.dot(text_emb, image_emb))
    
    def rank_images_by_text(
        self,
        text: str,
        images: List[Union[str, Any]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Rank images by similarity to text.
        
        Args:
            text: Query text
            images: List of images
            top_k: Number of results
            
        Returns:
            List of (image, score) tuples
        """
        text_emb = self.encode_text(text)[0]
        image_embs = self.encode_image(images)
        
        similarities = np.dot(image_embs, text_emb)
        indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(images[i], float(similarities[i])) for i in indices]
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._dim


# =============================================================================
# Utility Functions
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot_product / (norm_a * norm_b))


def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        embeddings: Embedding matrix of shape (n, dim)
        
    Returns:
        Similarity matrix of shape (n, n)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    
    # Compute similarity matrix
    return np.dot(normalized, normalized.T)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(a - b))


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Embeddings Module Demo")
    print("=" * 60)
    
    # 1. Text Embeddings
    print("\n1. Text Embeddings")
    print("-" * 40)
    
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "I love eating pizza on weekends."
    ]
    
    embeddings = embedder.encode(texts)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Compute similarities
    sim_12 = embedder.similarity(texts[0], texts[1])
    sim_13 = embedder.similarity(texts[0], texts[2])
    
    print(f"Similarity (ML vs DL): {sim_12:.4f}")
    print(f"Similarity (ML vs Pizza): {sim_13:.4f}")
    
    # 2. Caching
    print("\n2. Embedding Cache")
    print("-" * 40)
    
    cache = EmbeddingCache(max_size=100)
    cached_embedder = TextEmbedder(cache=cache)
    
    # First call (cache miss)
    _ = cached_embedder.encode("Hello world")
    print(f"After first call: {cache.stats}")
    
    # Second call (cache hit)
    _ = cached_embedder.encode("Hello world")
    print(f"After second call: {cache.stats}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
