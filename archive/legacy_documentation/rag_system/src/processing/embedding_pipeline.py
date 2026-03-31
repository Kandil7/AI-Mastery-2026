"""
Enhanced Embedding Pipeline - Production RAG 2026

Following RAG Pipeline Guide 2026 - Phase 3: Embeddings & Vector Databases

Features:
- Multi-model support (Sentence Transformers, OpenAI, Cohere, HF)
- Intelligent caching (memory + disk)
- Batch processing with progress tracking
- Cost tracking for API-based embeddings
- Fallback mechanisms
- Arabic-optimized models
- Embedding normalization and post-processing

Usage:
    pipeline = EmbeddingPipeline(model="multilingual")
    embeddings = pipeline.embed_texts(texts)
"""

import os
import json
import pickle
import hashlib
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple, Union, Literal
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ==================== Enums & Data Classes ====================


class EmbeddingProvider(Enum):
    """Supported embedding providers."""

    SENTENCE_TRANSFORMERS = "sentence_transformers"  # Local, free
    OPENAI = "openai"  # API, paid
    COHERE = "cohere"  # API, paid
    HUGGINGFACE = "huggingface"  # API, free/paid
    GOOGLE = "google"  # API, paid


class EmbeddingModel(Enum):
    """Pre-configured embedding models."""

    # Multilingual (Arabic-capable)
    MPNET_MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    BERT_MULTILINGUAL = "sentence-transformers/bert-base-multilingual-cased"
    LABSE = "sentence-transformers/LaBSE"

    # Arabic-specific
    ARABERT = "aubmindlab/bert-base-arabertv2"
    ARBERT = "UBC-NLP/ArBERTv2-base"
    MARBERT = "UBC-NLP/MARBERT"

    # OpenAI
    OPENAI_ADA2 = "text-embedding-ada-002"
    OPENAI_ADA3_SMALL = "text-embedding-3-small"
    OPENAI_ADA3_LARGE = "text-embedding-3-large"

    # Cohere
    COHERE_MULTILINGUAL = "embed-multilingual-v3.0"
    COHERE_ENGLISH = "embed-english-v3.0"

    # Google
    GOOGLE_EMBEDDING = "textembedding-gecko@001"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    # Model selection
    provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    model: EmbeddingModel = EmbeddingModel.MPNET_MULTILINGUAL

    # Batch processing
    batch_size: int = 32
    max_concurrent: int = 5

    # Caching
    cache_enabled: bool = True
    cache_dir: Optional[str] = None
    cache_ttl_hours: int = 168  # 1 week

    # Cost tracking
    track_costs: bool = True
    budget_limit_usd: Optional[float] = None

    # Post-processing
    normalize: bool = True
    reduce_dimension: Optional[int] = None  # PCA dimensionality reduction

    # Retry logic
    max_retries: int = 3
    retry_delay_seconds: int = 1


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embeddings: np.ndarray
    texts: List[str]
    model: str
    provider: str
    dimensions: int
    count: int
    latency_ms: float
    cost_usd: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class CostInfo:
    """Cost information for embedding generation."""

    provider: str
    model: str
    total_tokens: int
    total_cost_usd: float
    cost_per_1k_tokens: float


# Pricing information (as of 2026)
EMBEDDING_PRICES = {
    # OpenAI (per 1K tokens)
    "text-embedding-ada-002": 0.0001,
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,

    # Cohere (per 1K tokens)
    "embed-multilingual-v3.0": 0.0001,
    "embed-english-v3.0": 0.0001,

    # Google (per 1K characters)
    "textembedding-gecko@001": 0.000025,
}


# ==================== Cache System ====================


class EmbeddingCache:
    """
    Multi-level embedding cache.

    Levels:
    1. Memory cache (LRU, fast)
    2. Disk cache (persistent, slower)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_memory_items: int = 10000,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_items = max_memory_items

        # Memory cache (LRU)
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []

        # Disk cache index
        self._disk_index: Dict[str, str] = {}
        self._load_disk_index()

    def _load_disk_index(self):
        """Load disk cache index."""
        if not self.cache_dir:
            return

        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    self._disk_index = json.load(f)
                logger.info(f"Loaded disk cache index: {len(self._disk_index)} items")
            except Exception as e:
                logger.warning(f"Could not load disk cache index: {e}")
                self._disk_index = {}

    def _save_disk_index(self):
        """Save disk cache index."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        index_file = self.cache_dir / "index.json"

        try:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(self._disk_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not save disk cache index: {e}")

    def _generate_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""

        key = self._generate_key(text)

        # Try memory cache first
        if key in self._memory_cache:
            # Move to end (most recently used)
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._memory_cache[key]

        # Try disk cache
        if key in self._disk_index:
            embedding = self._load_from_disk(key)
            if embedding is not None:
                # Load into memory cache
                self._add_to_memory(key, embedding)
                return embedding

        return None

    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""

        key = self._generate_key(text)

        # Add to memory cache
        self._add_to_memory(key, embedding)

        # Add to disk cache
        self._save_to_disk(key, embedding)

    def _add_to_memory(self, key: str, embedding: np.ndarray):
        """Add to memory cache with LRU eviction."""

        if key in self._memory_cache:
            self._cache_order.remove(key)
        elif len(self._memory_cache) >= self.max_memory_items:
            # Evict oldest
            oldest_key = self._cache_order.pop(0)
            del self._memory_cache[oldest_key]

        self._memory_cache[key] = embedding
        self._cache_order.append(key)

    def _save_to_disk(self, key: str, embedding: np.ndarray):
        """Save to disk cache."""

        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Save embedding
        embedding_file = self.cache_dir / f"{key}.npy"
        try:
            np.save(embedding_file, embedding)
            self._disk_index[key] = f"{key}.npy"
            self._save_disk_index()
        except Exception as e:
            logger.warning(f"Could not save to disk cache: {e}")

    def _load_from_disk(self, key: str) -> Optional[np.ndarray]:
        """Load from disk cache."""

        if not self.cache_dir:
            return None

        filename = self._disk_index.get(key)
        if not filename:
            return None

        embedding_file = self.cache_dir / filename

        try:
            return np.load(embedding_file)
        except Exception as e:
            logger.warning(f"Could not load from disk cache: {e}")
            # Remove from index
            del self._disk_index[key]
            self._save_disk_index()
            return None

    def clear(self):
        """Clear all caches."""

        self._memory_cache.clear()
        self._cache_order.clear()

        if self.cache_dir:
            # Remove all .npy files
            for file in self.cache_dir.glob("*.npy"):
                file.unlink()
            # Remove index
            index_file = self.cache_dir / "index.json"
            if index_file.exists():
                index_file.unlink()

        self._disk_index.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""

        return {
            "memory_items": len(self._memory_cache),
            "disk_items": len(self._disk_index),
            "memory_size_mb": sum(
                e.nbytes for e in self._memory_cache.values()
            ) / (1024 * 1024),
        }


# ==================== Base Embedding Provider ====================


class BaseEmbeddingProvider:
    """Base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None

    async def embed_batch(
        self,
        texts: List[str],
    ) -> Tuple[np.ndarray, CostInfo]:
        """
        Generate embeddings for a batch of texts.

        Returns:
            Tuple of (embeddings, cost_info)
        """
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        raise NotImplementedError


# ==================== Sentence Transformers Provider ====================


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using Sentence Transformers.

    Supports:
    - Multilingual models
    - Arabic-specific models
    - CPU/GPU inference
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._model = None

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = self.config.model.value
                logger.info(f"Loading SentenceTransformer: {model_name}")

                self._model = SentenceTransformer(
                    model_name,
                    device="cuda" if self.config.batch_size > 32 else "cpu",
                )

                logger.info(f"Loaded model: {model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers torch torchvision"
                )
        return self._model

    async def embed_batch(
        self,
        texts: List[str],
    ) -> Tuple[np.ndarray, CostInfo]:
        """Generate embeddings locally."""

        import time
        start_time = time.time()

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Cost info (free for local)
        cost_info = CostInfo(
            provider="sentence_transformers",
            model=self.config.model.value,
            total_tokens=sum(len(t.split()) for t in texts),
            total_cost_usd=0.0,
            cost_per_1k_tokens=0.0,
        )

        return embeddings, cost_info

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


# ==================== OpenAI Provider ====================


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider.

    Models:
    - text-embedding-ada-002 (legacy)
    - text-embedding-3-small (cost-effective)
    - text-embedding-3-large (high quality)
    """

    def __init__(self, config: EmbeddingConfig, api_key: Optional[str] = None):
        super().__init__(config)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

    async def embed_batch(
        self,
        texts: List[str],
    ) -> Tuple[np.ndarray, CostInfo]:
        """Generate embeddings using OpenAI API."""

        import time
        start_time = time.time()

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key)

            # OpenAI has a limit of 2048 texts per batch
            batch_texts = texts[:2048]

            response = await client.embeddings.create(
                model=self.config.model.value,
                input=batch_texts,
            )

            # Extract embeddings
            embeddings = np.array([e.embedding for e in response.data])

            # Calculate cost
            usage = response.usage
            total_tokens = usage.total_tokens

            latency_ms = (time.time() - start_time) * 1000

            cost_per_1k = EMBEDDING_PRICES.get(self.config.model.value, 0.0001)
            total_cost = (total_tokens / 1000) * cost_per_1k

            cost_info = CostInfo(
                provider="openai",
                model=self.config.model.value,
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                cost_per_1k_tokens=cost_per_1k,
            )

            return embeddings, cost_info

        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""

        dimension_map = {
            EmbeddingModel.OPENAI_ADA2: 1536,
            EmbeddingModel.OPENAI_ADA3_SMALL: 1536,
            EmbeddingModel.OPENAI_ADA3_LARGE: 3072,
        }

        return dimension_map.get(self.config.model, 1536)


# ==================== Cohere Provider ====================


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """
    Cohere embedding provider.

    Models:
    - embed-multilingual-v3.0 (supports Arabic)
    - embed-english-v3.0
    """

    def __init__(self, config: EmbeddingConfig, api_key: Optional[str] = None):
        super().__init__(config)
        self.api_key = api_key or os.getenv("COHERE_API_KEY")

        if not self.api_key:
            raise ValueError("Cohere API key not provided")

    async def embed_batch(
        self,
        texts: List[str],
    ) -> Tuple[np.ndarray, CostInfo]:
        """Generate embeddings using Cohere API."""

        import time
        start_time = time.time()

        try:
            import cohere

            client = cohere.AsyncClient(api_key=self.api_key)

            response = await client.embed(
                texts=texts,
                model=self.config.model.value,
                input_type="search_document",
            )

            embeddings = np.array(response.embeddings)

            # Calculate cost
            total_tokens = sum(len(t.split()) for t in texts) * 1.3  # Estimate

            latency_ms = (time.time() - start_time) * 1000

            cost_per_1k = EMBEDDING_PRICES.get(self.config.model.value, 0.0001)
            total_cost = (total_tokens / 1000) * cost_per_1k

            cost_info = CostInfo(
                provider="cohere",
                model=self.config.model.value,
                total_tokens=int(total_tokens),
                total_cost_usd=total_cost,
                cost_per_1k_tokens=cost_per_1k,
            )

            return embeddings, cost_info

        except ImportError:
            raise ImportError("cohere not installed. Install with: pip install cohere")
        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return 1024  # Cohere multilingual v3


# ==================== Main Embedding Pipeline ====================


class EmbeddingPipeline:
    """
    Production embedding pipeline with multi-provider support.

    Features:
    - Multiple embedding providers
    - Intelligent caching
    - Batch processing
    - Cost tracking
    - Fallback mechanisms
    - Progress tracking
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()

        # Initialize cache
        self.cache = EmbeddingCache(
            cache_dir=self.config.cache_dir,
            max_memory_items=10000,
        )

        # Initialize provider
        self.provider = self._create_provider()

        # Cost tracking
        self._total_cost_usd = 0.0
        self._total_texts = 0

    def _create_provider(self) -> BaseEmbeddingProvider:
        """Create embedding provider based on config."""

        provider = self.config.provider

        if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return SentenceTransformersProvider(self.config)
        elif provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddingProvider(self.config)
        elif provider == EmbeddingProvider.COHERE:
            return CohereEmbeddingProvider(self.config)
        else:
            # Default to sentence transformers
            return SentenceTransformersProvider(self.config)

    async def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            EmbeddingResult with embeddings and metadata
        """

        import time
        start_time = time.time()

        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                texts=[],
                model=self.config.model.value,
                provider=self.config.provider.value,
                dimensions=0,
                count=0,
                latency_ms=0,
                cost_usd=0.0,
                cache_hits=0,
                cache_misses=0,
            )

        # Check cache
        cached_embeddings: Dict[int, np.ndarray] = {}
        uncached_texts: List[Tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                uncached_texts.append((i, text))

        cache_hits = len(cached_embeddings)
        cache_misses = len(uncached_texts)

        logger.info(f"Cache: {cache_hits} hits, {cache_misses} misses")

        # Generate embeddings for uncached texts
        new_embeddings: Dict[int, np.ndarray] = {}

        if uncached_texts:
            # Process in batches
            batch_size = self.config.batch_size
            total_cost = 0.0

            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch = uncached_texts[batch_start:batch_end]
                batch_texts = [text for _, text in batch]
                batch_indices = [idx for idx, _ in batch]

                # Generate embeddings
                embeddings, cost_info = await self.provider.embed_batch(batch_texts)

                # Store results
                for idx, emb in zip(batch_indices, embeddings):
                    new_embeddings[idx] = emb
                    # Cache the embedding
                    self.cache.set(batch_texts[batch_indices.index(idx)], emb)

                # Track cost
                total_cost += cost_info.total_cost_usd

            self._total_cost_usd += total_cost
            self._total_texts += len(uncached_texts)

            # Check budget
            if (
                self.config.track_costs
                and self.config.budget_limit_usd
                and self._total_cost_usd > self.config.budget_limit_usd
            ):
                logger.warning(
                    f"Budget exceeded: ${self._total_cost_usd:.2f} / ${self.config.budget_limit_usd:.2f}"
                )

        # Combine all embeddings
        all_embeddings = np.zeros(
            (len(texts), self.provider.dimension),
            dtype=np.float32,
        )

        for i in range(len(texts)):
            if i in cached_embeddings:
                all_embeddings[i] = cached_embeddings[i]
            elif i in new_embeddings:
                all_embeddings[i] = new_embeddings[i]

        # Apply post-processing
        if self.config.normalize:
            from numpy.linalg import norm
            all_embeddings = all_embeddings / norm(all_embeddings, axis=1, keepdims=True)

        if self.config.reduce_dimension:
            all_embeddings = self._reduce_dimension(
                all_embeddings,
                self.config.reduce_dimension,
            )

        latency_ms = (time.time() - start_time) * 1000

        return EmbeddingResult(
            embeddings=all_embeddings,
            texts=texts,
            model=self.config.model.value,
            provider=self.config.provider.value,
            dimensions=all_embeddings.shape[1],
            count=len(texts),
            latency_ms=latency_ms,
            cost_usd=self._total_cost_usd,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text (synchronous).

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """

        # Check cache
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        # Generate embedding (run async in sync context)
        import asyncio
        result = asyncio.run(self.embed_texts([text]))
        return result.embeddings[0]

    def _reduce_dimension(
        self,
        embeddings: np.ndarray,
        target_dim: int,
    ) -> np.ndarray:
        """Reduce embedding dimensionality using PCA."""

        try:
            from sklearn.decomposition import PCA

            if embeddings.shape[0] < target_dim:
                logger.warning(
                    f"Not enough samples for PCA ({embeddings.shape[0]} < {target_dim})"
                )
                return embeddings

            pca = PCA(n_components=target_dim)
            return pca.fit_transform(embeddings)

        except ImportError:
            logger.warning("scikit-learn not installed, skipping dimensionality reduction")
            return embeddings

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of chunk dictionaries with 'content' key
            show_progress: Whether to show progress

        Returns:
            List of chunks with added 'embedding' key
        """

        # Extract texts
        texts = [chunk.get("content", "") for chunk in chunks]

        # Generate embeddings
        import asyncio
        result = asyncio.run(self.embed_texts(texts, show_progress=show_progress))

        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = result.embeddings[i].tolist()

        return chunks

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        Args:
            query: Query text

        Returns:
            Query embedding
        """

        result = await self.embed_texts([query])
        return result.embeddings[0]

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""

        return {
            "provider": self.config.provider.value,
            "model": self.config.model.value,
            "dimension": self.provider.dimension,
            "batch_size": self.config.batch_size,
            "cache_stats": self.cache.stats(),
            "total_cost_usd": self._total_cost_usd,
            "total_texts_embedded": self._total_texts,
        }

    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")


# ==================== Factory Functions ====================


def create_embedding_pipeline(
    provider: str = "sentence_transformers",
    model: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> EmbeddingPipeline:
    """
    Create an embedding pipeline.

    Args:
        provider: Provider name (sentence_transformers, openai, cohere)
        model: Model name (optional, uses default for provider)
        cache_dir: Directory for embedding cache
        **kwargs: Additional config options

    Returns:
        EmbeddingPipeline instance
    """

    provider_map = {
        "sentence_transformers": EmbeddingProvider.SENTENCE_TRANSFORMERS,
        "openai": EmbeddingProvider.OPENAI,
        "cohere": EmbeddingProvider.COHERE,
    }

    model_map = {
        "sentence_transformers": EmbeddingModel.MPNET_MULTILINGUAL,
        "openai": EmbeddingModel.OPENAI_ADA3_SMALL,
        "cohere": EmbeddingModel.COHERE_MULTILINGUAL,
    }

    config = EmbeddingConfig(
        provider=provider_map.get(provider.lower(), EmbeddingProvider.SENTENCE_TRANSFORMERS),
        model=model_map.get(provider.lower(), EmbeddingModel.MPNET_MULTILINGUAL),
        cache_dir=cache_dir,
        **kwargs,
    )

    # Override model if specified
    if model:
        # Try to find matching model
        for m in EmbeddingModel:
            if model.lower() in m.value.lower():
                config.model = m
                break

    return EmbeddingPipeline(config)


def get_recommended_model(language: str = "ar") -> EmbeddingModel:
    """
    Get recommended embedding model for a language.

    Args:
        language: Language code (ar, en, multi)

    Returns:
        Recommended EmbeddingModel
    """

    recommendations = {
        "ar": EmbeddingModel.MARBERT,  # Arabic-specific
        "en": EmbeddingModel.OPENAI_ADA3_SMALL,  # English
        "multi": EmbeddingModel.MPNET_MULTILINGUAL,  # Multilingual
    }

    return recommendations.get(language.lower(), EmbeddingModel.MPNET_MULTILINGUAL)


# ==================== Main Entry ====================


if __name__ == "__main__":
    import asyncio

    async def main():
        """Demo embedding pipeline."""

        print("Embedding Pipeline - Demo")
        print("=" * 50)

        # Create pipeline with sentence transformers
        pipeline = create_embedding_pipeline(
            provider="sentence_transformers",
            cache_dir="rag_system/data/embedding_cache",
        )

        # Test texts
        texts = [
            "ما هو التوحيد في الإسلام؟",
            "Explain the concept of Tawhid",
            "ما حكم الزكاة في الإسلام؟",
        ]

        # Generate embeddings
        print(f"\nEmbedding {len(texts)} texts...")
        result = await pipeline.embed_texts(texts, show_progress=True)

        print(f"\nResults:")
        print(f"  Count: {result.count}")
        print(f"  Dimensions: {result.dimensions}")
        print(f"  Latency: {result.latency_ms:.2f}ms")
        print(f"  Cost: ${result.cost_usd:.4f}")
        print(f"  Cache hits: {result.cache_hits}")
        print(f"  Cache misses: {result.cache_misses}")

        # Get stats
        print(f"\nPipeline Stats:")
        stats = pipeline.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    asyncio.run(main())
