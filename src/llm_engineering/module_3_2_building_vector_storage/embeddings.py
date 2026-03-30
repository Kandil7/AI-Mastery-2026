"""
Embeddings Module

Production-ready embedding generation with:
- Multiple embedding model support
- Batch processing
- Caching and persistence
- Async operations
- Cost optimization
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embedding: List[float]
    model: str
    input_text: str
    tokens: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding": self.embedding,
            "model": self.model,
            "input_text": self.input_text,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
        }


@dataclass
class EmbeddingCacheEntry:
    """Cache entry for embeddings."""

    embedding: List[float]
    model: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding": self.embedding,
            "model": self.model,
            "created_at": self.created_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingCacheEntry":
        return cls(
            embedding=data["embedding"],
            model=data["model"],
            created_at=data.get("created_at", time.time()),
            access_count=data.get("access_count", 0),
        )


class EmbeddingCache:
    """
    Cache for embeddings to avoid redundant API calls.

    Features:
    - In-memory LRU cache
    - Optional disk persistence
    - TTL-based expiration
    - Statistics tracking
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: Optional[int] = None,
        persist_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.persist_path = Path(persist_path) if persist_path else None

        self._cache: Dict[str, EmbeddingCacheEntry] = {}
        self._access_order: List[str] = []
        self._hits = 0
        self._misses = 0

        if self.persist_path:
            self._load_cache()

    def _get_key(self, text: str, model: str) -> str:
        """Generate cache key."""
        return hashlib.sha256(f"{text}:{model}".encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._get_key(text, model)

        if key in self._cache:
            entry = self._cache[key]

            # Check TTL
            if self.ttl_seconds:
                if time.time() - entry.created_at > self.ttl_seconds:
                    del self._cache[key]
                    self._access_order.remove(key)
                    self._misses += 1
                    return None

            # Update access
            entry.access_count += 1
            self._access_order.remove(key)
            self._access_order.append(key)
            self._hits += 1

            return entry.embedding

        self._misses += 1
        return None

    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        key = self._get_key(text, model)

        # Evict if necessary
        if len(self._cache) >= self.max_size:
            self._evict()

        entry = EmbeddingCacheEntry(embedding=embedding, model=model)
        self._cache[key] = entry
        self._access_order.append(key)

    def _evict(self) -> None:
        """Evict oldest entry."""
        if self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_order.clear()

    def save(self) -> None:
        """Persist cache to disk."""
        if not self.persist_path:
            return

        data = {
            key: entry.to_dict()
            for key, entry in self._cache.items()
        }

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f)

        logger.info(f"Saved {len(self._cache)} cache entries to {self.persist_path}")

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)

            for key, entry_data in data.items():
                self._cache[key] = EmbeddingCacheEntry.from_dict(entry_data)
                self._access_order.append(key)

            logger.info(f"Loaded {len(self._cache)} cache entries from {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }


class BaseEmbeddingGenerator(ABC):
    """Abstract base class for embedding generators."""

    def __init__(
        self,
        model: str,
        batch_size: int = 32,
        max_retries: int = 3,
        cache: Optional[EmbeddingCache] = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.cache = cache

        self._total_embeddings = 0
        self._total_tokens = 0
        self._total_latency = 0.0

    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for batch of texts."""
        pass

    async def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        Generate embeddings with automatic batching.

        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar

        Returns:
            Single result or list of results
        """
        if isinstance(texts, str):
            return await self.embed_text(texts)

        # Check cache first
        if self.cache:
            cached_results = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.model)
                if cached:
                    cached_results.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Generate embeddings for uncached texts
            if uncached_texts:
                results = await self.embed_batch(uncached_texts)

                # Store in cache
                for text, result in zip(uncached_texts, results):
                    self.cache.set(text, self.model, result.embedding)

                # Merge results
                all_results = [None] * len(texts)
                for i, embedding in cached_results:
                    all_results[i] = EmbeddingResult(
                        embedding=embedding,
                        model=self.model,
                        input_text=texts[i],
                    )
                for i, result in zip(uncached_indices, results):
                    all_results[i] = result

                return all_results

            # All cached
            return [
                EmbeddingResult(
                    embedding=emb,
                    model=self.model,
                    input_text=texts[i],
                )
                for i, emb in cached_results
            ]

        # No cache, process in batches
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = await self.embed_batch(batch)
            results.extend(batch_results)

            if show_progress:
                logger.info(f"Progress: {i + len(batch)}/{len(texts)}")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        return {
            "model": self.model,
            "total_embeddings": self._total_embeddings,
            "total_tokens": self._total_tokens,
            "total_latency_ms": self._total_latency,
            "avg_latency_ms": self._total_latency / self._total_embeddings if self._total_embeddings > 0 else 0,
        }


class SentenceTransformerEmbeddings(BaseEmbeddingGenerator):
    """
    Sentence Transformers embedding generator.

    Uses Hugging Face sentence-transformers for local embedding generation.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)

        self.device = device
        self._model = None
        self._tokenizer = None

        self._load_model()

    def _load_model(self) -> None:
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model, device=self.device)
            logger.info(f"Loaded sentence transformer: {self.model}")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for single text."""
        start_time = time.time()

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, convert_to_numpy=True).tolist(),
        )

        latency_ms = (time.time() - start_time) * 1000

        self._total_embeddings += 1
        self._total_latency += latency_ms

        return EmbeddingResult(
            embedding=embedding,
            model=self.model,
            input_text=text,
            latency_ms=latency_ms,
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for batch of texts."""
        start_time = time.time()

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist(),
        )

        latency_ms = (time.time() - start_time) * 1000

        results = [
            EmbeddingResult(
                embedding=emb,
                model=self.model,
                input_text=text,
                latency_ms=latency_ms / len(texts),
            )
            for text, emb in zip(texts, embeddings)
        ]

        self._total_embeddings += len(texts)
        self._total_latency += latency_ms

        return results

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return 384  # Default for many models


class OpenAIEmbeddings(BaseEmbeddingGenerator):
    """
    OpenAI embedding generator.

    Supports text-embedding-ada-002 and newer models.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.dimensions = dimensions

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for single text."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for batch of texts."""
        start_time = time.time()

        # Prepare request
        payload = {
            "model": self.model,
            "input": texts,
        }

        if self.dimensions:
            payload["dimensions"] = self.dimensions

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        data = response.json()
        latency_ms = (time.time() - start_time) * 1000

        # Parse results
        results = []
        for item in data["data"]:
            result = EmbeddingResult(
                embedding=item["embedding"],
                model=data["model"],
                input_text=texts[item["index"]],
                tokens=item.get("embedding_tokens", 0),
                latency_ms=latency_ms / len(texts),
            )
            results.append(result)

            self._total_embeddings += 1
            self._total_tokens += result.tokens

        self._total_latency += latency_ms

        return results

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "OpenAIEmbeddings":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class HuggingFaceEmbeddings(BaseEmbeddingGenerator):
    """
    Hugging Face Inference API embedding generator.

    Uses Hugging Face's hosted inference API for embeddings.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)

        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.base_url = "https://api-inference.huggingface.co/models"

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                "Content-Type": "application/json",
            },
        )

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for single text."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for batch of texts."""
        start_time = time.time()

        results = []
        for text in texts:
            try:
                response = await self._client.post(
                    f"{self.base_url}/{self.model}",
                    json={"inputs": text},
                )
                response.raise_for_status()
                embedding = response.json()[0]

                results.append(EmbeddingResult(
                    embedding=embedding,
                    model=self.model,
                    input_text=text,
                    latency_ms=(time.time() - start_time) * 1000 / len(texts),
                ))
            except Exception as e:
                logger.warning(f"Failed to embed text: {e}")
                # Return zero vector as fallback
                results.append(EmbeddingResult(
                    embedding=[0.0] * 384,
                    model=self.model,
                    input_text=text,
                    latency_ms=0,
                ))

        self._total_embeddings += len(results)
        self._total_latency += (time.time() - start_time) * 1000

        return results

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()


class EmbeddingRouter:
    """
    Router for embedding requests across multiple providers.

    Features:
    - Automatic fallback on failure
    - Cost-based routing
    - Quality-based routing
    """

    def __init__(
        self,
        generators: Dict[str, BaseEmbeddingGenerator],
        default: str = "local",
    ) -> None:
        self.generators = generators
        self.default = default
        self._failure_counts: Dict[str, int] = {name: 0 for name in generators}

    async def embed(
        self,
        texts: Union[str, List[str]],
        priority: str = "cost",  # cost, quality, speed
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        Route embedding request to appropriate generator.

        Args:
            texts: Text(s) to embed
            priority: Routing priority

        Returns:
            Embedding result(s)
        """
        # Select generator based on priority
        generator_name = self._select_generator(priority)
        generator = self.generators.get(generator_name)

        if not generator:
            generator = self.generators[self.default]

        try:
            return await generator.embed(texts)
        except Exception as e:
            logger.warning(f"Generator {generator_name} failed: {e}")
            self._failure_counts[generator_name] += 1

            # Try fallback
            return await self._fallback_embed(texts, exclude=generator_name)

    def _select_generator(self, priority: str) -> str:
        """Select generator based on priority."""
        if priority == "cost":
            # Prefer local models
            for name, gen in self.generators.items():
                if isinstance(gen, SentenceTransformerEmbeddings):
                    return name
        elif priority == "quality":
            # Prefer OpenAI/large models
            for name, gen in self.generators.items():
                if isinstance(gen, OpenAIEmbeddings):
                    return name
        elif priority == "speed":
            # Prefer cached or small models
            for name, gen in self.generators.items():
                if gen.cache and gen.cache.get_stats()["hit_rate"] > 0.5:
                    return name

        return self.default

    async def _fallback_embed(
        self,
        texts: Union[str, List[str]],
        exclude: str,
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Try fallback generators."""
        for name, generator in self.generators.items():
            if name != exclude and self._failure_counts[name] < 3:
                try:
                    return await generator.embed(texts)
                except Exception as e:
                    logger.warning(f"Fallback {name} failed: {e}")
                    self._failure_counts[name] += 1

        raise RuntimeError("All embedding generators failed")


class EmbeddingPipeline:
    """
    Pipeline for processing documents through embedding generation.

    Features:
    - Batch processing
    - Progress tracking
    - Error handling
    - Statistics
    """

    def __init__(
        self,
        embedding_generator: BaseEmbeddingGenerator,
        batch_size: int = 32,
        max_concurrent: int = 5,
    ) -> None:
        self.embedding_generator = embedding_generator
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent

        self._processed = 0
        self._errors = 0
        self._results: List[EmbeddingResult] = []

    async def process(
        self,
        documents: List[Any],
        content_key: str = "content",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process documents through embedding pipeline.

        Args:
            documents: List of documents
            content_key: Key for content in document
            progress_callback: Optional progress callback

        Returns:
            List of documents with embeddings
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_document(doc: Any) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    content = doc.get(content_key) if isinstance(doc, dict) else getattr(doc, content_key, None)

                    if not content:
                        return None

                    result = await self.embedding_generator.embed_text(str(content))

                    # Add embedding to document
                    if isinstance(doc, dict):
                        doc["embedding"] = result.embedding
                        doc["embedding_model"] = result.model
                    else:
                        doc.embedding = result.embedding
                        doc.embedding_model = result.model

                    self._processed += 1
                    self._results.append(result)

                    if progress_callback:
                        progress_callback(self._processed, len(documents))

                    return doc

                except Exception as e:
                    logger.error(f"Failed to embed document: {e}")
                    self._errors += 1
                    return None

        tasks = [process_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks)

        return [r for r in results if r is not None]

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "processed": self._processed,
            "errors": self._errors,
            "success_rate": self._processed / (self._processed + self._errors) if (self._processed + self._errors) > 0 else 0,
            "embedding_stats": self.embedding_generator.get_stats(),
        }


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        return vector
    return [x / norm for x in vector]
