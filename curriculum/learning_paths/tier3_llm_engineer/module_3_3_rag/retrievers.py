"""
RAG Retrievers Module

Production-ready retrieval strategies:
- Similarity search
- Multi-query retrieval
- HyDE (Hypothetical Document Embeddings)
- Contextual compression
- Ensemble retrieval

Features:
- Async operations
- Metadata filtering
- Score thresholding
- Result reranking
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    def __init__(
        self,
        vector_store: Any,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.filter = filter

        self._stats = {
            "total_queries": 0,
            "avg_results": 0.0,
            "avg_latency_ms": 0.0,
            "total_latency_ms": 0.0,
        }

    @abstractmethod
    async def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve documents for query."""
        pass

    async def retrieve_with_scores(
        self,
        query: str,
    ) -> List[RetrievalResult]:
        """Retrieve with score filtering."""
        results = await self.retrieve(query)

        # Apply score threshold
        filtered = [r for r in results if r.score >= self.score_threshold]

        # Update stats
        self._stats["total_queries"] += 1
        self._stats["avg_results"] = (
            (self._stats["avg_results"] * (self._stats["total_queries"] - 1) + len(filtered))
            / self._stats["total_queries"]
        )

        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return self._stats.copy()


class SimilarityRetriever(BaseRetriever):
    """
    Standard similarity-based retriever.

    Uses vector similarity (cosine, dot product, etc.)
    to find relevant documents.
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_generator: Any,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
        collection: str = "default",
    ) -> None:
        super().__init__(vector_store, top_k, score_threshold, filter)
        self.embedding_generator = embedding_generator
        self.collection = collection

    async def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve documents using similarity search."""
        import time
        start_time = time.time()

        # Generate query embedding
        embedding_result = await self.embedding_generator.embed_text(query)

        # Search vector store
        search_result = await self.vector_store.search(
            collection=self.collection,
            query_vector=embedding_result.embedding,
            top_k=self.top_k,
            filter=self.filter,
        )

        # Convert to retrieval results
        results = [
            RetrievalResult(
                content=record.metadata.get('content', ''),
                score=record.score,
                metadata=record.metadata,
                id=record.id,
            )
            for record in search_result.records
        ]

        latency_ms = (time.time() - start_time) * 1000
        self._stats["total_latency_ms"] += latency_ms
        if self._stats["total_queries"] > 0:
            self._stats["avg_latency_ms"] = self._stats["total_latency_ms"] / self._stats["total_queries"]

        logger.debug(f"Similarity retrieval: {len(results)} results in {latency_ms:.2f}ms")
        return results


class MultiQueryRetriever(BaseRetriever):
    """
    Multi-query retriever.

    Generates multiple variations of the query and
    combines results for better coverage.
    """

    DEFAULT_QUERY_PROMPT = """Generate 3 different versions of the following question.
Each version should ask the same thing but use different wording.

Original question: {query}

Generate 3 variations (one per line):"""

    def __init__(
        self,
        vector_store: Any,
        embedding_generator: Any,
        llm_client: Any,
        top_k: int = 5,
        num_queries: int = 3,
        score_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
        collection: str = "default",
        deduplicate: bool = True,
    ) -> None:
        super().__init__(vector_store, top_k, score_threshold, filter)
        self.embedding_generator = embedding_generator
        self.llm_client = llm_client
        self.num_queries = num_queries
        self.collection = collection
        self.deduplicate = deduplicate

    async def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve using multiple query variations."""
        import time
        start_time = time.time()

        # Generate query variations
        variations = await self._generate_variations(query)
        all_queries = [query] + variations

        logger.info(f"Multi-query retrieval with {len(all_queries)} queries")

        # Retrieve for each query
        all_results = []
        for q in all_queries:
            try:
                embedding_result = await self.embedding_generator.embed_text(q)
                search_result = await self.vector_store.search(
                    collection=self.collection,
                    query_vector=embedding_result.embedding,
                    top_k=self.top_k,
                    filter=self.filter,
                )

                for record in search_result.records:
                    all_results.append(RetrievalResult(
                        content=record.metadata.get('content', ''),
                        score=record.score,
                        metadata=record.metadata,
                        id=record.id,
                    ))
            except Exception as e:
                logger.warning(f"Query failed: {e}")

        # Deduplicate and rerank
        if self.deduplicate:
            all_results = self._deduplicate(all_results)

        # Sort by score and take top_k
        all_results.sort(key=lambda x: x.score, reverse=True)
        results = all_results[:self.top_k]

        latency_ms = (time.time() - start_time) * 1000
        self._stats["total_latency_ms"] += latency_ms
        self._stats["total_queries"] += 1

        logger.debug(f"Multi-query retrieval: {len(results)} results in {latency_ms:.2f}ms")
        return results

    async def _generate_variations(self, query: str) -> List[str]:
        """Generate query variations using LLM."""
        prompt = self.DEFAULT_QUERY_PROMPT.format(query=query)

        try:
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )

            content = response.content if hasattr(response, 'content') else str(response)
            variations = [line.strip() for line in content.split('\n') if line.strip()]
            return variations[:self.num_queries]
        except Exception as e:
            logger.warning(f"Failed to generate variations: {e}")
            return []

    def _deduplicate(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Deduplicate results by ID or content."""
        seen_ids = set()
        unique = []

        for result in results:
            if result.id and result.id not in seen_ids:
                seen_ids.add(result.id)
                unique.append(result)
            elif not result.id:
                # Check content hash
                content_hash = hash(result.content)
                if content_hash not in seen_ids:
                    seen_ids.add(content_hash)
                    unique.append(result)

        return unique


class HyDERetriever(BaseRetriever):
    """
    HyDE (Hypothetical Document Embeddings) retriever.

    Generates a hypothetical answer and uses its embedding
    for retrieval, which can improve recall.
    """

    DEFAULT_HYDE_PROMPT = """Please write a passage to answer the question.
Be specific and detailed. Include relevant facts and information.

Question: {query}

Passage:"""

    def __init__(
        self,
        vector_store: Any,
        embedding_generator: Any,
        llm_client: Any,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
        collection: str = "default",
        hyde_weight: float = 0.5,
    ) -> None:
        super().__init__(vector_store, top_k, score_threshold, filter)
        self.embedding_generator = embedding_generator
        self.llm_client = llm_client
        self.collection = collection
        self.hyde_weight = hyde_weight

    async def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve using HyDE approach."""
        import time
        start_time = time.time()

        # Generate hypothetical document
        hyde_text = await self._generate_hyde(query)

        # Get embeddings for both
        query_embedding = await self.embedding_generator.embed_text(query)
        hyde_embedding = await self.embedding_generator.embed_text(hyde_text)

        # Search with both embeddings
        query_results = await self._search_with_embedding(query_embedding.embedding)
        hyde_results = await self._search_with_embedding(hyde_embedding.embedding)

        # Combine results with weighted scoring
        combined = self._combine_results(query_results, hyde_results)

        latency_ms = (time.time() - start_time) * 1000
        self._stats["total_latency_ms"] += latency_ms
        self._stats["total_queries"] += 1

        logger.debug(f"HyDE retrieval: {len(combined)} results in {latency_ms:.2f}ms")
        return combined

    async def _generate_hyde(self, query: str) -> str:
        """Generate hypothetical document."""
        prompt = self.DEFAULT_HYDE_PROMPT.format(query=query)

        try:
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return query

    async def _search_with_embedding(
        self,
        embedding: List[float],
    ) -> List[RetrievalResult]:
        """Search with a single embedding."""
        search_result = await self.vector_store.search(
            collection=self.collection,
            query_vector=embedding,
            top_k=self.top_k,
            filter=self.filter,
        )

        return [
            RetrievalResult(
                content=record.metadata.get('content', ''),
                score=record.score,
                metadata=record.metadata,
                id=record.id,
            )
            for record in search_result.records
        ]

    def _combine_results(
        self,
        query_results: List[RetrievalResult],
        hyde_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Combine results from both searches."""
        # Create score maps
        query_scores = {r.id: r.score for r in query_results if r.id}
        hyde_scores = {r.id: r.score for r in hyde_results if r.id}

        # Combine scores
        all_ids = set(query_scores.keys()) | set(hyde_scores.keys())
        combined = []

        for doc_id in all_ids:
            query_score = query_scores.get(doc_id, 0)
            hyde_score = hyde_scores.get(doc_id, 0)

            # Weighted combination
            combined_score = (
                self.hyde_weight * hyde_score +
                (1 - self.hyde_weight) * query_score
            )

            # Find original result for metadata
            original = next(
                (r for r in query_results + hyde_results if r.id == doc_id),
                None,
            )

            if original:
                combined.append(RetrievalResult(
                    content=original.content,
                    score=combined_score,
                    metadata=original.metadata,
                    id=doc_id,
                ))

        # Sort and return top_k
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:self.top_k]


class ContextualCompressionRetriever(BaseRetriever):
    """
    Contextual compression retriever.

    Retrieves documents and then compresses them to
    keep only the most relevant parts.
    """

    DEFAULT_COMPRESSION_PROMPT = """Extract only the most relevant sentences from the following text that relate to the query.
Keep the original wording. Return only the extracted sentences.

Query: {query}

Text: {text}

Relevant sentences:"""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_client: Any,
        compression_ratio: float = 0.3,
        min_sentences: int = 1,
    ) -> None:
        # Pass None for vector_store as we use base_retriever
        super().__init__(None, base_retriever.top_k, base_retriever.score_threshold)
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.compression_ratio = compression_ratio
        self.min_sentences = min_sentences

    async def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve and compress documents."""
        import time
        start_time = time.time()

        # First retrieve normally
        results = await self.base_retriever.retrieve(query)

        # Then compress each result
        compressed_results = []
        for result in results:
            compressed_content = await self._compress_text(query, result.content)
            if compressed_content.strip():
                compressed_results.append(RetrievalResult(
                    content=compressed_content,
                    score=result.score,
                    metadata={**result.metadata, "original_length": len(result.content)},
                    id=result.id,
                ))

        latency_ms = (time.time() - start_time) * 1000
        self._stats["total_latency_ms"] += latency_ms
        self._stats["total_queries"] += 1

        logger.debug(f"Compression retrieval: {len(compressed_results)} results in {latency_ms:.2f}ms")
        return compressed_results

    async def _compress_text(self, query: str, text: str) -> str:
        """Compress text to relevant parts."""
        prompt = self.DEFAULT_COMPRESSION_PROMPT.format(query=query, text=text)

        try:
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=int(len(text) * self.compression_ratio),
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return text  # Return original on failure


class EnsembleRetriever(BaseRetriever):
    """
    Ensemble retriever combining multiple retrieval strategies.

    Uses reciprocal rank fusion or weighted scoring to
    combine results from multiple retrievers.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        fusion_method: str = "rrf",  # rrf, weighted
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        super().__init__(None, top_k)
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    async def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve using ensemble of retrievers."""
        import time
        start_time = time.time()

        # Get results from all retrievers
        all_results = []
        for retriever in self.retrievers:
            try:
                results = await retriever.retrieve(query)
                all_results.append(results)
            except Exception as e:
                logger.warning(f"Retriever failed: {e}")
                all_results.append([])

        # Fuse results
        if self.fusion_method == "rrf":
            fused = self._reciprocal_rank_fusion(all_results)
        else:
            fused = self._weighted_fusion(all_results)

        latency_ms = (time.time() - start_time) * 1000
        self._stats["total_latency_ms"] += latency_ms
        self._stats["total_queries"] += 1

        logger.debug(f"Ensemble retrieval: {len(fused)} results in {latency_ms:.2f}ms")
        return fused

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
    ) -> List[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                doc_id = result.id or hash(result.content)
                score = 1 / (self.rrf_k + rank + 1)

                if doc_id in scores:
                    scores[doc_id] += score
                else:
                    scores[doc_id] = score
                    result_map[doc_id] = result

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused = []
        for doc_id in sorted_ids[:self.top_k]:
            result = result_map[doc_id]
            result.score = scores[doc_id]
            fused.append(result)

        return fused

    def _weighted_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
    ) -> List[RetrievalResult]:
        """Fuse results using weighted scoring."""
        scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        for results, weight in zip(result_lists, self.weights):
            for result in results:
                doc_id = result.id or hash(result.content)
                weighted_score = result.score * weight

                if doc_id in scores:
                    scores[doc_id] += weighted_score
                else:
                    scores[doc_id] = weighted_score
                    result_map[doc_id] = result

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused = []
        for doc_id in sorted_ids[:self.top_k]:
            result = result_map[doc_id]
            result.score = scores[doc_id]
            fused.append(result)

        return fused


class RetrieverFactory:
    """Factory for creating retrievers."""

    @staticmethod
    def create(
        retriever_type: str,
        vector_store: Any,
        embedding_generator: Any,
        llm_client: Any = None,
        **kwargs: Any,
    ) -> BaseRetriever:
        """
        Create a retriever.

        Args:
            retriever_type: Type of retriever
            vector_store: Vector store instance
            embedding_generator: Embedding generator
            llm_client: Optional LLM client
            **kwargs: Additional arguments

        Returns:
            Configured retriever
        """
        retrievers = {
            "similarity": SimilarityRetriever,
            "multi_query": MultiQueryRetriever,
            "hyde": HyDERetriever,
            "compression": ContextualCompressionRetriever,
        }

        retriever_class = retrievers.get(retriever_type.lower())
        if not retriever_class:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

        # Special handling for compression retriever
        if retriever_type.lower() == "compression":
            base_retriever = SimilarityRetriever(
                vector_store=vector_store,
                embedding_generator=embedding_generator,
                **kwargs,
            )
            return ContextualCompressionRetriever(
                base_retriever=base_retriever,
                llm_client=llm_client,
                **kwargs,
            )

        # Standard retrievers
        return retriever_class(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            llm_client=llm_client,
            **kwargs,
        )
