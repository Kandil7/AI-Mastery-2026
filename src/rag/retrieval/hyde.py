"""
HyDE Retrieval
==============

Hypothetical Document Embeddings for improved retrieval.

Reference: Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels"
"""

from typing import Any, Dict, List, Optional, Callable

from .base import BaseRetriever, RetrievalResult, RetrievalResults
from src.rag.vector_stores.base import VectorStore


class HyDERetriever(BaseRetriever):
    """
    HyDE (Hypothetical Document Embeddings) retriever.

    Instead of embedding the query directly, generates a hypothetical
    document that would answer the query, then uses its embedding
    for retrieval.

    Benefits:
    - Better handles query-document domain gap
    - Improves retrieval for complex queries
    - Works well with instructional queries

    Example:
        >>> def generate_hypothetical(query):
        ...     # Use LLM to generate hypothetical answer
        ...     return f"The answer to {query} is..."
        >>>
        >>> retriever = HyDERetriever(
        ...     vector_store,
        ...     hypothetical_generator=generate_hypothetical,
        ...     embedder=embedder,
        ... )
        >>> results = retriever.retrieve("What is quantum computing?")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        hypothetical_generator: Callable[[str], str],
        embedder: Callable[[str], List[float]],
        top_k: int = 5,
    ):
        """
        Initialize HyDE retriever.

        Args:
            vector_store: Vector store to search
            hypothetical_generator: Function to generate hypothetical document
            embedder: Function to embed text to vectors
            top_k: Default number of results
        """
        super().__init__(top_k)
        self.vector_store = vector_store
        self.hypothetical_generator = hypothetical_generator
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        """
        Retrieve using HyDE approach.

        Args:
            query: Query text
            top_k: Number of results
            **kwargs: Additional parameters

        Returns:
            RetrievalResults object
        """
        top_k = top_k or self.top_k

        # Generate hypothetical document
        hypothetical_doc = self.hypothetical_generator(query)

        # Embed hypothetical document
        query_vector = self.embedder(hypothetical_doc)

        # Search with hypothetical embedding
        search_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        # Convert to RetrievalResults
        results = [
            RetrievalResult(
                id=result.id,
                content=result.metadata.get("content", ""),
                score=result.score,
                metadata=result.metadata,
            )
            for result in search_results.results
        ]

        return RetrievalResults(
            results=results,
            query_time_ms=search_results.query_time_ms,
            total_count=len(results),
        )

    def retrieve_with_fallback(
        self,
        query: str,
        top_k: Optional[int] = None,
        fallback_threshold: float = 0.3,
        **kwargs: Any,
    ) -> RetrievalResults:
        """
        Retrieve with fallback to direct query embedding.

        If HyDE results have low scores, falls back to
        embedding the query directly.

        Args:
            query: Query text
            top_k: Number of results
            fallback_threshold: Score threshold for fallback
            **kwargs: Additional parameters

        Returns:
            RetrievalResults object
        """
        # Try HyDE first
        results = self.retrieve(query, top_k, **kwargs)

        # Check if fallback needed
        if results.results and results.results[0].score < fallback_threshold:
            # Fall back to direct query embedding
            query_vector = self.embedder(query)
            search_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
            )

            results = RetrievalResults(
                results=[
                    RetrievalResult(
                        id=result.id,
                        content=result.metadata.get("content", ""),
                        score=result.score,
                        metadata=result.metadata,
                    )
                    for result in search_results.results
                ],
                query_time_ms=search_results.query_time_ms,
                total_count=len(search_results.results),
            )

        return results

    def __repr__(self) -> str:
        return f"HyDERetriever(store={self.vector_store})"

