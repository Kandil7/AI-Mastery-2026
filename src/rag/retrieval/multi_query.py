"""
Multi-Query Retrieval
=====================

Generates multiple query variations for better coverage.
"""

from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict

from .base import BaseRetriever, RetrievalResult, RetrievalResults
from src.rag.vector_stores.base import VectorStore


class MultiQueryRetriever(BaseRetriever):
    """
    Multi-query retrieval for improved coverage.

    Generates multiple variations of the input query,
    retrieves for each, and combines results.

    Benefits:
    - Reduces impact of poor query formulation
    - Increases recall with diverse query angles
    - Handles ambiguous queries better

    Example:
        >>> def generate_variations(query):
        ...     return [
        ...         query,
        ...         f"What is {query}?",
        ...         f"Explain {query}",
        ...     ]
        >>>
        >>> retriever = MultiQueryRetriever(
        ...     vector_store,
        ...     query_generator=generate_variations,
        ... )
        >>> results = retriever.retrieve("machine learning")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        query_generator: Callable[[str], List[str]],
        top_k: int = 5,
        merge_strategy: str = "rrf",
    ):
        """
        Initialize multi-query retriever.

        Args:
            vector_store: Vector store to search
            query_generator: Function to generate query variations
            top_k: Number of results per query
            merge_strategy: How to merge results ("rrf", "average", "union")
        """
        super().__init__(top_k)
        self.vector_store = vector_store
        self.query_generator = query_generator
        self.merge_strategy = merge_strategy

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        """
        Retrieve using multiple query variations.

        Args:
            query: Original query
            top_k: Total number of results to return
            **kwargs: Additional parameters

        Returns:
            Combined RetrievalResults
        """
        top_k = top_k or self.top_k

        # Generate query variations
        queries = self.query_generator(query)

        # Retrieve for each query
        all_results: Dict[str, List[Any]] = defaultdict(list)
        result_map: Dict[str, Any] = {}

        for q in queries:
            results = self.vector_store.search(
                query_vector=q if isinstance(q, list) else [],
                top_k=top_k,
            )

            for result in results.results:
                all_results[result.id].append(result.score)
                if result.id not in result_map:
                    result_map[result.id] = result

        # Merge scores based on strategy
        if self.merge_strategy == "rrf":
            final_scores = self._reciprocal_rank_merge(all_results)
        elif self.merge_strategy == "average":
            final_scores = self._average_merge(all_results)
        else:  # union
            final_scores = self._union_merge(all_results)

        # Sort by final score
        sorted_ids = sorted(
            final_scores.keys(),
            key=lambda x: final_scores[x],
            reverse=True,
        )[:top_k]

        # Build results
        results = [
            RetrievalResult(
                id=id_,
                content=result_map[id_].metadata.get("content", ""),
                score=final_scores[id_],
                metadata=result_map[id_].metadata,
            )
            for id_ in sorted_ids
        ]

        return RetrievalResults(
            results=results,
            query_time_ms=0.0,  # Would need to track across queries
            total_count=len(results),
        )

    def _reciprocal_rank_merge(
        self,
        all_results: Dict[str, List[float]],
        k: int = 60,
    ) -> Dict[str, float]:
        """Merge using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}

        for id_, score_list in all_results.items():
            # Use best rank (lowest position)
            best_rank = min(range(len(score_list)), key=lambda i: -score_list[i]) + 1
            scores[id_] = 1.0 / (k + best_rank)

        return scores

    def _average_merge(self, all_results: Dict[str, List[float]]) -> Dict[str, float]:
        """Merge by averaging scores."""
        return {
            id_: sum(scores) / len(scores)
            for id_, scores in all_results.items()
        }

    def _union_merge(self, all_results: Dict[str, List[float]]) -> Dict[str, float]:
        """Merge by taking maximum score."""
        return {
            id_: max(scores)
            for id_, scores in all_results.items()
        }

    def __repr__(self) -> str:
        return f"MultiQueryRetriever(store={self.vector_store}, strategy={self.merge_strategy})"

