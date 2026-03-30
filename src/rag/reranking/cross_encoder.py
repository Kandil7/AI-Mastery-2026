"""
Cross-Encoder Reranker
======================

Uses cross-encoder models for accurate reranking.

Cross-encoders process query and document together,
providing more accurate relevance scores than
dot-product similarity.

Models:
- cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
- cross-encoder/ms-marco-electra-base (better quality)
- cross-encoder/stsb-roberta-base (semantic similarity)

Installation:
    pip install sentence-transformers

Example:
    >>> reranker = CrossEncoderReranker(
    ...     "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ... )
    >>> results = reranker.rerank(query, documents)
"""

import time
from typing import Any, Dict, List, Optional

from .base import BaseReranker, RerankResult, RerankResults


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder based reranker.

    Uses sentence-transformers CrossEncoder for accurate
    query-document relevance scoring.

    Example:
        >>> reranker = CrossEncoderReranker(
        ...     model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ...     top_k=5,
        ... )
        >>>
        >>> documents = [
        ...     {"id": "1", "content": "Python is a programming language"},
        ...     {"id": "2", "content": "Snakes are reptiles"},
        ... ]
        >>>
        >>> results = reranker.rerank("python programming", documents)
        >>> print(results[0].id)  # Should be "1"
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        max_length: int = 512,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name
            top_k: Number of results to return
            max_length: Maximum sequence length
        """
        super().__init__(top_k)
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    def _load_model(self):
        """Lazily load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> RerankResults:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Query text
            results: List of documents to rerank
            top_k: Number of results to return

        Returns:
            RerankResults with new rankings
        """
        top_k = top_k or self.top_k
        start_time = time.time()

        # Load model if needed
        self._load_model()

        if not results:
            return RerankResults(results=[], original_count=0)

        # Prepare pairs for cross-encoder
        pairs = [[query, doc["content"]] for doc in results]

        # Get scores
        scores = self._model.predict(pairs)

        # Build reranked results
        reranked = []
        for i, (doc, score) in enumerate(zip(results, scores)):
            reranked.append(
                RerankResult(
                    id=doc["id"],
                    content=doc["content"],
                    original_score=doc.get("score", 0.0),
                    rerank_score=float(score),
                    metadata=doc.get("metadata", {}),
                )
            )

        # Sort by rerank score
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)

        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1

        # Take top_k
        top_results = reranked[:top_k]

        rerank_time = (time.time() - start_time) * 1000

        return RerankResults(
            results=top_results,
            rerank_time_ms=rerank_time,
            original_count=len(results),
        )

    def score(self, query: str, document: str) -> float:
        """
        Get relevance score for a single query-document pair.

        Args:
            query: Query text
            document: Document text

        Returns:
            Relevance score (0-1)
        """
        self._load_model()
        score = self._model.predict([[query, document]])[0]
        return float(score)

    def score_batch(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        Get relevance scores for multiple documents.

        Args:
            query: Query text
            documents: List of document texts

        Returns:
            List of relevance scores
        """
        self._load_model()
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self.model_name})"
