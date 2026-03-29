"""
RAG Reranking Module
====================

Reranking strategies for improving retrieval results.

Provides:
- Cross-encoder reranking
- LLM-based reranking
- Diversity-based reranking
- Reciprocal rank fusion

Quick Start:
    >>> from src.rag.reranking import CrossEncoderReranker
    >>> from src.rag.retrieval import RetrievalResults
    >>>
    >>> reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    >>> reranked = reranker.rerank(query, results)
"""

from .base import BaseReranker, RerankResult, RerankResults
from .cross_encoder import CrossEncoderReranker
from .llm_reranker import LLMReranker
from .diversity import DiversityReranker

__all__ = [
    # Base classes
    "BaseReranker",
    "RerankResult",
    "RerankResults",
    
    # Reranking strategies
    "CrossEncoderReranker",
    "LLMReranker",
    "DiversityReranker",
]
