"""
Evaluation Module

Handles RAG evaluation:
- Retrieval metrics (Precision, Recall, MRR, NDCG)
- Generation metrics (Faithfulness, Relevance)
- Islamic-specific metrics
"""

from .evaluator import (
    RAGEvaluator,
    ArabicTestDataset,
    EvaluationSample,
    RetrievalMetrics,
    GenerationMetrics,
)

from .islamic_metrics import (
    IslamicRAGEvaluator,
    IslamicEvaluationMetrics,
    create_islamic_evaluator,
    AUTHENTICITY_DB,
)

__all__ = [
    # Evaluator
    "RAGEvaluator",
    "ArabicTestDataset",
    "EvaluationSample",
    "RetrievalMetrics",
    "GenerationMetrics",
    # Islamic Metrics
    "IslamicRAGEvaluator",
    "IslamicEvaluationMetrics",
    "create_islamic_evaluator",
    "AUTHENTICITY_DB",
]
