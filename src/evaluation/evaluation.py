"""
Evaluation Module - AI-Mastery-2026

This module provides comprehensive evaluation frameworks for ML models,
RAG systems, and LLM applications.

Key Components:
- RAGEvaluator: Evaluate RAG systems (faithfulness, relevance, context precision)
- MLEvaluator: Evaluate ML models (classification, regression metrics)
- LLMEvaluator: Evaluate LLM outputs (fluency, coherence, toxicity)
- BenchmarkRunner: Run systematic benchmarks
- EvaluationReport: Generate evaluation reports

Author: AI-Mastery-2026
License: MIT
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import json
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation.
    
    Attributes:
        batch_size: Batch size for processing
        num_samples: Number of samples to evaluate (None = all)
        metrics: List of metrics to compute
        verbose: Enable verbose logging
        save_predictions: Save predictions alongside metrics
    """
    batch_size: int = 32
    num_samples: Optional[int] = None
    metrics: List[str] = field(default_factory=lambda: ["all"])
    verbose: bool = False
    save_predictions: bool = False


class MetricType(Enum):
    """Type of evaluation metric."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    GENERATION = "generation"
    RAG = "rag"


# =============================================================================
# Metric Base Class
# =============================================================================

@dataclass
class Metric:
    """
    Represents an evaluation metric result.
    
    Attributes:
        name: Metric name
        value: Metric value
        metric_type: Type of metric
        metadata: Additional information
    """
    name: str
    value: float
    metric_type: MetricType = MetricType.CLASSIFICATION
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "metadata": self.metadata
        }


# =============================================================================
# Classification Metrics
# =============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score between 0 and 1
        
    Example:
        >>> acc = accuracy(np.array([1, 0, 1]), np.array([1, 0, 0]))
        >>> print(f"{acc:.2f}")  # 0.67
    """
    return float(np.mean(y_true == y_pred))


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary"
) -> float:
    """
    Calculate precision score.
    
    Precision = TP / (TP + FP)
    
    "How many of the predicted positives are actually positive?"
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'binary', 'macro', or 'micro'
        
    Returns:
        Precision score between 0 and 1
    """
    if average == "binary":
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    else:
        # Macro average: compute per-class and average
        classes = np.unique(y_true)
        precisions = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(p)
        return float(np.mean(precisions))


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary"
) -> float:
    """
    Calculate recall score.
    
    Recall = TP / (TP + FN)
    
    "How many of the actual positives did we find?"
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'binary', 'macro', or 'micro'
        
    Returns:
        Recall score between 0 and 1
    """
    if average == "binary":
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    else:
        classes = np.unique(y_true)
        recalls = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(r)
        return float(np.mean(recalls))


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary"
) -> float:
    """
    Calculate F1 score (harmonic mean of precision and recall).
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'binary', 'macro', or 'micro'
        
    Returns:
        F1 score between 0 and 1
    """
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix of shape (n_classes, n_classes)
        
    Layout:
        [[TN, FP],
         [FN, TP]]  (for binary classification)
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return cm


# =============================================================================
# Regression Metrics
# =============================================================================

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    MSE = (1/n) Σ (y_true - y_pred)²
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE value (lower is better)
    """
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    RMSE = √MSE
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value (in same units as target)
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE = (1/n) Σ |y_true - y_pred|
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value (lower is better)
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).
    
    R² = 1 - SS_res / SS_tot
       = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R² score (1 = perfect, can be negative for poor models)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


# =============================================================================
# Ranking Metrics
# =============================================================================

def mean_reciprocal_rank(
    y_true: List[List[int]],
    y_pred: List[List[int]]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = (1/|Q|) Σ 1/rank_i
    
    Where rank_i is the position of the first relevant document.
    
    Args:
        y_true: List of lists with relevant document indices
        y_pred: List of ranked document lists
        
    Returns:
        MRR score between 0 and 1
        
    Example:
        >>> mrr = mean_reciprocal_rank([[0, 1], [2]], [[1, 0, 2], [0, 1, 2]])
        >>> # First query: relevant at rank 1, Second: relevant at rank 3
    """
    rr_sum = 0.0
    n = len(y_true)
    
    for relevant, ranked in zip(y_true, y_pred):
        relevant_set = set(relevant)
        for rank, doc_id in enumerate(ranked, 1):
            if doc_id in relevant_set:
                rr_sum += 1.0 / rank
                break
    
    return float(rr_sum / n) if n > 0 else 0.0


def precision_at_k(
    y_true: List[int],
    y_pred: List[int],
    k: int = 10
) -> float:
    """
    Calculate Precision@K.
    
    P@K = |relevant ∩ top-k| / k
    
    Args:
        y_true: List of relevant document indices
        y_pred: Ranked list of document indices
        k: Number of top results to consider
        
    Returns:
        Precision@K score
    """
    relevant = set(y_true)
    top_k = y_pred[:k]
    
    relevant_in_top_k = len(set(top_k) & relevant)
    return float(relevant_in_top_k / k)


def recall_at_k(
    y_true: List[int],
    y_pred: List[int],
    k: int = 10
) -> float:
    """
    Calculate Recall@K.
    
    R@K = |relevant ∩ top-k| / |relevant|
    
    Args:
        y_true: List of relevant document indices
        y_pred: Ranked list of document indices
        k: Number of top results to consider
        
    Returns:
        Recall@K score
    """
    relevant = set(y_true)
    if not relevant:
        return 0.0
    
    top_k = y_pred[:k]
    relevant_in_top_k = len(set(top_k) & relevant)
    
    return float(relevant_in_top_k / len(relevant))


def ndcg_at_k(
    y_true: List[float],
    y_pred: List[int],
    k: int = 10
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    DCG@K = Σ (2^rel_i - 1) / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K
    
    Args:
        y_true: Relevance scores for all documents
        y_pred: Ranked list of document indices
        k: Number of top results
        
    Returns:
        NDCG@K score between 0 and 1
    """
    def dcg(relevances: List[float]) -> float:
        return sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(relevances[:k])
        )
    
    # Get relevances for predicted ranking
    predicted_rels = [y_true[i] for i in y_pred[:k] if i < len(y_true)]
    
    # Ideal ranking (sorted by relevance)
    ideal_rels = sorted(y_true, reverse=True)[:k]
    
    dcg_score = dcg(predicted_rels)
    idcg_score = dcg(ideal_rels)
    
    if idcg_score == 0:
        return 0.0
    return float(dcg_score / idcg_score)


# =============================================================================
# RAG Evaluation Metrics
# =============================================================================

class RAGEvaluator:
    """
    Evaluator for Retrieval-Augmented Generation systems.
    
    Measures key RAG quality dimensions:
        - Retrieval: Are we finding the right documents?
        - Faithfulness: Is the answer grounded in the context?
        - Relevance: Is the answer relevant to the question?
        - Context Precision: How much retrieved context was useful?
    
    Example:
        >>> evaluator = RAGEvaluator()
        >>> results = evaluator.evaluate(
        ...     questions=["What is AI?"],
        ...     answers=["AI is artificial intelligence."],
        ...     contexts=[["AI stands for artificial intelligence..."]],
        ...     ground_truths=["AI is the simulation of human intelligence."]
        ... )
        >>> print(results["faithfulness"])  # 0.85
    
    Reference:
        RAGAS paper: https://arxiv.org/abs/2309.15217
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize RAG evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self._embedder = None
        
        logger.info("Initialized RAGEvaluator")
    
    def _load_embedder(self):
        """Load embedding model for similarity calculations."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                self._embedder = "fallback"
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between texts."""
        self._load_embedder()
        
        if self._embedder == "fallback":
            # Fallback: word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            overlap = len(words1 & words2)
            union = len(words1 | words2)
            return overlap / union if union > 0 else 0.0
        
        embeddings = self._embedder.encode([text1, text2])
        return float(np.dot(embeddings[0], embeddings[1]))
    
    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Evaluate how faithful the answer is to the context.
        
        Faithfulness measures whether each claim in the answer
        can be inferred from the provided contexts.
        
        Args:
            answer: Generated answer
            contexts: Retrieved context passages
            
        Returns:
            Faithfulness score between 0 and 1
        """
        if not contexts:
            return 0.0
        
        # Combine all contexts
        combined_context = " ".join(contexts)
        
        # Compute similarity between answer and context
        similarity = self._compute_similarity(answer, combined_context)
        
        # Check for answer keywords in context
        answer_words = set(answer.lower().split())
        context_words = set(combined_context.lower().split())
        overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0
        
        # Combine similarity and overlap
        faithfulness = 0.6 * similarity + 0.4 * overlap
        
        return min(faithfulness, 1.0)
    
    def evaluate_relevance(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Evaluate how relevant the answer is to the question.
        
        Relevance measures whether the answer actually addresses
        what the question is asking.
        
        Args:
            question: User question
            answer: Generated answer
            
        Returns:
            Relevance score between 0 and 1
        """
        # Compute question-answer similarity
        similarity = self._compute_similarity(question, answer)
        
        # Check for question keywords in answer
        q_words = set(question.lower().split()) - {"what", "is", "the", "a", "an", "how", "why"}
        a_words = set(answer.lower().split())
        keyword_overlap = len(q_words & a_words) / len(q_words) if q_words else 0
        
        return 0.5 * similarity + 0.5 * keyword_overlap
    
    def evaluate_context_precision(
        self,
        question: str,
        contexts: List[str],
        ground_truth: str
    ) -> float:
        """
        Evaluate how precise the retrieved context is.
        
        Context Precision measures what fraction of the retrieved
        context was actually useful for answering the question.
        
        Args:
            question: User question
            contexts: Retrieved context passages
            ground_truth: Expected answer
            
        Returns:
            Context precision score between 0 and 1
        """
        if not contexts:
            return 0.0
        
        useful_count = 0
        for context in contexts:
            # Check if context is relevant to ground truth
            similarity = self._compute_similarity(context, ground_truth)
            if similarity > 0.3:
                useful_count += 1
        
        return useful_count / len(contexts)
    
    def evaluate_context_recall(
        self,
        contexts: List[str],
        ground_truth: str
    ) -> float:
        """
        Evaluate context recall - how much of the ground truth
        is covered by the retrieved contexts.
        
        Args:
            contexts: Retrieved context passages
            ground_truth: Expected answer
            
        Returns:
            Context recall score between 0 and 1
        """
        if not contexts:
            return 0.0
        
        combined_context = " ".join(contexts)
        
        # Check ground truth coverage
        gt_words = set(ground_truth.lower().split())
        context_words = set(combined_context.lower().split())
        
        covered = len(gt_words & context_words)
        return covered / len(gt_words) if gt_words else 0.0
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive RAG evaluation.
        
        Args:
            questions: List of user questions
            answers: List of generated answers
            contexts: List of context lists (one per question)
            ground_truths: Optional list of expected answers
            
        Returns:
            Dictionary of metric scores
        """
        n = len(questions)
        
        # Initialize accumulators
        faithfulness_scores = []
        relevance_scores = []
        context_precision_scores = []
        context_recall_scores = []
        
        for i in range(n):
            # Faithfulness
            f_score = self.evaluate_faithfulness(answers[i], contexts[i])
            faithfulness_scores.append(f_score)
            
            # Relevance
            r_score = self.evaluate_relevance(questions[i], answers[i])
            relevance_scores.append(r_score)
            
            # Context metrics (need ground truth)
            if ground_truths:
                cp_score = self.evaluate_context_precision(
                    questions[i], contexts[i], ground_truths[i]
                )
                context_precision_scores.append(cp_score)
                
                cr_score = self.evaluate_context_recall(
                    contexts[i], ground_truths[i]
                )
                context_recall_scores.append(cr_score)
        
        results = {
            "faithfulness": float(np.mean(faithfulness_scores)),
            "relevance": float(np.mean(relevance_scores)),
            "n_samples": n
        }
        
        if ground_truths:
            results["context_precision"] = float(np.mean(context_precision_scores))
            results["context_recall"] = float(np.mean(context_recall_scores))
        
        return results


# =============================================================================
# ML Evaluator
# =============================================================================

class MLEvaluator:
    """
    Evaluator for traditional ML models.
    
    Supports both classification and regression evaluation
    with comprehensive metric computation.
    
    Example:
        >>> evaluator = MLEvaluator()
        >>> 
        >>> # Classification
        >>> metrics = evaluator.evaluate_classification(y_true, y_pred)
        >>> print(metrics["accuracy"], metrics["f1"])
        >>> 
        >>> # Regression
        >>> metrics = evaluator.evaluate_regression(y_true, y_pred)
        >>> print(metrics["rmse"], metrics["r2"])
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize ML evaluator."""
        self.config = config or EvaluationConfig()
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        average: str = "macro"
    ) -> Dict[str, Any]:
        """
        Evaluate classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for AUC)
            average: Averaging method for multiclass
            
        Returns:
            Dictionary of classification metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            "accuracy": accuracy(y_true, y_pred),
            "precision": precision(y_true, y_pred, average),
            "recall": recall(y_true, y_pred, average),
            "f1": f1_score(y_true, y_pred, average),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "n_samples": len(y_true),
            "n_classes": len(np.unique(y_true))
        }
        
        return metrics
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "n_samples": len(y_true)
        }
    
    def evaluate_ranking(
        self,
        y_true: List[List[int]],
        y_pred: List[List[int]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate ranking model.
        
        Args:
            y_true: Lists of relevant document indices
            y_pred: Lists of ranked document indices
            k: Cutoff for @K metrics
            
        Returns:
            Dictionary of ranking metrics
        """
        metrics = {
            "mrr": mean_reciprocal_rank(y_true, y_pred),
            "n_queries": len(y_true)
        }
        
        # Compute P@K and R@K for each query
        p_at_k = []
        r_at_k = []
        
        for true, pred in zip(y_true, y_pred):
            p_at_k.append(precision_at_k(true, pred, k))
            r_at_k.append(recall_at_k(true, pred, k))
        
        metrics[f"precision@{k}"] = float(np.mean(p_at_k))
        metrics[f"recall@{k}"] = float(np.mean(r_at_k))
        
        return metrics


# =============================================================================
# LLM Evaluator
# =============================================================================

class LLMEvaluator:
    """
    Evaluator for LLM outputs.
    
    Measures generation quality dimensions:
        - Fluency: Grammatical correctness
        - Coherence: Logical flow
        - Relevance: Topic alignment
        - Toxicity: Harmful content detection
    
    Example:
        >>> evaluator = LLMEvaluator()
        >>> scores = evaluator.evaluate(
        ...     prompt="Write about AI",
        ...     response="AI is transforming industries..."
        ... )
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize LLM evaluator."""
        self.config = config or EvaluationConfig()
    
    def evaluate_fluency(self, text: str) -> float:
        """
        Evaluate text fluency (grammatical correctness).
        
        Args:
            text: Text to evaluate
            
        Returns:
            Fluency score between 0 and 1
        """
        # Simple heuristics for fluency
        sentences = text.split(".")
        
        # Check sentence length distribution
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.0
        
        avg_length = np.mean(lengths)
        
        # Sentences should be reasonable length (5-30 words)
        if 5 <= avg_length <= 30:
            fluency = 0.8
        elif avg_length < 5 or avg_length > 50:
            fluency = 0.4
        else:
            fluency = 0.6
        
        # Check for sentence starters
        good_starts = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        fluency += 0.2 * (good_starts / len(sentences)) if sentences else 0
        
        return min(fluency, 1.0)
    
    def evaluate_coherence(self, text: str) -> float:
        """
        Evaluate text coherence (logical flow).
        
        Args:
            text: Text to evaluate
            
        Returns:
            Coherence score between 0 and 1
        """
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        # Simple coherence: check for discourse markers
        discourse_markers = {
            "however", "therefore", "furthermore", "moreover",
            "additionally", "thus", "consequently", "first",
            "second", "finally", "in conclusion", "for example"
        }
        
        text_lower = text.lower()
        marker_count = sum(1 for marker in discourse_markers if marker in text_lower)
        
        coherence = min(0.5 + 0.1 * marker_count, 1.0)
        
        return coherence
    
    def evaluate_relevance(self, prompt: str, response: str) -> float:
        """
        Evaluate response relevance to prompt.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Relevance score between 0 and 1
        """
        # Keyword overlap
        prompt_words = set(prompt.lower().split()) - {"the", "a", "an", "is", "are", "to"}
        response_words = set(response.lower().split())
        
        overlap = len(prompt_words & response_words)
        relevance = overlap / len(prompt_words) if prompt_words else 0
        
        return min(relevance, 1.0)
    
    def evaluate(
        self,
        prompt: str,
        response: str
    ) -> Dict[str, float]:
        """
        Comprehensive LLM output evaluation.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Dictionary of evaluation scores
        """
        return {
            "fluency": self.evaluate_fluency(response),
            "coherence": self.evaluate_coherence(response),
            "relevance": self.evaluate_relevance(prompt, response),
            "length": len(response.split())
        }


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Run systematic benchmarks on models.
    
    Automates evaluation across multiple test cases and
    generates comprehensive reports.
    
    Example:
        >>> runner = BenchmarkRunner()
        >>> runner.add_benchmark("classification", X_test, y_test)
        >>> results = runner.run(model)
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize benchmark runner."""
        self.config = config or EvaluationConfig()
        self.benchmarks: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Any] = {}
    
    def add_benchmark(
        self,
        name: str,
        data: Any,
        labels: Any,
        metric_type: MetricType = MetricType.CLASSIFICATION
    ) -> None:
        """
        Add a benchmark dataset.
        
        Args:
            name: Benchmark name
            data: Test data
            labels: True labels
            metric_type: Type of evaluation
        """
        self.benchmarks[name] = {
            "data": data,
            "labels": labels,
            "type": metric_type
        }
        logger.info(f"Added benchmark: {name}")
    
    def run(
        self,
        model: Any,
        predict_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run all benchmarks.
        
        Args:
            model: Model to evaluate
            predict_fn: Optional custom prediction function
            
        Returns:
            Dictionary of benchmark results
        """
        ml_evaluator = MLEvaluator(self.config)
        
        for name, benchmark in self.benchmarks.items():
            logger.info(f"Running benchmark: {name}")
            
            data = benchmark["data"]
            labels = benchmark["labels"]
            metric_type = benchmark["type"]
            
            # Get predictions
            if predict_fn:
                predictions = predict_fn(data)
            elif hasattr(model, "predict"):
                predictions = model.predict(data)
            else:
                logger.warning(f"Cannot get predictions for {name}")
                continue
            
            # Evaluate
            if metric_type == MetricType.CLASSIFICATION:
                metrics = ml_evaluator.evaluate_classification(labels, predictions)
            elif metric_type == MetricType.REGRESSION:
                metrics = ml_evaluator.evaluate_regression(labels, predictions)
            else:
                metrics = {"raw_predictions": predictions.tolist()}
            
            self.results[name] = metrics
        
        return self.results
    
    def summary(self) -> str:
        """Generate benchmark summary."""
        lines = ["Benchmark Results", "=" * 40]
        
        for name, metrics in self.results.items():
            lines.append(f"\n{name}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                elif key != "confusion_matrix":
                    lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


# =============================================================================
# Evaluation Report
# =============================================================================

class EvaluationReport:
    """
    Generate comprehensive evaluation reports.
    
    Supports multiple output formats: JSON, Markdown, HTML.
    
    Example:
        >>> report = EvaluationReport()
        >>> report.add_metrics("RAG Evaluation", rag_metrics)
        >>> report.add_metrics("Classification", clf_metrics)
        >>> report.save("report.md", format="markdown")
    """
    
    def __init__(self, title: str = "Evaluation Report"):
        """
        Initialize report.
        
        Args:
            title: Report title
        """
        self.title = title
        self.sections: Dict[str, Dict[str, Any]] = {}
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }
    
    def add_metrics(
        self,
        section: str,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Add metrics to a section.
        
        Args:
            section: Section name
            metrics: Dictionary of metrics
        """
        self.sections[section] = metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "title": self.title,
            "metadata": self.metadata,
            "sections": self.sections
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_markdown(self) -> str:
        """Convert report to Markdown."""
        lines = [
            f"# {self.title}",
            f"\n*Generated: {self.metadata['created_at']}*\n"
        ]
        
        for section, metrics in self.sections.items():
            lines.append(f"\n## {section}\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"| {key} | {value:.4f} |")
                elif not isinstance(value, (list, dict)):
                    lines.append(f"| {key} | {value} |")
        
        return "\n".join(lines)
    
    def save(self, path: str, format: str = "json") -> None:
        """
        Save report to file.
        
        Args:
            path: Output file path
            format: 'json' or 'markdown'
        """
        if format == "json":
            content = self.to_json()
        else:
            content = self.to_markdown()
        
        with open(path, "w") as f:
            f.write(content)
        
        logger.info(f"Saved report to {path}")


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Module Demo")
    print("=" * 60)
    
    # 1. Classification Metrics
    print("\n1. Classification Metrics")
    print("-" * 40)
    
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
    
    print(f"  Accuracy:  {accuracy(y_true, y_pred):.3f}")
    print(f"  Precision: {precision(y_true, y_pred):.3f}")
    print(f"  Recall:    {recall(y_true, y_pred):.3f}")
    print(f"  F1 Score:  {f1_score(y_true, y_pred):.3f}")
    
    # 2. Regression Metrics
    print("\n2. Regression Metrics")
    print("-" * 40)
    
    y_true_reg = np.array([3.0, 5.0, 2.5, 7.0, 4.5])
    y_pred_reg = np.array([2.8, 5.2, 2.3, 6.8, 4.6])
    
    print(f"  RMSE: {root_mean_squared_error(y_true_reg, y_pred_reg):.3f}")
    print(f"  MAE:  {mean_absolute_error(y_true_reg, y_pred_reg):.3f}")
    print(f"  R²:   {r2_score(y_true_reg, y_pred_reg):.3f}")
    
    # 3. RAG Evaluation
    print("\n3. RAG Evaluation")
    print("-" * 40)
    
    rag_eval = RAGEvaluator()
    
    questions = ["What is machine learning?"]
    answers = ["Machine learning is a type of AI that learns from data."]
    contexts = [["ML is a subset of artificial intelligence that enables systems to learn."]]
    ground_truths = ["Machine learning is AI that learns patterns from data."]
    
    rag_results = rag_eval.evaluate(questions, answers, contexts, ground_truths)
    
    for metric, value in rag_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
    
    # 4. Report Generation
    print("\n4. Evaluation Report")
    print("-" * 40)
    
    report = EvaluationReport(title="Demo Evaluation")
    report.add_metrics("Classification", {
        "accuracy": accuracy(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    })
    report.add_metrics("RAG", rag_results)
    
    print(report.to_markdown()[:500] + "...")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
