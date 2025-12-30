"""
Legal Document RAG - Evaluation Metrics
=======================================
Metrics for evaluating RAG system performance.

Metrics Implemented:
- Retrieval: Precision, Recall, MRR, NDCG
- Generation: BLEU, ROUGE, Citation accuracy
- End-to-end: Answer correctness, Faithfulness

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import re


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@K: Fraction of top-K retrieved docs that are relevant.
    
    P@K = |{relevant docs in top K}| / K
    
    Args:
        retrieved: List of retrieved document IDs (ordered by rank)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
    
    Returns:
        Precision score in [0, 1]
    """
    if k == 0:
        return 0.0
    
    top_k = set(retrieved[:k])
    relevant_in_k = len(top_k & relevant)
    
    return relevant_in_k / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@K: Fraction of relevant docs that are in top-K.
    
    R@K = |{relevant docs in top K}| / |{all relevant docs}|
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs
        k: Number of top results to consider
    
    Returns:
        Recall score in [0, 1]
    """
    if len(relevant) == 0:
        return 0.0
    
    top_k = set(retrieved[:k])
    relevant_in_k = len(top_k & relevant)
    
    return relevant_in_k / len(relevant)


def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Mean Reciprocal Rank (MRR): 1/rank of first relevant doc.
    
    MRR = 1 / rank_of_first_relevant
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs
    
    Returns:
        MRR score in [0, 1]
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    
    return 0.0


def ndcg_at_k(
    retrieved: List[str], 
    relevance_scores: Dict[str, float], 
    k: int
) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K).
    
    NDCG = DCG / IDCG
    DCG = Σᵢ (2^rel(i) - 1) / log₂(i + 1)
    
    Args:
        retrieved: List of retrieved document IDs
        relevance_scores: Dict mapping doc_id to relevance score
        k: Number of top results to consider
    
    Returns:
        NDCG score in [0, 1]
    """
    def dcg(scores: List[float]) -> float:
        return sum(
            (2**score - 1) / np.log2(i + 2)
            for i, score in enumerate(scores)
        )
    
    # Get relevance scores for retrieved docs
    retrieved_scores = [
        relevance_scores.get(doc_id, 0.0) 
        for doc_id in retrieved[:k]
    ]
    
    # Calculate DCG
    actual_dcg = dcg(retrieved_scores)
    
    # Calculate ideal DCG (perfect ranking)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_scores)
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Average Precision (AP): Average of P@K at each relevant position.
    
    AP = (1/|R|) × Σₖ P@k × rel(k)
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs
    
    Returns:
        AP score in [0, 1]
    """
    if len(relevant) == 0:
        return 0.0
    
    precisions = []
    relevant_count = 0
    
    for k, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            relevant_count += 1
            precisions.append(relevant_count / k)
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / len(relevant)


# ============================================================
# GENERATION METRICS
# ============================================================

def _tokenize(text: str) -> List[str]:
    """Simple tokenization for metric computation."""
    return re.findall(r'\w+', text.lower())


def bleu_score(
    reference: str, 
    candidate: str, 
    max_n: int = 4,
    weights: Optional[List[float]] = None
) -> float:
    """
    BLEU Score (Bilingual Evaluation Understudy).
    
    Measures n-gram overlap between candidate and reference.
    
    Args:
        reference: Reference (ground truth) text
        candidate: Candidate (generated) text
        max_n: Maximum n-gram size
        weights: Weights for each n-gram (default: uniform)
    
    Returns:
        BLEU score in [0, 1]
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # Calculate n-gram precisions
    precisions = []
    
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i+n]) 
            for i in range(len(ref_tokens) - n + 1)
        )
        cand_ngrams = Counter(
            tuple(cand_tokens[i:i+n]) 
            for i in range(len(cand_tokens) - n + 1)
        )
        
        # Clipped count
        overlap = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        
        if total > 0:
            precisions.append(overlap / total)
        else:
            precisions.append(0.0)
    
    # Brevity penalty
    if len(cand_tokens) < len(ref_tokens):
        bp = np.exp(1 - len(ref_tokens) / len(cand_tokens))
    else:
        bp = 1.0
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_prec = sum(w * np.log(p) for w, p in zip(weights, precisions))
    
    return bp * np.exp(log_prec)


def rouge_l(reference: str, candidate: str) -> Dict[str, float]:
    """
    ROUGE-L Score based on Longest Common Subsequence.
    
    Args:
        reference: Reference text
        candidate: Candidate text
    
    Returns:
        Dict with precision, recall, and F1 scores
    """
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # LCS dynamic programming
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    precision = lcs_length / n
    recall = lcs_length / m
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def citation_accuracy(
    generated_citations: List[str],
    ground_truth_citations: List[str]
) -> Dict[str, float]:
    """
    Citation accuracy metrics.
    
    Args:
        generated_citations: Citations in generated answer
        ground_truth_citations: Expected citations
    
    Returns:
        Dict with precision, recall, F1
    """
    gen_set = set(generated_citations)
    truth_set = set(ground_truth_citations)
    
    if len(gen_set) == 0 and len(truth_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if len(gen_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if len(truth_set) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    
    correct = len(gen_set & truth_set)
    
    precision = correct / len(gen_set)
    recall = correct / len(truth_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}


# ============================================================
# END-TO-END METRICS
# ============================================================

@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    end_to_end_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval": self.retrieval_metrics,
            "generation": self.generation_metrics,
            "end_to_end": self.end_to_end_metrics
        }
    
    def summary(self) -> str:
        """Get formatted summary."""
        lines = ["=== Evaluation Results ==="]
        
        lines.append("\nRetrieval:")
        for k, v in self.retrieval_metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        
        lines.append("\nGeneration:")
        for k, v in self.generation_metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        
        lines.append("\nEnd-to-End:")
        for k, v in self.end_to_end_metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        
        return "\n".join(lines)


class RAGEvaluator:
    """
    Complete evaluation suite for RAG systems.
    
    Example:
        >>> evaluator = RAGEvaluator()
        >>> result = evaluator.evaluate(
        ...     query="What are the termination clauses?",
        ...     generated_answer="The contract can be terminated...",
        ...     ground_truth_answer="According to Section 5...",
        ...     retrieved_docs=["doc1", "doc2"],
        ...     relevant_docs={"doc1", "doc3"},
        ...     generated_citations=["42 U.S.C. § 1983"],
        ...     ground_truth_citations=["42 U.S.C. § 1983", "28 U.S.C. § 1331"]
        ... )
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Args:
            k_values: K values for P@K and R@K metrics
        """
        self.k_values = k_values or [1, 3, 5, 10]
    
    def evaluate_retrieval(
        self,
        retrieved: List[str],
        relevant: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        metrics = {}
        
        for k in self.k_values:
            metrics[f"P@{k}"] = precision_at_k(retrieved, relevant, k)
            metrics[f"R@{k}"] = recall_at_k(retrieved, relevant, k)
        
        metrics["MRR"] = mean_reciprocal_rank(retrieved, relevant)
        metrics["AP"] = average_precision(retrieved, relevant)
        
        if relevance_scores:
            for k in self.k_values:
                metrics[f"NDCG@{k}"] = ndcg_at_k(retrieved, relevance_scores, k)
        
        return metrics
    
    def evaluate_generation(
        self,
        generated: str,
        reference: str,
        generated_citations: Optional[List[str]] = None,
        reference_citations: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate generation quality."""
        metrics = {}
        
        # Text quality
        metrics["BLEU"] = bleu_score(reference, generated)
        rouge = rouge_l(reference, generated)
        metrics["ROUGE-L-P"] = rouge["precision"]
        metrics["ROUGE-L-R"] = rouge["recall"]
        metrics["ROUGE-L-F1"] = rouge["f1"]
        
        # Citation quality
        if generated_citations is not None and reference_citations is not None:
            citation = citation_accuracy(generated_citations, reference_citations)
            metrics["Citation-P"] = citation["precision"]
            metrics["Citation-R"] = citation["recall"]
            metrics["Citation-F1"] = citation["f1"]
        
        return metrics
    
    def evaluate(
        self,
        query: str,
        generated_answer: str,
        ground_truth_answer: str,
        retrieved_docs: List[str],
        relevant_docs: Set[str],
        generated_citations: Optional[List[str]] = None,
        ground_truth_citations: Optional[List[str]] = None,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> EvaluationResult:
        """Run complete evaluation suite."""
        
        retrieval_metrics = self.evaluate_retrieval(
            retrieved_docs, relevant_docs, relevance_scores
        )
        
        generation_metrics = self.evaluate_generation(
            generated_answer, 
            ground_truth_answer,
            generated_citations,
            ground_truth_citations
        )
        
        # End-to-end metrics
        end_to_end = {}
        
        # Answer relevance (simple overlap)
        gen_tokens = set(_tokenize(generated_answer))
        ref_tokens = set(_tokenize(ground_truth_answer))
        if gen_tokens:
            end_to_end["answer_overlap"] = len(gen_tokens & ref_tokens) / len(gen_tokens)
        else:
            end_to_end["answer_overlap"] = 0.0
        
        return EvaluationResult(
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            end_to_end_metrics=end_to_end
        )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Retrieval metrics
    'precision_at_k', 'recall_at_k', 'mean_reciprocal_rank',
    'ndcg_at_k', 'average_precision',
    # Generation metrics
    'bleu_score', 'rouge_l', 'citation_accuracy',
    # Evaluation
    'EvaluationResult', 'RAGEvaluator',
]
