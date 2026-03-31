"""
Production-Grade RAG Evaluation Module with RAGAS Integration

This module implements comprehensive evaluation capabilities for RAG systems,
following the 2026 production standards for measuring RAG quality. It includes
integration with RAGAS (RAG Assessment) framework and custom evaluation metrics
to assess different aspects of RAG performance.

The evaluation framework measures:
- Context Recall: How well the retrieval captures relevant information
- Faithfulness: How factually consistent the generation is with retrieved context
- Answer Relevancy: How relevant the answer is to the question
- Context Precision: How precise the retrieved context is
- Context Relevancy: How relevant the retrieved context is to the question

Key Features:
- RAGAS metric integration
- Custom evaluation metrics
- Batch evaluation capabilities
- Performance benchmarking
- Quality assurance workflows
- Integration with popular LLMs for evaluation

Example:
    >>> from src.eval import RAGEvaluator
    >>> evaluator = RAGEvaluator()
    >>> 
    >>> # Evaluate a single query-response pair
    >>> result = evaluator.evaluate_single(
    ...     question="What is AI?",
    ...     answer="Artificial Intelligence is...",
    ...     contexts=["AI is a branch of computer science..."],
    ...     ground_truth="Artificial Intelligence is a branch of computer science..."
    ... )
    >>> print(f"Faithfulness: {result['faithfulness']}")
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        context_relevancy,
    )
    from datasets import Dataset
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False
    print("RAGAS not available. Evaluation will be limited.")


@dataclass
class EvaluationResult:
    """
    Container for evaluation results.
    
    Attributes:
        faithfulness (float): Score measuring factual consistency (0-1)
        answer_relevancy (float): Score measuring answer relevance (0-1)
        context_recall (float): Score measuring retrieval completeness (0-1)
        context_precision (float): Score measuring retrieval precision (0-1)
        context_relevancy (float): Score measuring context relevance (0-1)
        latency (float): Time taken for evaluation in seconds
        total_score (float): Average of all metrics
    """
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    context_relevancy: float = 0.0
    latency: float = 0.0
    total_score: float = 0.0


class RAGEvaluator:
    """
    Main evaluation class for RAG system assessment.
    
    This class provides methods to evaluate different aspects of RAG performance
    using both automated metrics and custom evaluation functions. It supports
    both single-query evaluation and batch evaluation for continuous monitoring.
    
    Args:
        metrics (List[str]): List of metrics to evaluate ('faithfulness', 'answer_relevancy', etc.)
        llm_model (str): LLM model to use for evaluation (if needed)
        
    Example:
        >>> evaluator = RAGEvaluator(metrics=['faithfulness', 'answer_relevancy'])
        >>> result = evaluator.evaluate_single(
        ...     question="What is AI?",
        ...     answer="Artificial Intelligence is...",
        ...     contexts=["AI is a branch of computer science..."]
        ... )
        >>> print(f"Evaluation score: {result.total_score:.2f}")
    """
    
    def __init__(self, metrics: Optional[List[str]] = None, llm_model: str = "gpt-3.5-turbo"):
        """
        Initialize the evaluator with specified metrics.
        
        Args:
            metrics: List of metrics to evaluate. If None, uses all available metrics
            llm_model: LLM model to use for evaluation
        """
        if metrics is None:
            self.metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'context_relevancy']
        else:
            self.metrics = metrics
        self.llm_model = llm_model
        
        # Validate metrics
        available_metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'context_relevancy']
        for metric in self.metrics:
            if metric not in available_metrics:
                raise ValueError(f"Unknown metric: {metric}. Available: {available_metrics}")
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single query-response pair.
        
        Args:
            question: The original question
            answer: The generated answer
            contexts: List of retrieved contexts
            ground_truth: Ground truth answer (optional, needed for some metrics)
            
        Returns:
            EvaluationResult: Results of the evaluation
        """
        if not HAS_RAGAS:
            raise ImportError("RAGAS is required for evaluation. Please install it: pip install ragas")
        
        import time
        start_time = time.time()
        
        # Prepare data in RAGAS format
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        
        if ground_truth:
            data["ground_truth"] = [ground_truth]
        
        dataset = Dataset.from_dict(data)
        
        # Select metrics based on what's available and what's requested
        ragas_metrics = []
        if 'faithfulness' in self.metrics:
            ragas_metrics.append(faithfulness)
        if 'answer_relevancy' in self.metrics:
            ragas_metrics.append(answer_relevancy)
        if 'context_recall' in self.metrics and ground_truth:
            ragas_metrics.append(context_recall)
        if 'context_precision' in self.metrics:
            ragas_metrics.append(context_precision)
        if 'context_relevancy' in self.metrics:
            ragas_metrics.append(context_relevancy)
        
        # Run evaluation
        try:
            scores = evaluate(dataset, metrics=ragas_metrics)
            results = scores.to_pandas().iloc[0].to_dict()
        except Exception as e:
            print(f"Error during evaluation: {e}")
            # Return default values in case of error
            results = {metric: 0.0 for metric in self.metrics}
        
        # Calculate total score
        total_score = np.mean(list(results.values())) if results else 0.0
        
        latency = time.time() - start_time
        
        return EvaluationResult(
            faithfulness=results.get('faithfulness', 0.0),
            answer_relevancy=results.get('answer_relevancy', 0.0),
            context_recall=results.get('context_recall', 0.0),
            context_precision=results.get('context_precision', 0.0),
            context_relevancy=results.get('context_relevancy', 0.0),
            latency=latency,
            total_score=total_score
        )
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of query-response pairs.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts_list: List of context lists (one per question-answer pair)
            ground_truths: List of ground truth answers (optional)
            
        Returns:
            List[EvaluationResult]: Results for each query-response pair
        """
        if not HAS_RAGAS:
            raise ImportError("RAGAS is required for evaluation. Please install it: pip install ragas")
        
        results = []
        for i in range(len(questions)):
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            result = self.evaluate_single(
                question=questions[i],
                answer=answers[i],
                contexts=contexts_list[i],
                ground_truth=gt
            )
            results.append(result)
        
        return results
    
    def evaluate_pipeline(
        self,
        pipeline,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate an entire RAG pipeline on test cases.
        
        Args:
            pipeline: RAG pipeline object with a query method
            test_cases: List of test cases with 'question' and optional 'ground_truth'
            
        Returns:
            Dict[str, Any]: Summary evaluation results
        """
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []
        
        for case in test_cases:
            question = case['question']
            result = pipeline.query(question)
            
            questions.append(question)
            answers.append(result['response'])
            
            # Extract contexts from the pipeline result
            contexts = [doc['content'] for doc in result['retrieved_documents']]
            contexts_list.append(contexts)
            
            if 'ground_truth' in case:
                ground_truths.append(case['ground_truth'])
        
        # Perform batch evaluation
        results = self.evaluate_batch(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
            ground_truths=ground_truths if ground_truths else None
        )
        
        # Calculate aggregate metrics
        aggregate = {
            'faithfulness_avg': np.mean([r.faithfulness for r in results]),
            'answer_relevancy_avg': np.mean([r.answer_relevancy for r in results]),
            'context_recall_avg': np.mean([r.context_recall for r in results]),
            'context_precision_avg': np.mean([r.context_precision for r in results]),
            'context_relevancy_avg': np.mean([r.context_relevancy for r in results]),
            'total_score_avg': np.mean([r.total_score for r in results]),
            'latency_avg': np.mean([r.latency for r in results]),
            'latency_p95': np.percentile([r.latency for r in results], 95),
            'sample_count': len(results),
            'individual_results': results
        }
        
        return aggregate
    
    def generate_evaluation_report(self, results: List[EvaluationResult]) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            results: List of evaluation results
            
        Returns:
            str: Formatted evaluation report
        """
        if not results:
            return "No evaluation results to report."
        
        avg_faithfulness = np.mean([r.faithfulness for r in results])
        avg_answer_relevancy = np.mean([r.answer_relevancy for r in results])
        avg_context_recall = np.mean([r.context_recall for r in results])
        avg_context_precision = np.mean([r.context_precision for r in results])
        avg_context_relevancy = np.mean([r.context_relevancy for r in results])
        avg_total_score = np.mean([r.total_score for r in results])
        avg_latency = np.mean([r.latency for r in results])
        
        report = f"""
RAG Evaluation Report
=====================

Summary Statistics:
- Faithfulness: {avg_faithfulness:.3f}
- Answer Relevancy: {avg_answer_relevancy:.3f}
- Context Recall: {avg_context_recall:.3f}
- Context Precision: {avg_context_precision:.3f}
- Context Relevancy: {avg_context_relevancy:.3f}
- Overall Score: {avg_total_score:.3f}
- Avg Latency: {avg_latency:.3f}s

Sample Count: {len(results)}
        """.strip()
        
        return report


class CustomEvaluator:
    """
    Custom evaluation metrics for RAG systems.
    
    This class provides additional evaluation metrics that complement
    the RAGAS framework with custom business logic or domain-specific
    measurements.
    """
    
    @staticmethod
    def calculate_response_completeness(answer: str, ground_truth: str) -> float:
        """
        Calculate how complete the answer is compared to ground truth.
        
        Args:
            answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            float: Completeness score (0-1)
        """
        if not ground_truth or not answer:
            return 0.0
        
        # Simple word overlap approach
        gt_words = set(ground_truth.lower().split())
        ans_words = set(answer.lower().split())
        
        if not gt_words:
            return 0.0
        
        overlap = len(gt_words.intersection(ans_words))
        completeness = overlap / len(gt_words)
        
        return min(completeness, 1.0)  # Clamp to 1.0
    
    @staticmethod
    def calculate_citation_accuracy(answer: str, contexts: List[str]) -> float:
        """
        Calculate how accurately the answer cites information from contexts.
        
        Args:
            answer: Generated answer
            contexts: List of retrieved contexts
            
        Returns:
            float: Citation accuracy score (0-1)
        """
        if not contexts or not answer:
            return 0.0
        
        # Count how many context elements are referenced in the answer
        answer_lower = answer.lower()
        referenced_contexts = 0
        
        for context in contexts:
            if context.lower() in answer_lower:
                referenced_contexts += 1
        
        # Calculate accuracy as ratio of referenced contexts
        accuracy = referenced_contexts / len(contexts) if contexts else 0.0
        return min(accuracy, 1.0)
    
    @staticmethod
    def calculate_answer_conciseness(answer: str, max_length: int = 200) -> float:
        """
        Calculate how concise the answer is.
        
        Args:
            answer: Generated answer
            max_length: Maximum acceptable answer length
            
        Returns:
            float: Conciseness score (0-1), where 1 is perfectly concise
        """
        if not answer:
            return 0.0
        
        length_ratio = len(answer) / max_length
        # Score decreases as length goes beyond max_length
        if length_ratio <= 1.0:
            # Perfect or under limit gets high score
            return 1.0
        else:
            # Decrease score based on how much it exceeds the limit
            return max(0.0, 1.0 - (length_ratio - 1.0))


def create_default_evaluator() -> RAGEvaluator:
    """
    Create a default evaluator with commonly used metrics.
    
    Returns:
        RAGEvaluator: Default evaluator instance
    """
    return RAGEvaluator(
        metrics=['faithfulness', 'answer_relevancy', 'context_precision']
    )


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the evaluator
    if HAS_RAGAS:
        evaluator = RAGEvaluator()
        
        # Example evaluation
        result = evaluator.evaluate_single(
            question="What is artificial intelligence?",
            answer="Artificial Intelligence is a branch of computer science that aims to create software or machines that exhibit human-like intelligence.",
            contexts=[
                "Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
                "Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals."
            ],
            ground_truth="Artificial Intelligence is a branch of computer science that aims to create software or machines that exhibit human-like intelligence. It involves machine learning, neural networks, and other technologies."
        )
        
        print(f"Faithfulness: {result.faithfulness:.3f}")
        print(f"Answer Relevancy: {result.answer_relevancy:.3f}")
        print(f"Context Recall: {result.context_recall:.3f}")
        print(f"Total Score: {result.total_score:.3f}")
    else:
        print("RAGAS not available. Please install to run evaluations.")