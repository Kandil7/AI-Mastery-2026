"""
RAG Evaluation Module

Production-ready RAG evaluation with:
- RAGAS integration
- Context precision metrics
- Faithfulness metrics
- Answer relevancy metrics

Features:
- Automated evaluation pipelines
- Batch evaluation
- Report generation
- Regression testing
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """A single evaluation sample."""

    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a single sample."""

    sample: EvaluationSample
    scores: Dict[str, float]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.sample.question,
            "answer": self.sample.answer,
            "scores": self.scores,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""

    total_samples: int
    scores: Dict[str, Dict[str, float]]  # metric -> {mean, std, min, max}
    sample_results: List[EvaluationResult]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "scores": self.scores,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved evaluation report to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationReport":
        """Load report from file."""
        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            total_samples=data["total_samples"],
            scores=data["scores"],
            sample_results=[],  # Not loaded from file
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    async def score(self, sample: EvaluationSample) -> float:
        """Calculate score for a single sample."""
        pass

    @abstractmethod
    async def score_batch(self, samples: List[EvaluationSample]) -> List[float]:
        """Calculate scores for batch of samples."""
        pass


class ContextPrecisionMetric(BaseMetric):
    """
    Context Precision metric.

    Measures whether relevant items in the context are
    ranked higher than irrelevant ones.
    """

    def __init__(self, llm_client: Any = None) -> None:
        super().__init__("context_precision")
        self.llm_client = llm_client

    async def score(self, sample: EvaluationSample) -> float:
        """Calculate context precision."""
        if not sample.ground_truth or not sample.contexts:
            return 0.0

        # Check if ground truth is in contexts
        relevance_scores = await self._evaluate_relevance(
            sample.question,
            sample.contexts,
            sample.ground_truth,
        )

        # Calculate precision at each position
        precisions = []
        relevant_count = 0

        for i, score in enumerate(relevance_scores):
            if score > 0.5:
                relevant_count += 1
            precisions.append(relevant_count / (i + 1))

        if not precisions:
            return 0.0

        return statistics.mean(precisions)

    async def _evaluate_relevance(
        self,
        question: str,
        contexts: List[str],
        ground_truth: str,
    ) -> List[float]:
        """Evaluate relevance of each context."""
        if not self.llm_client:
            # Fallback: simple string matching
            return [
                1.0 if ground_truth.lower() in ctx.lower() else 0.0
                for ctx in contexts
            ]

        scores = []
        for ctx in contexts:
            prompt = f"""Does the following context contain information relevant to answering the question?
Answer with just a number between 0 and 1.

Question: {question}
Context: {ctx[:1000]}

Relevance score (0-1):"""

            try:
                response = await self.llm_client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=5,
                )
                content = response.content if hasattr(response, 'content') else str(response)
                score = float(content.strip())
                scores.append(max(0, min(1, score)))
            except Exception:
                scores.append(0.5)

        return scores

    async def score_batch(self, samples: List[EvaluationSample]) -> List[float]:
        """Calculate batch scores."""
        tasks = [self.score(sample) for sample in samples]
        return await asyncio.gather(*tasks)


class FaithfulnessMetric(BaseMetric):
    """
    Faithfulness metric.

    Measures whether the answer is factually consistent
    with the provided context (no hallucinations).
    """

    FAITHFULNESS_PROMPT = """Given the context and answer, determine if the answer is faithful to the context.
The answer should only contain information that can be inferred from the context.

Context: {context}

Answer: {answer}

Is the answer faithful to the context? Answer with just a number between 0 and 1, where:
1 = Completely faithful (all information in answer is from context)
0 = Not faithful at all (answer contains hallucinated information)

Faithfulness score:"""

    def __init__(self, llm_client: Any) -> None:
        super().__init__("faithfulness")
        self.llm_client = llm_client

    async def score(self, sample: EvaluationSample) -> float:
        """Calculate faithfulness score."""
        if not sample.contexts:
            return 0.0

        # Combine contexts
        full_context = "\n\n".join(sample.contexts)

        prompt = self.FAITHFULNESS_PROMPT.format(
            context=full_context[:3000],
            answer=sample.answer[:1000],
        )

        try:
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
            )
            content = response.content if hasattr(response, 'content') else str(response)

            # Extract score
            import re
            match = re.search(r'(\d+\.?\d*)', content)
            if match:
                score = float(match.group(1))
                return max(0, min(1, score))
            return 0.5
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return 0.5

    async def score_batch(self, samples: List[EvaluationSample]) -> List[float]:
        """Calculate batch scores."""
        # Process in parallel with concurrency limit
        semaphore = asyncio.Semaphore(5)

        async def limited_score(sample: EvaluationSample) -> float:
            async with semaphore:
                return await self.score(sample)

        tasks = [limited_score(sample) for sample in samples]
        return await asyncio.gather(*tasks)


class AnswerRelevancyMetric(BaseMetric):
    """
    Answer Relevancy metric.

    Measures how relevant the answer is to the question.
    """

    RELEVANCY_PROMPT = """Evaluate how relevant the answer is to the question.
Consider:
- Does the answer directly address the question?
- Is the answer on-topic?
- Does the answer provide useful information?

Question: {question}

Answer: {answer}

Relevancy score (0-1, where 1 is perfectly relevant):"""

    def __init__(self, llm_client: Any) -> None:
        super().__init__("answer_relevancy")
        self.llm_client = llm_client

    async def score(self, sample: EvaluationSample) -> float:
        """Calculate answer relevancy."""
        prompt = self.RELEVANCY_PROMPT.format(
            question=sample.question,
            answer=sample.answer[:1000],
        )

        try:
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
            )
            content = response.content if hasattr(response, 'content') else str(response)

            import re
            match = re.search(r'(\d+\.?\d*)', content)
            if match:
                score = float(match.group(1))
                return max(0, min(1, score))
            return 0.5
        except Exception as e:
            logger.warning(f"Relevancy evaluation failed: {e}")
            return 0.5

    async def score_batch(self, samples: List[EvaluationSample]) -> List[float]:
        """Calculate batch scores."""
        semaphore = asyncio.Semaphore(5)

        async def limited_score(sample: EvaluationSample) -> float:
            async with semaphore:
                return await self.score(sample)

        tasks = [limited_score(sample) for sample in samples]
        return await asyncio.gather(*tasks)


class ContextRecallMetric(BaseMetric):
    """
    Context Recall metric.

    Measures whether the ground truth can be found in the retrieved context.
    """

    RECALL_PROMPT = """Given the ground truth answer and the context, determine what portion of the ground truth can be found in the context.

Ground Truth: {ground_truth}

Context: {context}

What percentage of the ground truth information is present in the context? Answer with a number between 0 and 1.

Recall score:"""

    def __init__(self, llm_client: Any) -> None:
        super().__init__("context_recall")
        self.llm_client = llm_client

    async def score(self, sample: EvaluationSample) -> float:
        """Calculate context recall."""
        if not sample.ground_truth or not sample.contexts:
            return 0.0

        full_context = "\n\n".join(sample.contexts)

        prompt = self.RECALL_PROMPT.format(
            ground_truth=sample.ground_truth[:500],
            context=full_context[:3000],
        )

        try:
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
            )
            content = response.content if hasattr(response, 'content') else str(response)

            import re
            match = re.search(r'(\d+\.?\d*)', content)
            if match:
                score = float(match.group(1))
                return max(0, min(1, score))
            return 0.5
        except Exception as e:
            logger.warning(f"Recall evaluation failed: {e}")
            return 0.5

    async def score_batch(self, samples: List[EvaluationSample]) -> List[float]:
        """Calculate batch scores."""
        semaphore = asyncio.Semaphore(5)

        async def limited_score(sample: EvaluationSample) -> float:
            async with semaphore:
                return await self.score(sample)

        tasks = [limited_score(sample) for sample in samples]
        return await asyncio.gather(*tasks)


class RAGASWrapper:
    """
    Wrapper for RAGAS evaluation framework.

    Provides integration with the RAGAS library when available,
    with fallback to custom implementations.
    """

    def __init__(self, llm_client: Any, embeddings: Any = None) -> None:
        self.llm_client = llm_client
        self.embeddings = embeddings

        self._ragas = None
        self._metrics = []

        self._try_import_ragas()

    def _try_import_ragas(self) -> None:
        """Try to import RAGAS library."""
        try:
            import ragas
            from ragas.metrics import (
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            )

            self._ragas = ragas
            self._metrics = [
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ]
            logger.info("RAGAS library imported successfully")
        except ImportError:
            logger.warning("RAGAS not installed. Using custom metrics.")
            self._ragas = None

    async def evaluate(
        self,
        samples: List[EvaluationSample],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate samples using RAGAS.

        Args:
            samples: Evaluation samples
            metrics: List of metric names to use

        Returns:
            Dictionary of metric scores
        """
        if self._ragas:
            return await self._evaluate_with_ragas(samples, metrics)
        else:
            return await self._evaluate_custom(samples, metrics)

    async def _evaluate_with_ragas(
        self,
        samples: List[EvaluationSample],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate using RAGAS library."""
        # Convert to RAGAS dataset format
        from datasets import Dataset

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
        }

        if samples[0].ground_truth:
            data["ground_truth"] = [s.ground_truth for s in samples]

        dataset = Dataset.from_dict(data)

        # Run evaluation
        from ragas import evaluate as ragas_evaluate

        result = ragas_evaluate(
            dataset=dataset,
            metrics=self._metrics,
            llm=self.llm_client,
            embeddings=self.embeddings,
        )

        return result.scores

    async def _evaluate_custom(
        self,
        samples: List[EvaluationSample],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate using custom metrics."""
        metrics = metrics or ["context_precision", "faithfulness", "answer_relevancy"]

        scores = {}

        if "context_precision" in metrics:
            metric = ContextPrecisionMetric(self.llm_client)
            results = await metric.score_batch(samples)
            scores["context_precision"] = statistics.mean(results) if results else 0

        if "faithfulness" in metrics:
            metric = FaithfulnessMetric(self.llm_client)
            results = await metric.score_batch(samples)
            scores["faithfulness"] = statistics.mean(results) if results else 0

        if "answer_relevancy" in metrics:
            metric = AnswerRelevancyMetric(self.llm_client)
            results = await metric.score_batch(samples)
            scores["answer_relevancy"] = statistics.mean(results) if results else 0

        if "context_recall" in metrics:
            metric = ContextRecallMetric(self.llm_client)
            results = await metric.score_batch(samples)
            scores["context_recall"] = statistics.mean(results) if results else 0

        return scores


class RAGEvaluator:
    """
    Main RAG evaluation orchestrator.

    Coordinates evaluation across multiple metrics and
    generates comprehensive reports.
    """

    def __init__(
        self,
        llm_client: Any,
        embeddings: Any = None,
        metrics: Optional[List[str]] = None,
    ) -> None:
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.metrics = metrics or [
            "context_precision",
            "faithfulness",
            "answer_relevancy",
            "context_recall",
        ]

        self.ragas_wrapper = RAGASWrapper(llm_client, embeddings)

        # Custom metrics
        self._custom_metrics: Dict[str, BaseMetric] = {
            "context_precision": ContextPrecisionMetric(llm_client),
            "faithfulness": FaithfulnessMetric(llm_client),
            "answer_relevancy": AnswerRelevancyMetric(llm_client),
            "context_recall": ContextRecallMetric(llm_client),
        }

        self._evaluation_history: List[EvaluationReport] = []

    async def evaluate(
        self,
        samples: List[EvaluationSample],
        batch_size: int = 10,
    ) -> EvaluationReport:
        """
        Evaluate RAG system on samples.

        Args:
            samples: Evaluation samples
            batch_size: Batch size for processing

        Returns:
            Evaluation report
        """
        start_time = time.time()

        # Process in batches
        all_results: List[EvaluationResult] = []

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batch_results = await self._evaluate_batch(batch)
            all_results.extend(batch_results)

            logger.info(f"Evaluated batch {i // batch_size + 1}/{(len(samples) - 1) // batch_size + 1}")

        # Aggregate scores
        scores = self._aggregate_scores(all_results)

        total_time = (time.time() - start_time) * 1000

        report = EvaluationReport(
            total_samples=len(samples),
            scores=scores,
            sample_results=all_results,
            metadata={
                "evaluation_time_ms": total_time,
                "metrics": self.metrics,
            },
        )

        self._evaluation_history.append(report)

        logger.info(f"Evaluation complete: {len(samples)} samples in {total_time:.2f}ms")
        return report

    async def _evaluate_batch(
        self,
        samples: List[EvaluationSample],
    ) -> List[EvaluationResult]:
        """Evaluate a batch of samples."""
        results = []

        for sample in samples:
            sample_start = time.time()
            scores = {}

            for metric_name in self.metrics:
                if metric_name in self._custom_metrics:
                    metric = self._custom_metrics[metric_name]
                    score = await metric.score(sample)
                    scores[metric_name] = score

            latency_ms = (time.time() - sample_start) * 1000

            results.append(EvaluationResult(
                sample=sample,
                scores=scores,
                latency_ms=latency_ms,
            ))

        return results

    def _aggregate_scores(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate scores across all results."""
        aggregated = {}

        for metric_name in self.metrics:
            values = [r.scores.get(metric_name, 0) for r in results]

            if values:
                aggregated[metric_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "median": statistics.median(values),
                }

        return aggregated

    async def evaluate_from_rag(
        self,
        rag_orchestrator: Any,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> EvaluationReport:
        """
        Evaluate RAG system by running queries.

        Args:
            rag_orchestrator: RAG orchestrator to evaluate
            questions: List of test questions
            ground_truths: Optional ground truth answers

        Returns:
            Evaluation report
        """
        samples = []

        for i, question in enumerate(questions):
            # Run RAG query
            response = await rag_orchestrator.query(question)

            sample = EvaluationSample(
                question=question,
                answer=response.answer,
                contexts=[s.content for s in response.sources],
                ground_truth=ground_truths[i] if ground_truths else None,
                metadata={
                    "latency_ms": response.latency_ms,
                    "num_sources": len(response.sources),
                },
            )
            samples.append(sample)

        return await self.evaluate(samples)

    def compare_systems(
        self,
        reports: Dict[str, EvaluationReport],
    ) -> Dict[str, Any]:
        """
        Compare multiple RAG system evaluations.

        Args:
            reports: Dictionary of system name -> report

        Returns:
            Comparison results
        """
        comparison = {
            "systems": list(reports.keys()),
            "metrics": {},
        }

        for metric_name in self.metrics:
            comparison["metrics"][metric_name] = {}

            for system_name, report in reports.items():
                if metric_name in report.scores:
                    comparison["metrics"][metric_name][system_name] = report.scores[metric_name]["mean"]

        # Find best system per metric
        comparison["best_per_metric"] = {}
        for metric_name, systems in comparison["metrics"].items():
            if systems:
                best = max(systems.items(), key=lambda x: x[1])
                comparison["best_per_metric"][metric_name] = best[0]

        return comparison

    def get_history(self) -> List[EvaluationReport]:
        """Get evaluation history."""
        return self._evaluation_history.copy()


# Utility functions

def create_evaluation_samples(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: Optional[List[str]] = None,
) -> List[EvaluationSample]:
    """Create evaluation samples from lists."""
    samples = []

    for i, question in enumerate(questions):
        sample = EvaluationSample(
            question=question,
            answer=answers[i],
            contexts=contexts[i],
            ground_truth=ground_truths[i] if ground_truths else None,
        )
        samples.append(sample)

    return samples


def load_evaluation_samples(path: Union[str, Path]) -> List[EvaluationSample]:
    """Load evaluation samples from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    return [
        EvaluationSample(
            question=item["question"],
            answer=item["answer"],
            contexts=item["contexts"],
            ground_truth=item.get("ground_truth"),
            metadata=item.get("metadata", {}),
        )
        for item in data
    ]


def save_evaluation_samples(samples: List[EvaluationSample], path: Union[str, Path]) -> None:
    """Save evaluation samples to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump([s.to_dict() for s in samples], f, indent=2)

    logger.info(f"Saved {len(samples)} evaluation samples to {path}")
