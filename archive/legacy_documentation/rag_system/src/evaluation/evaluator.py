"""
RAG Evaluation Framework for Arabic Islamic Literature

Following RAG Pipeline Guide 2026 - Phase 7: Evaluation & Monitoring

This module provides comprehensive evaluation for:
- Retrieval Quality (Precision, Recall, MRR)
- Generation Quality (Faithfulness, Relevance)
- System Performance (Latency, Cost)
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import logging

logger = logging.getLogger(__name__)


# ==================== Data Models ====================


@dataclass
class EvaluationSample:
    """A single evaluation sample."""

    query: str
    ground_truth_answer: str
    relevant_book_titles: List[str] = field(default_factory=list)
    category: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    average_precision: float = 0.0


@dataclass
class GenerationMetrics:
    """Generation evaluation metrics."""

    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    hallucination_rate: float = 0.0


@dataclass
class SystemMetrics:
    """System performance metrics."""

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    tokens_per_query: float = 0.0
    cost_per_query_usd: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    sample: EvaluationSample
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    system_metrics: SystemMetrics
    latency_ms: float
    timestamp: str


# ==================== Arabic Test Dataset ====================


class ArabicTestDataset:
    """
    Pre-built Arabic evaluation dataset for Islamic literature.
    """

    # Arabic test questions covering different categories
    ARABIC_SAMPLES = [
        # Theology (العقيدة)
        EvaluationSample(
            query="ما هو التوحيد وما أركانه؟",
            ground_truth_answer="التوحيد هو إفراد الله بالعبادة وأركانه: توحيد الربوبية، توحيد الألوهية، توحيد الأسماء والصفات",
            relevant_book_titles=["التوحيد", "الواسطية", "العقيدة"],
            category="العقيدة",
            difficulty="easy",
        ),
        EvaluationSample(
            query="ما هي أسماء الله الحسنى؟",
            ground_truth_answer="أسماء الله الحسنى هي التسعة والتسعون اسماً ذكرها الله في القرآن",
            relevant_book_titles=["تفسير أسماء الله الحسنى"],
            category="العقيدة",
            difficulty="easy",
        ),
        # Quranic Exegesis (التفسير)
        EvaluationSample(
            query="ما تفسير سورة الإخلاص؟",
            ground_truth_answer="سورة الإخلاص تثبت صفات الله وتُنزله النبي صلى الله عليه وسلم",
            relevant_book_titles=["تفسير ابن كثير", "تفسير القرطبي"],
            category="التفسير",
            difficulty="easy",
        ),
        # Hadith (السنة)
        EvaluationSample(
            query="ما حديث يوم عرفة؟",
            ground_truth_answer="حديث يوم عرفة من أصح الأحاديث في بيان دين الإسلام",
            relevant_book_titles=["صحيح البخاري", "صحيح مسلم"],
            category="كتب السنة",
            difficulty="medium",
        ),
        # Fiqh (الفقه)
        EvaluationSample(
            query="ما شروط الصلاة؟",
            ground_truth_answer="شروط الصلاة: الطهارة، استقبال القبلة، دخول الوقت،遮挡 العورة",
            relevant_book_titles=["الرياض", "المغني"],
            category="الفقه وأصوله",
            difficulty="easy",
        ),
        # Spirituality (الرقائق)
        EvaluationSample(
            query="ما أهمية الدعاء في الإسلام؟",
            ground_truth_answer="الدعاء عبادة وهو سلاح المؤمن",
            relevant_book_titles=["الرقائق", "الأذكار"],
            category="الرقائق والآداب والأذكار",
            difficulty="easy",
        ),
        # Biography (التراجم)
        EvaluationSample(
            query="من هو الصحابي أبو بكر الصديق؟",
            ground_truth_answer="أبو بكر الصديق أول الخلفاء الراشدين وأفضل الصحابة",
            relevant_book_titles=["الطبقات", "الاستيعاب"],
            category="التراجم والطبقات",
            difficulty="easy",
        ),
        # Arabic Language (اللغة)
        EvaluationSample(
            query="ما هي أدوات النصب في اللغة العربية؟",
            ground_truth_answer="أدوات النصب: أن، لن، كي، حتى، لام التعليل، فاء السببية",
            relevant_book_titles=["النحو الواضح", "قطر الندى"],
            category="اللغة العربية",
            difficulty="medium",
        ),
    ]

    # English test questions
    ENGLISH_SAMPLES = [
        EvaluationSample(
            query="What is Tawhid in Islam?",
            ground_truth_answer="Tawhid is the concept of divine oneness in Islam",
            relevant_book_titles=["Tawhid", "Islamic Theology"],
            category="Theology",
            difficulty="easy",
        ),
        EvaluationSample(
            query="Explain the Five Pillars of Islam",
            ground_truth_answer="The Five Pillars are: Shahada, Salat, Zakat, Sawm, Hajj",
            relevant_book_titles=["Islamic Jurisprudence"],
            category="Fiqh",
            difficulty="easy",
        ),
    ]

    @classmethod
    def get_samples(
        cls,
        language: str = "both",
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> List[EvaluationSample]:
        """Get evaluation samples based on filters."""

        samples = []

        if language == "arabic":
            samples = cls.ARABIC_SAMPLES
        elif language == "english":
            samples = cls.ENGLISH_SAMPLES
        else:
            samples = cls.ARABIC_SAMPLES + cls.ENGLISH_SAMPLES

        # Apply filters
        if category:
            samples = [s for s in samples if s.category == category]

        if difficulty:
            samples = [s for s in samples if s.difficulty == difficulty]

        return samples

    @classmethod
    def save_samples(cls, filepath: str, language: str = "both"):
        """Save evaluation samples to JSON file."""

        samples = cls.get_samples(language=language)

        data = [
            {
                "query": s.query,
                "ground_truth_answer": s.ground_truth_answer,
                "relevant_book_titles": s.relevant_book_titles,
                "category": s.category,
                "difficulty": s.difficulty,
            }
            for s in samples
        ]

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(samples)} evaluation samples to {filepath}")


# ==================== RAG Evaluator ====================


class RAGEvaluator:
    """
    Comprehensive RAG evaluation system.

    Evaluates:
    - Retrieval Quality: Precision@K, Recall@K, MRR, NDCG
    - Generation Quality: Faithfulness, Relevance (using LLM-as-judge)
    - System Performance: Latency, Cost
    """

    def __init__(
        self,
        pipeline: Any,
        embedding_model: Any = None,
        llm_client: Any = None,
    ):
        self.pipeline = pipeline
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.results: List[EvaluationResult] = []

    async def evaluate_sample(
        self,
        sample: EvaluationSample,
        top_k: int = 5,
    ) -> EvaluationResult:
        """Evaluate a single sample."""

        start_time = time.time()

        # Step 1: Retrieve documents
        try:
            query_result = await self.pipeline.query(
                question=sample.query,
                top_k=top_k,
            )
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise

        latency_ms = (time.time() - start_time) * 1000

        # Step 2: Calculate retrieval metrics
        retrieval_metrics = self._evaluate_retrieval(
            query_result=query_result,
            sample=sample,
            top_k=top_k,
        )

        # Step 3: Calculate generation metrics (if LLM available)
        generation_metrics = GenerationMetrics()

        if self.llm_client:
            generation_metrics = await self._evaluate_generation(
                query=sample.query,
                answer=query_result.answer,
                retrieved_sources=query_result.sources,
                ground_truth=sample.ground_truth_answer,
            )

        # Step 4: Calculate system metrics
        system_metrics = SystemMetrics(
            latency_p50_ms=latency_ms,
            latency_p95_ms=latency_ms,
            latency_p99_ms=latency_ms,
            tokens_per_query=query_result.tokens_used,
        )

        result = EvaluationResult(
            sample=sample,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            system_metrics=system_metrics,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat(),
        )

        self.results.append(result)

        return result

    def _evaluate_retrieval(
        self,
        query_result: Any,
        sample: EvaluationSample,
        top_k: int,
    ) -> RetrievalMetrics:
        """Calculate retrieval metrics."""

        metrics = RetrievalMetrics()

        retrieved_books = [source["book_title"] for source in query_result.sources]

        relevant_books = sample.relevant_book_titles

        # Precision@K
        relevant_retrieved = sum(
            1
            for book in retrieved_books[:top_k]
            if any(r.lower() in book.lower() for r in relevant_books)
        )
        metrics.precision_at_k = relevant_retrieved / top_k if top_k > 0 else 0

        # Recall@K
        metrics.recall_at_k = (
            relevant_retrieved / len(relevant_books) if relevant_books else 0
        )

        # MRR (Mean Reciprocal Rank)
        for i, book in enumerate(retrieved_books[:top_k], 1):
            if any(r.lower() in book.lower() for r in relevant_books):
                metrics.mrr = 1.0 / i
                break

        # NDCG (simplified)
        dcg = 0.0
        for i, book in enumerate(retrieved_books[:top_k], 1):
            if any(r.lower() in book.lower() for r in relevant_books):
                dcg += 1.0 / i

        idcg = sum(1.0 / i for i in range(1, min(len(relevant_books), top_k) + 1))
        metrics.ndcg = dcg / idcg if idcg > 0 else 0

        # Average Precision
        precisions = []
        relevant_count = 0
        for i, book in enumerate(retrieved_books[:top_k], 1):
            if any(r.lower() in book.lower() for r in relevant_books):
                relevant_count += 1
                precisions.append(relevant_count / i)

        metrics.average_precision = (
            sum(precisions) / len(relevant_books) if relevant_books else 0
        )

        return metrics

    async def _evaluate_generation(
        self,
        query: str,
        answer: str,
        retrieved_sources: List[Dict],
        ground_truth: str,
    ) -> GenerationMetrics:
        """Evaluate generation quality using LLM-as-judge."""

        metrics = GenerationMetrics()

        if not self.llm_client:
            return metrics

        try:
            # Faithfulness check
            context = "\n\n".join(
                [src.get("content_preview", "")[:500] for src in retrieved_sources[:3]]
            )

            prompt = f"""Evaluate whether the following answer is grounded in the provided context.

Context:
{context}

Answer:
{answer}

Is the answer fully supported by the context? Rate from 1-5:
1 = Not supported at all
3 = Partially supported
5 = Fully supported

Respond with just the number."""

            result = await self.llm_client.generate(prompt)

            try:
                score = int(result.answer.strip()[0])
                metrics.faithfulness = score / 5.0
            except:
                metrics.faithfulness = 0.5

            # Answer relevance
            prompt = f"""Evaluate whether the answer addresses the question.

Question:
{query}

Answer:
{answer}

Does the answer address the question? Rate from 1-5:
1 = Doesn't address at all
3 = Partially addresses
5 = Fully addresses

Respond with just the number."""

            result = await self.llm_client.generate(prompt)

            try:
                score = int(result.answer.strip()[0])
                metrics.answer_relevance = score / 5.0
            except:
                metrics.answer_relevance = 0.5

        except Exception as e:
            logger.error(f"Generation evaluation error: {e}")

        return metrics

    async def evaluate_dataset(
        self,
        samples: List[EvaluationSample],
        top_k: int = 5,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Evaluate entire dataset."""

        logger.info(f"Evaluating {len(samples)} samples...")

        results = []

        for i, sample in enumerate(samples):
            try:
                result = await self.evaluate_sample(sample, top_k=top_k)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(samples))

            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")

        # Calculate aggregate metrics
        return self.compute_aggregate_metrics(results)

    def compute_aggregate_metrics(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """Calculate aggregate metrics across all results."""

        if not results:
            return {"error": "No results to aggregate"}

        # Retrieval metrics
        retrieval_precisions = [r.retrieval_metrics.precision_at_k for r in results]
        retrieval_recalls = [r.retrieval_metrics.recall_at_k for r in results]
        retrieval_mrrs = [r.retrieval_metrics.mrr for r in results]
        retrieval_ndcgs = [r.retrieval_metrics.ndcg for r in results]

        # Generation metrics
        faithfulness = [r.generation_metrics.faithfulness for r in results]
        relevance = [r.generation_metrics.answer_relevance for r in results]

        # System metrics
        latencies = [r.system_metrics.latency_p50_ms for r in results]

        # Compute aggregates
        return {
            "retrieval": {
                "precision_at_k": statistics.mean(retrieval_precisions),
                "recall_at_k": statistics.mean(retrieval_recalls),
                "mrr": statistics.mean(retrieval_mrrs),
                "ndcg": statistics.mean(retrieval_ndcgs),
            },
            "generation": {
                "faithfulness": statistics.mean(faithfulness) if faithfulness else 0,
                "answer_relevance": statistics.mean(relevance) if relevance else 0,
            },
            "system": {
                "avg_latency_ms": statistics.mean(latencies),
                "p50_latency_ms": statistics.median(latencies),
                "p95_latency_ms": (
                    sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
                ),
            },
            "total_samples": len(results),
        }

    def save_results(self, filepath: str):
        """Save evaluation results to file."""

        data = []

        for result in self.results:
            data.append(
                {
                    "query": result.sample.query,
                    "ground_truth": result.sample.ground_truth_answer,
                    "retrieval": {
                        "precision_at_k": result.retrieval_metrics.precision_at_k,
                        "recall_at_k": result.retrieval_metrics.recall_at_k,
                        "mrr": result.retrieval_metrics.mrr,
                        "ndcg": result.retrieval_metrics.ndcg,
                    },
                    "generation": {
                        "faithfulness": result.generation_metrics.faithfulness,
                        "answer_relevance": result.generation_metrics.answer_relevance,
                    },
                    "system": {
                        "latency_ms": result.latency_ms,
                    },
                    "timestamp": result.timestamp,
                }
            )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved evaluation results to {filepath}")


# ==================== Usage Example ====================


async def run_evaluation():
    """Example evaluation run."""

    from src.pipeline.complete_pipeline import create_rag_pipeline, RAGConfig

    # Create pipeline
    config = RAGConfig(
        llm_provider="mock",
    )
    pipeline = create_rag_pipeline(config)

    # Try loading existing index
    try:
        pipeline.load_indexes()
    except:
        print("No index available. Please index first.")
        return

    # Create evaluator
    evaluator = RAGEvaluator(pipeline=pipeline)

    # Get test samples
    samples = ArabicTestDataset.get_samples(language="arabic")

    # Run evaluation
    results = await evaluator.evaluate_dataset(
        samples=samples[:3],  # Test with 3 samples
        top_k=5,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print("\nRetrieval Metrics:")
    for key, value in results["retrieval"].items():
        print(f"  {key}: {value:.3f}")

    print("\nGeneration Metrics:")
    for key, value in results["generation"].items():
        print(f"  {key}: {value:.3f}")

    print("\nSystem Metrics:")
    for key, value in results["system"].items():
        print(f"  {key}: {value:.2f}")

    # Save results
    evaluator.save_results("data/evaluation_results.json")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
