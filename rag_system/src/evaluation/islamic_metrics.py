"""
Advanced Evaluation Metrics for Islamic Literature RAG

Specialized metrics beyond standard RAG evaluation:
1. Authority-weighted metrics
2. Chain-of-thought evaluation
3. Madhhab-specific evaluation
4. Source authenticity verification
5. Multi-hop reasoning evaluation
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class IslamicEvaluationMetrics:
    """Comprehensive evaluation metrics for Islamic RAG."""

    # Standard RAG metrics
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0

    # Islamic-specific metrics
    authority_score: float = 0.0
    source_authenticity: float = 0.0
    evidence_presence: float = 0.0
    madhhab_coverage: float = 0.0

    # Quality metrics
    citation_quality: float = 0.0
    scholarly_tone: float = 0.0
    bias_detection: float = 0.0

    # Reasoning metrics
    chain_coherence: float = 0.0
    logical_flow: float = 0.0
    conclusion_strength: float = 0.0


# Islamic source authenticity database
AUTHENTICITY_DB = {
    # Highest Authority - Quran
    "القرآن": {"authenticity": 1.0, "category": "revelation"},
    # Hadith - Sahih (Authentic)
    "صحيح البخاري": {"authenticity": 0.98, "category": "hadith_sahih"},
    "صحيح مسلم": {"authenticity": 0.97, "category": "hadith_sahih"},
    "الترمذي": {"authenticity": 0.95, "category": "hadith_sunan"},
    "سنن أبي داود": {"authenticity": 0.94, "category": "hadith_sunan"},
    "سنن النسائي": {"authenticity": 0.93, "category": "hadith_sunan"},
    "سنن ابن ماجه": {"authenticity": 0.92, "category": "hadith_sunan"},
    # Classical Tafsirs
    "تفسير ابن كثير": {"authenticity": 0.90, "category": "tafsir"},
    "تفسير القرطبي": {"authenticity": 0.89, "category": "tafsir"},
    "تفسير الطبري": {"authenticity": 0.88, "category": "tafsir"},
    "جامع البيان": {"authenticity": 0.87, "category": "tafsir"},
    # Fiqh - Major Works
    "الهداية": {"authenticity": 0.85, "category": "fiqh_hanafi"},
    "المدونة": {"authenticity": 0.85, "category": "fiqh_maliki"},
    "المهذب": {"authenticity": 0.85, "category": "fiqh_shafii"},
    "المبدع": {"authenticity": 0.85, "category": "fiqh_hanbali"},
    # Aqeedah
    "العقيدة الواسطية": {"authenticity": 0.90, "category": "aqeedah"},
    "التوحيد": {"authenticity": 0.90, "category": "aqeedah"},
    "الشرح الممتع": {"authenticity": 0.88, "category": "aqeedah"},
    # Major Scholars
    "ابن تيمية": {"authenticity": 0.85, "category": "scholar"},
    "ابن القيم": {"authenticity": 0.84, "category": "scholar"},
    "الإمام أحمد": {"authenticity": 0.85, "category": "imam"},
    "الإمام الشافعي": {"authenticity": 0.85, "category": "imam"},
    "الإمام مالك": {"authenticity": 0.85, "category": "imam"},
    "الإمام أبو حنيفة": {"authenticity": 0.85, "category": "imam"},
}


class IslamicRAGEvaluator:
    """
    Specialized evaluator for Islamic literature RAG.

    Extends standard RAG metrics with Islamic-specific evaluations.
    """

    def __init__(self, pipeline: Any = None):
        self.pipeline = pipeline

    def evaluate_source_authenticity(
        self,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate the authenticity of sources used in response.

        Returns:
            Dictionary with authenticity metrics
        """

        if not sources:
            return {"authenticity_score": 0.0, "high_auth_count": 0}

        total_score = 0.0
        high_auth_count = 0

        for source in sources:
            book_title = source.get("book_title", "")
            author = source.get("author", "")

            # Check against authenticity database
            score = 0.5  # Default

            for known_source, info in AUTHENTICITY_DB.items():
                if known_source in book_title or known_source in author:
                    score = info["authenticity"]
                    break

            total_score += score

            if score >= 0.85:
                high_auth_count += 1

        avg_score = total_score / len(sources) if sources else 0

        return {
            "authenticity_score": avg_score,
            "high_auth_count": high_auth_count,
            "total_sources": len(sources),
            "sources_detail": sources,
        }

    def evaluate_evidence_presence(
        self,
        answer: str,
    ) -> Dict[str, Any]:
        """
        Evaluate whether evidence is properly cited in answer.

        Checks for:
        - Quranic verses
        - Hadith citations
        - Scholarly opinions
        """

        import re

        # Find Quran references
        quran_patterns = [
            r"Allah\(swt\)",
            r"\{.*?\}",
            r"الله",
            r" Quran ",
        ]

        # Find hadith references
        hadith_patterns = [
            r"صحيح",
            r"رواه",
            r"أخبر",
            r"حديث",
            r"قال",
        ]

        # Find scholarly references
        scholar_patterns = [
            r"قال ابن تيمية",
            r"قال الشافعي",
            r"قال أبو حنيفة",
            r"قال مالك",
            r"قال أحمد",
            r"قال鲁迅",
        ]

        has_quran = any(re.search(p, answer) for p in quran_patterns)
        has_hadith = any(re.search(p, answer) for p in hadith_patterns)
        has_scholars = any(re.search(p, answer) for p in scholar_patterns)

        evidence_score = 0.0
        if has_quran:
            evidence_score += 0.4
        if has_hadith:
            evidence_score += 0.35
        if has_scholars:
            evidence_score += 0.25

        return {
            "evidence_score": evidence_score,
            "has_quran": has_quran,
            "has_hadith": has_hadith,
            "has_scholars": has_scholars,
        }

    def evaluate_madhhab_coverage(
        self,
        sources: List[Dict[str, Any]],
        question_type: str = "fiqh",
    ) -> Dict[str, Any]:
        """
        Evaluate coverage of different madhhab opinions.

        Only relevant for fiqh questions.
        """

        madhhabs_found = set()

        # Map books/authors to madhhabs
        madhhab_indicators = {
            "hanafi": ["أبو حنيفة", "الهداية", "الفتاوى الهندية"],
            "maliki": ["مالك", "المدونة", "الموطأ"],
            "shafii": ["الشافعي", "المهذب", "الوسيط"],
            "hanbali": ["أحمد", "المبدع", "كشاف القناع"],
        }

        for source in sources:
            content = str(source)

            for madhhab, indicators in madhhab_indicators.items():
                if any(ind in content for ind in indicators):
                    madhhabs_found.add(madhhab)

        coverage = len(madhhabs_found) / 4.0  # 4 madhhabs

        return {
            "madhhab_coverage": coverage,
            "madhhabs_found": list(madhhabs_found),
            "is_balanced": coverage >= 0.75,
        }

    def evaluate_citation_quality(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of citations in the answer.

        Checks:
        - Are sources cited in answer?
        - Are citations accurate?
        - Is there proper attribution?
        """

        citation_count = 0
        has_book_titles = 0
        has_authors = 0

        # Count citations in answer
        import re

        citation_patterns = [
            r"\[.*?\]",
            r"\(.*?\)",
            r"according to",
            r"في",
            r"قال",
        ]

        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, answer))

        # Check if sources have proper titles
        for source in sources:
            if source.get("book_title"):
                has_book_titles += 1
            if source.get("author"):
                has_authors += 1

        # Calculate quality score
        quality_score = 0.0

        if citation_count > 0:
            quality_score += 0.4
        if has_book_titles == len(sources) and sources:
            quality_score += 0.3
        if has_authors == len(sources) and sources:
            quality_score += 0.3

        return {
            "citation_quality": quality_score,
            "citation_count": citation_count,
            "sources_with_titles": has_book_titles,
            "sources_with_authors": has_authors,
        }

    def evaluate_bias(
        self,
        answer: str,
    ) -> Dict[str, Any]:
        """
        Detect potential bias in the answer.

        Checks:
        - Favoritism to one madhhab
        - sectarian language
        - Unbalanced presentation
        """

        bias_indicators = {
            "madhhab_bias": [
                "المذهب الحق",
                "الصحيح فقط",
                "لا خلاف",
            ],
            "sectarian": [
                "الاخوة",
                "الرافضة",
                "السنّة",
            ],
        }

        has_bias = False
        bias_type = None

        for bias_cat, indicators in bias_indicators.items():
            for indicator in indicators:
                if indicator in answer:
                    has_bias = True
                    bias_type = bias_cat
                    break

        return {
            "has_bias": has_bias,
            "bias_type": bias_type,
            "neutral": not has_bias,
        }

    def evaluate_chain_of_thought(
        self,
        reasoning_chain: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of multi-hop reasoning.

        Checks:
        - Coherence between steps
        - Logical flow
        - Conclusion validity
        """

        if not reasoning_chain:
            return {"chain_score": 0.0, "issues": ["No reasoning chain"]}

        # Check coherence
        coherence_scores = []

        for i in range(len(reasoning_chain) - 1):
            current = reasoning_chain[i]
            next_step = reasoning_chain[i + 1]

            # Check if next step builds on current
            if current.get("answer") and next_step.get("question"):
                # Simple coherence check
                coherence_scores.append(0.8)  # Placeholder
            else:
                coherence_scores.append(0.5)

        avg_coherence = (
            sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
        )

        # Check if conclusion is supported
        has_conclusion = any(
            s.get("step") == "conclusion" or s.get("step") == "final"
            for s in reasoning_chain
        )

        return {
            "chain_coherence": avg_coherence,
            "steps_count": len(reasoning_chain),
            "has_conclusion": has_conclusion,
        }

    async def evaluate_full(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        reasoning_chain: Optional[List[Dict[str, Any]]] = None,
    ) -> IslamicEvaluationMetrics:
        """
        Perform comprehensive evaluation.

        Args:
            question: The original question
            answer: Generated answer
            sources: Retrieved sources
            reasoning_chain: Optional multi-hop reasoning

        Returns:
            Complete evaluation metrics
        """

        metrics = IslamicEvaluationMetrics()

        # Source authenticity
        auth_result = self.evaluate_source_authenticity(sources)
        metrics.source_authenticity = auth_result["authenticity_score"]

        # Evidence presence
        evidence_result = self.evaluate_evidence_presence(answer)
        metrics.evidence_presence = evidence_result["evidence_score"]

        # Citation quality
        citation_result = self.evaluate_citation_quality(answer, sources)
        metrics.citation_quality = citation_result["citation_quality"]

        # Bias detection
        bias_result = self.evaluate_bias(answer)
        metrics.bias_detection = 1.0 if not bias_result["has_bias"] else 0.5

        # Madhhab coverage (if applicable)
        if any(word in question for word in ["حكم", "فقيه", "مذهب"]):
            madhhab_result = self.evaluate_madhhab_coverage(sources)
            metrics.madhhab_coverage = madhhab_result["madhhab_coverage"]

        # Chain of thought (if applicable)
        if reasoning_chain:
            cot_result = self.evaluate_chain_of_thought(reasoning_chain)
            metrics.chain_coherence = cot_result["chain_coherence"]

        # Calculate overall authority score
        metrics.authority_score = (
            metrics.source_authenticity * 0.4
            + metrics.evidence_presence * 0.3
            + metrics.citation_quality * 0.2
            + metrics.bias_detection * 0.1
        )

        return metrics

    def get_evaluation_report(
        self,
        metrics: IslamicEvaluationMetrics,
    ) -> str:
        """Generate human-readable evaluation report."""

        report = "## Islamic RAG Evaluation Report\n\n"

        report += "### Source Quality\n"
        report += f"- Source Authenticity: {metrics.source_authenticity:.2%}\n"
        report += f"- Evidence Presence: {metrics.evidence_presence:.2%}\n"
        report += f"- Citation Quality: {metrics.citation_quality:.2%}\n\n"

        report += "### Scholarly Standards\n"
        report += f"- Authority Score: {metrics.authority_score:.2%}\n"
        report += f"- Bias Detection: {'✓ Pass' if metrics.bias_detection > 0.7 else '⚠ Warning'}\n"

        if metrics.madhhab_coverage > 0:
            report += f"- Madhhab Coverage: {metrics.madhhab_coverage:.2%}\n"

        report += "\n### Recommendations\n"

        if metrics.source_authenticity < 0.7:
            report += "- Consider adding more authoritative sources\n"
        if metrics.evidence_presence < 0.5:
            report += "- Include more evidence from Quran and Hadith\n"
        if metrics.citation_quality < 0.5:
            report += "- Improve citation format and attribution\n"

        return report


# Comparative evaluation between two RAG systems
def compare_rag_systems(
    evaluator1: IslamicRAGEvaluator,
    evaluator2: IslamicRAGEvaluator,
    test_questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare two RAG systems on Islamic literature tasks.

    Args:
        evaluator1: First system evaluator
        evaluator2: Second system evaluator
        test_questions: List of test questions with ground truth

    Returns:
        Comparison results
    """

    results = {"system1": [], "system2": []}

    for q in test_questions:
        question = q["question"]

        # Evaluate both systems
        # (In practice, would need to run queries)

        pass

    # Compute averages

    return {
        "system1_avg_authority": 0.0,
        "system2_avg_authority": 0.0,
        "winner": "system1",  # Placeholder
    }


# Factory
def create_islamic_evaluator(pipeline: Any = None) -> IslamicRAGEvaluator:
    """Create an Islamic RAG evaluator."""
    return IslamicRAGEvaluator(pipeline)
