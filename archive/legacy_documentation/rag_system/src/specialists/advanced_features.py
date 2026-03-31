"""
Advanced RAG Features for Islamic Literature

Features:
1. Multi-hop Reasoning - Complex theological questions requiring multiple sources
2. Cross-Reference System - Connect related concepts across books
3. Timeline Reconstruction - Historical events and scholars
4. Authority Ranking - Prioritize authoritative sources
5. Progressive Retrieval - Iterative context refinement
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class Scholar:
    """Represents an Islamic scholar."""

    id: int
    name: str
    death_year: int
    madhhab: Optional[str] = None
    specialty: List[str] = field(default_factory=list)
    works: List[str] = field(default_factory=list)
    authority_score: float = 0.0


@dataclass
class Book:
    """Represents a book in the corpus."""

    id: int
    title: str
    author_id: int
    category: str
    date: int  # Composition date
    authority_score: float = 0.0
    cross_references: List[str] = field(default_factory=list)


# Authority ranking for Islamic sources
AUTHORITY_RANKINGS = {
    # Quran (always highest)
    "القرآن": 100.0,
    # Hadith Collections
    "صحيح البخاري": 95.0,
    "صحيح مسلم": 94.0,
    "سنن الترمذي": 92.0,
    "سنن أبي داود": 91.0,
    "سنن النسائي": 90.0,
    "سنن ابن ماجه": 89.0,
    "الموطأ": 88.0,
    "مسند أحمد": 87.0,
    # Major Tafsirs
    "تفسير ابن كثير": 85.0,
    "تفسير القرطبي": 84.0,
    "تفسير الطبري": 83.0,
    "تفسير السعدي": 82.0,
    # Fiqh - Four Madhhabs
    # Hanafi
    "الهداية": 80.0,
    "الفتاوى الهندية": 79.0,
    "بدائع الصنائع": 78.0,
    # Maliki
    "المدونة": 80.0,
    "موطأ مالك": 80.0,
    # Shafi'i
    "المهذب": 80.0,
    "الوسيط": 79.0,
    # Hanbali
    "المبدع": 80.0,
    "كشاف القناع": 79.0,
    # Aqeedah
    "العقيدة الواسطية": 85.0,
    "التوحيد": 85.0,
    "الشرح الممتع": 84.0,
    # Early Scholars
    "ابن تيمية": 90.0,
    "ابن القيم": 89.0,
    "الإمام أحمد": 88.0,
    "الإمام الشافعي": 88.0,
    "الإمام مالك": 88.0,
    "الإمام أبو حنيفة": 88.0,
}


class AuthorityRanker:
    """
    Rank sources by Islamic scholarly authority.

    Considerations:
    - Authenticity of hadith
    - Age of the source
    - Scholar's reputation
    - Acceptance across Muslim community
    """

    def __init__(self):
        self.authority_cache = {}

    def get_authority_score(
        self,
        book_title: str,
        author: str = "",
        category: str = "",
    ) -> float:
        """Get authority score for a source."""

        # Check cache
        cache_key = f"{book_title}:{author}"
        if cache_key in self.authority_cache:
            return self.authority_cache[cache_key]

        score = 0.0

        # Check exact match
        for known_source, known_score in AUTHORITY_RANKINGS.items():
            if known_source in book_title or known_source in author:
                score = max(score, known_score)

        # Category-based scoring
        if score == 0:
            category_weights = {
                "كتب السنة": 80.0,
                "التفسير": 75.0,
                "العقيدة": 75.0,
                "الفقه العام": 70.0,
                "شروح الحديث": 65.0,
                "الرقائق والآداب والأذكار": 60.0,
                "التاريخ": 50.0,
                "الأدب": 40.0,
            }
            score = category_weights.get(category, 50.0)

        # Cache result
        self.authority_cache[cache_key] = score

        return score

    def rerank_by_authority(
        self,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank sources by authority."""

        for source in sources:
            authority = self.get_authority_score(
                source.get("book_title", ""),
                source.get("author", ""),
                source.get("category", ""),
            )
            # Combine relevance with authority
            original_score = source.get("score", 0)
            source["combined_score"] = original_score * 0.6 + authority / 100.0 * 0.4

        # Sort by combined score
        return sorted(sources, key=lambda x: x.get("combined_score", 0), reverse=True)


class CrossReferenceSystem:
    """
    System for finding cross-references between concepts in Islamic literature.

    Features:
    - Track concept relationships
    - Find related concepts
    - Build knowledge graph
    """

    def __init__(self):
        self.concepts: Dict[str, Set[str]] = defaultdict(set)
        self.relationships: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.concept_books: Dict[str, List[str]] = defaultdict(list)

    def index_concepts(
        self,
        concepts: List[Dict[str, Any]],
    ):
        """Index concepts from processed documents."""

        for item in concepts:
            concept = item.get("concept", "")
            related = item.get("related_concepts", [])
            book_title = item.get("book_title", "")

            if concept:
                self.concepts[concept].update(related)
                # Fixed: concept_cepts -> concept_books (typo)
                self.concept_books[concept].append(book_title)

                # Track relationships
                for rel in related:
                    self.relationships[concept].append((rel, "related"))

    def find_related(
        self,
        concept: str,
        depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """Find related concepts up to a certain depth."""

        visited = set()
        results = []

        def search(current: str, current_depth: int):
            if current_depth > depth or current in visited:
                return

            visited.add(current)

            # Get direct relationships
            for rel_type, related in self.relationships.get(current, []):
                results.append(
                    {
                        "concept": related,
                        "relationship": rel_type,
                        "distance": current_depth,
                        "from_concept": current,
                    }
                )

                # Recurse
                search(related, current_depth + 1)

        search(concept, 1)

        return results

    def find_common_books(
        self,
        concepts: List[str],
    ) -> Dict[str, int]:
        """Find books that contain multiple concepts."""

        book_counts = defaultdict(int)

        for concept in concepts:
            for book in self.concept_books.get(concept, []):
                book_counts[book] += 1

        # Sort by count
        return dict(sorted(book_counts.items(), key=lambda x: x[1], reverse=True))


class MultiHopReasoning:
    """
    Multi-hop reasoning for complex Islamic questions.

    Example:
    Q: "What is the evidence for intermittent fasting in Ramadan?"
    Hop 1: Find hadith about Ramadan
    Hop 2: Find rulings about fasting
    Hop 3: Find specific evidence for intermittent fasting
    """

    def __init__(
        self,
        pipeline: Any,
        max_hops: int = 3,
    ):
        self.pipeline = pipeline
        self.max_hops = max_hops

    async def reason(
        self,
        question: str,
        context: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform multi-hop reasoning.

        Args:
            question: The complex question
            context: Previous context from earlier hops

        Returns:
            Reasoning chain and final answer
        """

        reasoning_chain = []
        current_context = context or []

        for hop in range(self.max_hops):
            # Refine question based on context
            refined_question = self._refine_question(
                question,
                current_context,
                hop,
            )

            # Retrieve
            result = await self.pipeline.query(refined_question, top_k=3)

            # Store result
            reasoning_chain.append(
                {
                    "hop": hop + 1,
                    "question": refined_question,
                    "answer": result.answer,
                    "sources": result.sources[:2],  # Top 2
                }
            )

            # Update context
            new_context = [s.get("content_preview", "") for s in result.sources[:2]]
            current_context.extend(new_context)

            # Check if we have enough information
            if self._has_sufficient_info(result, hop):
                break

        # Generate final answer
        final_answer = self._generate_answer(question, reasoning_chain)

        return {
            "question": question,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer,
            "total_hops": len(reasoning_chain),
        }

    def _refine_question(
        self,
        original: str,
        context: List[str],
        hop: int,
    ) -> str:
        """Refine question based on previous context."""

        if hop == 0:
            return original

        # Add context to question
        context_text = " | ".join(context[-2:])

        refined = f"""
Previous context: {context_text}

Building on above, answer: {original}

Focus on detailed explanation with evidence.
"""

        return refined.strip()

    def _has_sufficient_info(
        self,
        result: Any,
        hop: int,
    ) -> bool:
        """Check if we have sufficient information."""

        # If we have sources and this isn't the first hop
        if result.sources and hop > 0:
            # Check if sources have relevance scores above threshold
            top_score = result.sources[0].get("score", 0)
            if top_score > 0.7:
                return True

        return False

    def _generate_answer(
        self,
        question: str,
        chain: List[Dict[str, Any]],
    ) -> str:
        """Generate final answer from reasoning chain."""

        if len(chain) == 1:
            return chain[0].get("answer", "")

        # Build multi-hop answer
        answer = "## Analysis\n\n"

        for hop_result in chain:
            answer += f"### Step {hop_result['hop']}\n"
            answer += f"{hop_result.get('answer', '')[:200]}...\n\n"

        answer += "---\n\n## Conclusion\n"
        answer += chain[-1].get("answer", "")

        return answer


class TimelineReconstructor:
    """
    Reconstruct timelines from Islamic historical texts.

    Features:
    - Extract dates and events
    - Build scholar timelines
    - Identify historical context
    """

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.scholar_activities: Dict[str, List[Dict]] = defaultdict(list)

    def extract_events(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract historical events from documents."""

        import re

        # Date patterns in Arabic
        date_patterns = [
            r"(\d+)هـ",  # Hijri year
            r"(\d+)/(\d+)/(\d+)م",  # Gregorian
            r"سنة (\d+)",  # Year
            r"عام (\d+)",  # Year
        ]

        for doc in documents:
            content = doc.get("content", "")
            title = doc.get("book_title", "")

            # Simple extraction (would need NLP in production)
            # This is a simplified version

            # Add to events
            self.events.append(
                {
                    "title": title,
                    "content": content[:500],
                    "date_source": "document_date",  # Could extract from metadata
                }
            )

        return self.events

    def get_scholar_timeline(
        self,
        scholar_name: str,
    ) -> List[Dict[str, Any]]:
        """Get timeline of a scholar's life and works."""

        timeline = []

        # Find events mentioning scholar
        for event in self.events:
            if scholar_name in event.get("content", ""):
                timeline.append(event)

        # Sort by date
        timeline.sort(key=lambda x: x.get("date", 0))

        return timeline

    def get_events_by_period(
        self,
        start_year: int,
        end_year: int,
    ) -> List[Dict[str, Any]]:
        """Get events from a specific period."""

        return [e for e in self.events if start_year <= e.get("date", 0) <= end_year]


class ProgressiveRetrieval:
    """
    Progressive context retrieval for complex questions.

    Strategy:
    1. Start with broad retrieval
    2. Analyze gaps in context
    3. Retrieve additional context
    4. Repeat until complete
    """

    def __init__(
        self,
        pipeline: Any,
        max_iterations: int = 3,
    ):
        self.pipeline = pipeline
        self.max_iterations = max_iterations

    async def retrieve_progressive(
        self,
        question: str,
        target_context_size: int = 2000,
    ) -> Dict[str, Any]:
        """Progressively retrieve context until target size."""

        all_sources = []
        current_context = ""
        iterations = 0

        while (
            len(current_context) < target_context_size
            and iterations < self.max_iterations
        ):
            # Determine what's missing
            missing_info = self._identify_gaps(question, current_context)

            if not missing_info:
                break

            # Retrieve for missing info
            retrieval_query = self._build_retrieval_query(question, missing_info)
            result = await self.pipeline.query(retrieval_query, top_k=3)

            # Add new sources
            all_sources.extend(result.sources)
            current_context += "\n\n" + "\n\n".join(
                [s.get("content_preview", "") for s in result.sources]
            )

            iterations += 1

        return {
            "question": question,
            "context": current_context,
            "sources": all_sources,
            "iterations": iterations,
        }

    def _identify_gaps(
        self,
        question: str,
        current_context: str,
    ) -> List[str]:
        """Identify what's missing in current context."""

        gaps = []

        # Check for key elements
        if "حكم" in question or "يجوز" in question:
            if "حكم" not in current_context:
                gaps.append("ruling")

        if "دليل" in question or "evidence" in question.lower():
            if "دليل" not in current_context and "آية" not in current_context:
                gaps.append("evidence")

        if "مذهب" in question:
            if "مذهب" not in current_context:
                gaps.append("madhhab_opinion")

        return gaps

    def _build_retrieval_query(
        self,
        original: str,
        gaps: List[str],
    ) -> str:
        """Build retrieval query based on gaps."""

        queries = []

        for gap in gaps:
            if gap == "ruling":
                queries.append("حكم الفقهاء وفتاوى")
            elif gap == "evidence":
                queries.append("أدلة من القرآن والسنة")
            elif gap == "madhhab_opinion":
                queries.append("آراء المذاهب الأربعة")

        # Combine with original
        combined = f"{original} | {' '.join(queries)}"

        return combined


class ConceptExtractor:
    """
    Extract key Islamic concepts from text.

    Concepts:
    - Aqeedah (theological concepts)
    - Fiqh rulings
    - Hadith terminology
    - Arabic linguistic terms
    """

    # Islamic concept dictionaries
    CONCEPTS = {
        # Aqeedah
        "التوحيد": {"category": "aqeedah", "arabic_def": "إفراد الله بالعبادة"},
        "الصفات": {"category": "aqeedah", "arabic_def": "صفات الله تعالى"},
        "القدر": {"category": "aqeedah", "arabic_def": "تقدير الله للأحداث"},
        "الإيمان": {
            "category": "aqeedah",
            "arabic_def": "تصديق بالقلب وقول باللسان وعمل",
        },
        # Fiqh
        "الوضوء": {"category": "fiqh", "arabic_def": "طهارة مائية"},
        "الصلاة": {"category": "fiqh", "arabic_def": "عبادة ذات أركان"},
        "الزكاة": {"category": "fiqh", "arabic_def": "إخراج المال المحوصب"},
        "الصيام": {"category": "fiqh", "arabic_def": "الإمساك عن المفطرات"},
        "الحج": {"category": "fiqh", "arabic_def": "زيارة البيت الحرام"},
        # Hadith
        "صحيح": {"category": "hadith", "arabic_def": "الحديث المثبت"},
        "ضعيف": {"category": "hadith", "arabic_def": "الحديث الناقص"},
        "مرسل": {"category": "hadith", "arabic_def": "التابعى without تابعي"},
        "موقوف": {"category": "hadith", "arabic_def": "كلام الصحابي"},
        # Arabic
        "إعراب": {"category": "language", "arabic_def": "توقع الكلمة في الجملة"},
        "بناء": {"category": "language", "arabic_def": "ت固定 الكلمة"},
        "اشتقاق": {"category": "language", "arabic_def": "أخذ كلمة من أخرى"},
    }

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts from text."""

        found_concepts = []

        for concept, info in self.CONCEPTS.items():
            if concept in text:
                found_concepts.append(
                    {
                        "concept": concept,
                        "category": info["category"],
                        "definition": info["arabic_def"],
                    }
                )

        return found_concepts

    def categorize_by_importance(
        self,
        concepts: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize concepts by importance."""

        categorized = defaultdict(list)

        for concept in concepts:
            categorized[concept["category"]].append(concept)

        return dict(categorized)


# Factory functions
def create_authority_ranker() -> AuthorityRanker:
    """Create an authority ranker."""
    return AuthorityRanker()


def create_cross_reference_system() -> CrossReferenceSystem:
    """Create a cross-reference system."""
    return CrossReferenceSystem()


def create_multi_hop_reasoning(pipeline: Any) -> MultiHopReasoning:
    """Create a multi-hop reasoning system."""
    return MultiHopReasoning(pipeline)


def create_timeline_reconstructor() -> TimelineReconstructor:
    """Create a timeline reconstructor."""
    return TimelineReconstructor()


def create_progressive_retrieval(pipeline: Any) -> ProgressiveRetrieval:
    """Create a progressive retrieval system."""
    return ProgressiveRetrieval(pipeline)


def create_concept_extractor() -> ConceptExtractor:
    """Create a concept extractor."""
    return ConceptExtractor()
