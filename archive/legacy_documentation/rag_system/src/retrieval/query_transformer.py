"""
Query Transformation Module - Production RAG 2026

Following RAG Pipeline Guide 2026 - Phase 4: Query Transformation

Features:
- Query rewriting (clarification, expansion)
- Query decomposition (multi-hop questions)
- Sub-question generation
- HyDE (Hypothetical Document Embedding)
- Step-back prompting (generalization for better retrieval)
- Arabic query optimization

Usage:
    transformer = QueryTransformer()
    rewritten = transformer.rewrite(query)
    sub_questions = transformer.decompose(query)
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


# ==================== Data Classes ====================


class QueryType(Enum):
    """Types of queries."""

    FACTUAL = "factual"  # Who, what, when, where
    HOW_TO = "how_to"  # How to, method
    COMPARISON = "comparison"  # Versus, compare
    DEFINITION = "definition"  # What is, define
    EXPLANATION = "explanation"  # Why, explain
    LIST = "list"  # List all, enumerate
    MULTI_HOP = "multi_hop"  # Requires multiple steps
    UNKNOWN = "unknown"


@dataclass
class TransformedQuery:
    """Result of query transformation."""

    original_query: str
    rewritten_query: str
    query_type: QueryType
    sub_queries: List[str] = field(default_factory=list)
    hypothetical_document: Optional[str] = None
    expanded_terms: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryTransformationConfig:
    """Configuration for query transformation."""

    # LLM settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    temperature: float = 0.3

    # Transformation settings
    enable_rewriting: bool = True
    enable_decomposition: bool = True
    enable_hyde: bool = True
    enable_step_back: bool = True
    enable_expansion: bool = True

    # Arabic-specific
    arabic_expansion: bool = True
    islamic_term_expansion: bool = True

    # Limits
    max_sub_queries: int = 5
    max_expanded_terms: int = 10


# ==================== Query Classifier ====================


class QueryClassifier:
    """
    Classify query type for appropriate transformation.

    Uses keyword matching + optional LLM for complex cases.
    """

    # Arabic question words
    ARABIC_QUESTION_WORDS = {
        "ما": QueryType.FACTUAL,
        "ماذا": QueryType.FACTUAL,
        "من": QueryType.FACTUAL,
        "متى": QueryType.FACTUAL,
        "أين": QueryType.FACTUAL,
        "كم": QueryType.FACTUAL,
        "كيف": QueryType.HOW_TO,
        "لماذا": QueryType.EXPLANATION,
        "هل": QueryType.FACTUAL,
    }

    # English question words
    ENGLISH_QUESTION_WORDS = {
        "who": QueryType.FACTUAL,
        "what": QueryType.FACTUAL,
        "when": QueryType.FACTUAL,
        "where": QueryType.FACTUAL,
        "how": QueryType.HOW_TO,
        "why": QueryType.EXPLANATION,
        "which": QueryType.FACTUAL,
    }

    # Comparison keywords
    COMPARISON_KEYWORDS = [
        "versus", "vs", "compare", "مقارنة", "أفضل", "افضل",
        "between", "difference", "فرق", "اختلاف",
    ]

    # Definition keywords
    DEFINITION_KEYWORDS = [
        "what is", "what are", "define", "definition",
        "ما هو", "ما هي", "تعريف", "شرح",
    ]

    # List keywords
    LIST_KEYWORDS = [
        "list", "enumerate", "all", "types",
        "قائمة", "جميع", "أنواع", "أقسام",
    ]

    def classify(self, query: str) -> QueryType:
        """Classify query type."""

        query_lower = query.lower()

        # Check for multi-hop indicators
        if self._is_multi_hop(query):
            return QueryType.MULTI_HOP

        # Check Arabic question words
        for word, qtype in self.ARABIC_QUESTION_WORDS.items():
            if query.startswith(word):
                return qtype

        # Check English question words
        for word, qtype in self.ENGLISH_QUESTION_WORDS.items():
            if query_lower.startswith(word):
                return qtype

        # Check comparison
        if any(kw in query_lower for kw in self.COMPARISON_KEYWORDS):
            return QueryType.COMPARISON

        # Check definition
        if any(kw in query_lower for kw in self.DEFINITION_KEYWORDS):
            return QueryType.DEFINITION

        # Check list
        if any(kw in query_lower for kw in self.LIST_KEYWORDS):
            return QueryType.LIST

        # Check how-to
        if any(kw in query_lower for kw in ["كيف", "طريقة", "كيفية", "how to"]):
            return QueryType.HOW_TO

        return QueryType.UNKNOWN

    def _is_multi_hop(self, query: str) -> bool:
        """Check if query requires multiple reasoning steps."""

        # Multi-hop indicators
        indicators = [
            "and then", "after that", "before",
            "ثم", "بعد", "قبل",
            "compare", "versus", "difference",
            "impact", "effect", "relationship",
            "تأثير", "علاقة", "سبب",
        ]

        query_lower = query.lower()

        # Count indicators
        count = sum(1 for ind in indicators if ind in query_lower)

        # Also check for multiple question words
        question_words = set(
            list(self.ARABIC_QUESTION_WORDS.keys()) +
            list(self.ENGLISH_QUESTION_WORDS.keys())
        )
        question_count = sum(1 for word in question_words if word in query_lower)

        return count >= 2 or question_count >= 2


# ==================== Query Rewriter ====================


class QueryRewriter:
    """
    Rewrite queries for better retrieval.

    Techniques:
    - Clarification (add context)
    - Expansion (add synonyms)
    - Simplification (remove noise)
    - Arabic-specific rewriting
    """

    def __init__(self, config: QueryTransformationConfig):
        self.config = config
        self._llm_client = None

        # Islamic term expansions
        self.islamic_terms = {
            "التوحيد": ["توحيد", "أركان التوحيد", "أنواع التوحيد", "الله"],
            "الصلاة": ["صلاة", "صلوات", "ركعات", "وضوء", "أركان الصلاة"],
            "الزكاة": ["زكاة", "صدقة", "مال", "فقراء", "مساكين"],
            "الصيام": ["صيام", "صوم", "رمضان", "إفطار", "سحور"],
            "الحج": ["حج", "عمرة", "مكة", "مشاعر", "طواف"],
            "القرآن": ["قرآن", "آية", "سورة", "وحي", "تنزيل"],
            "الحديث": ["حديث", "سنة", "نبوي", "إسناد", "متن"],
            "الفقه": ["فقه", "حكم", "شرع", "حلال", "حرام"],
        }

    def rewrite(self, query: str, query_type: QueryType) -> str:
        """Rewrite query for better retrieval."""

        rewritten = query

        # Apply simplification
        rewritten = self._simplify(rewritten)

        # Apply expansion based on type
        if query_type == QueryType.DEFINITION:
            rewritten = self._expand_definition(rewritten)
        elif query_type == QueryType.HOW_TO:
            rewritten = self._expand_howto(rewritten)
        elif query_type == QueryType.COMPARISON:
            rewritten = self._expand_comparison(rewritten)

        # Apply Islamic term expansion
        if self.config.islamic_term_expansion:
            rewritten = self._expand_islamic_terms(rewritten)

        return rewritten

    def _simplify(self, query: str) -> str:
        """Simplify query by removing noise words."""

        # Remove polite but unnecessary words
        noise_words = [
            "من فضلك", "لو سمحت", "أريد أن أعرف",
            "please", "i want to know", "can you tell me",
            "أرجوك", "هل يمكن",
        ]

        simplified = query
        for word in noise_words:
            simplified = simplified.replace(word, "")

        return simplified.strip()

    def _expand_definition(self, query: str) -> str:
        """Expand definition queries."""

        # Add definition-related terms
        expansions = [
            "تعريف", "معنى", "مفهوم",
            "definition", "meaning", "concept",
        ]

        return f"{query} {' '.join(expansions)}"

    def _expand_howto(self, query: str) -> str:
        """Expand how-to queries."""

        # Add method-related terms
        expansions = [
            "طريقة", "كيفية", "خطوات",
            "method", "steps", "procedure",
        ]

        return f"{query} {' '.join(expansions)}"

    def _expand_comparison(self, query: str) -> str:
        """Expand comparison queries."""

        # Add comparison-related terms
        expansions = [
            "مقارنة", "فرق", "اختلاف",
            "comparison", "difference", "versus",
        ]

        return f"{query} {' '.join(expansions)}"

    def _expand_islamic_terms(self, query: str) -> str:
        """Expand Islamic terminology."""

        expanded = query

        for term, synonyms in self.islamic_terms.items():
            if term in query:
                # Add synonyms
                expanded += " " + " ".join(synonyms[:5])
                break

        return expanded

    async def rewrite_with_llm(self, query: str) -> str:
        """Rewrite query using LLM for better understanding."""

        if not self._llm_client:
            return self.rewrite(query, QueryType.UNKNOWN)

        prompt = f"""Rewrite the following question to make it more specific and searchable.
Keep the same meaning but add relevant context and terms.

Original: {query}

Rewritten:"""

        response = await self._llm_client.generate(prompt)
        return response.answer.strip()


# ==================== Query Decomposer ====================


class QueryDecomposer:
    """
    Decompose complex queries into sub-questions.

    For multi-hop reasoning questions.
    """

    def __init__(self, config: QueryTransformationConfig):
        self.config = config
        self._llm_client = None

    def decompose(self, query: str) -> List[str]:
        """
        Decompose query into sub-questions.

        Uses rule-based approach with optional LLM enhancement.
        """

        # Check for conjunction-based decomposition
        if " و " in query or " and " in query:
            return self._decompose_conjunction(query)

        # Check for temporal decomposition
        if any(word in query for word in ["قبل", "بعد", "before", "after"]):
            return self._decompose_temporal(query)

        # Check for comparison decomposition
        if any(word in query for word in ["مقارنة", "versus", "compare", "فرق"]):
            return self._decompose_comparison(query)

        # Default: no decomposition needed
        return [query]

    def _decompose_conjunction(self, query: str) -> List[str]:
        """Decompose queries with conjunctions."""

        # Split by "and" / "و"
        parts = re.split(r'\s+و\s+|\s+and\s+', query, flags=re.IGNORECASE)

        if len(parts) < 2:
            return [query]

        # Convert each part to a question
        sub_queries = []
        for part in parts:
            part = part.strip()
            if part:
                # Try to make it a complete question
                if not self._is_complete_question(part):
                    part = self._make_question(part)
                sub_queries.append(part)

        return sub_queries[:self.config.max_sub_queries]

    def _decompose_temporal(self, query: str) -> List[str]:
        """Decompose temporal queries."""

        sub_queries = []

        # Extract temporal markers
        before_query = query
        after_query = query

        if "قبل" in query:
            # Split into before/after
            parts = query.split("قبل")
            if len(parts) == 2:
                sub_queries = [
                    f"ما حدث قبل {parts[1].strip()}",
                    f"ما حدث بعد {parts[1].strip()}",
                ]

        if "بعد" in query:
            parts = query.split("بعد")
            if len(parts) == 2:
                sub_queries = [
                    f"ما حدث قبل {parts[1].strip()}",
                    f"ما حدث بعد {parts[1].strip()}",
                ]

        return sub_queries if sub_queries else [query]

    def _decompose_comparison(self, query: str) -> List[str]:
        """Decompose comparison queries."""

        sub_queries = []

        # Find comparison targets
        comparison_words = ["مقارنة", "versus", "vs", "compare", "فرق", "بين"]

        for word in comparison_words:
            if word in query:
                parts = query.split(word)
                if len(parts) >= 2:
                    # Create individual queries for each target
                    for part in parts:
                        part = part.strip()
                        if part:
                            sub_queries.append(f"ما هو {part}")

                    break

        return sub_queries[:self.config.max_sub_queries] if sub_queries else [query]

    def _is_complete_question(self, text: str) -> bool:
        """Check if text is a complete question."""

        question_starts = [
            "ما", "ماذا", "من", "متى", "أين", "كيف", "لماذا", "هل",
            "who", "what", "when", "where", "how", "why",
        ]

        text_lower = text.lower().strip()
        return any(text_lower.startswith(q) for q in question_starts)

    def _make_question(self, text: str) -> str:
        """Convert text fragment to question."""

        text = text.strip()

        # Add question prefix
        if not self._is_complete_question(text):
            return f"ما هو {text}"

        return text

    async def decompose_with_llm(self, query: str) -> List[str]:
        """Decompose query using LLM."""

        if not self._llm_client:
            return self.decompose(query)

        prompt = f"""Decompose the following complex question into simpler sub-questions.
Each sub-question should be answerable independently.
Return only the sub-questions, one per line.

Question: {query}

Sub-questions:"""

        response = await self._llm_client.generate(prompt)

        # Parse response
        lines = response.answer.strip().split("\n")
        sub_queries = [line.strip() for line in lines if line.strip()]

        return sub_queries[:self.config.max_sub_queries]


# ==================== HyDE (Hypothetical Document Embedding) ====================


class HyDEGenerator:
    """
    Generate hypothetical document for query.

    HyDE technique:
    1. Generate a hypothetical answer document
    2. Embed the hypothetical document
    3. Use for similarity search

    This helps bridge the vocabulary gap between query and documents.
    """

    def __init__(self, config: QueryTransformationConfig):
        self.config = config
        self._llm_client = None

    async def generate(self, query: str, query_type: QueryType) -> str:
        """Generate hypothetical document for query."""

        if not self._llm_client:
            return self._generate_rule_based(query, query_type)

        # Build prompt based on query type
        if query_type == QueryType.DEFINITION:
            prompt = self._build_definition_prompt(query)
        elif query_type == QueryType.HOW_TO:
            prompt = self._build_howto_prompt(query)
        elif query_type == QueryType.FACTUAL:
            prompt = self._build_factual_prompt(query)
        else:
            prompt = self._build_general_prompt(query)

        response = await self._llm_client.generate(prompt)
        return response.answer.strip()

    def _generate_rule_based(self, query: str, query_type: QueryType) -> str:
        """Generate hypothetical document without LLM."""

        if query_type == QueryType.DEFINITION:
            return f"تعريف: {query}\n\nهذا المفهوم يشير إلى..."

        elif query_type == QueryType.HOW_TO:
            return f"طريقة {query}:\n\nالخطوة الأولى...\nالخطوة الثانية..."

        else:
            return f"إجابة على: {query}\n\nالمعلومات المتعلقة بهذا السؤال تشمل..."

    def _build_definition_prompt(self, query: str) -> str:
        """Build prompt for definition queries."""

        return f"""Generate a hypothetical encyclopedia entry that would answer this question.
Write in an informative, factual style. Include key terms and concepts.

Question: {query}

Encyclopedia Entry:"""

    def _build_howto_prompt(self, query: str) -> str:
        """Build prompt for how-to queries."""

        return f"""Generate a hypothetical how-to guide that would answer this question.
Write step-by-step instructions with clear explanations.

Question: {query}

How-to Guide:"""

    def _build_factual_prompt(self, query: str) -> str:
        """Build prompt for factual queries."""

        return f"""Generate a hypothetical document that contains the answer to this question.
Write in a factual, informative style with specific details.

Question: {query}

Document:"""

    def _build_general_prompt(self, query: str) -> str:
        """Build general prompt."""

        return f"""Generate a hypothetical document that would be relevant to this query.
Include key information and terminology that would appear in a good answer.

Question: {query}

Document:"""


# ==================== Step-Back Prompting ====================


class StepBackPrompter:
    """
    Generate more general query for better retrieval.

    Step-back prompting:
    1. Ask LLM to generate a more general concept
    2. Retrieve with general query
    3. Use results to answer specific question

    This helps when specific queries miss relevant context.
    """

    def __init__(self, config: QueryTransformationConfig):
        self.config = config
        self._llm_client = None

    async def step_back(self, query: str) -> str:
        """Generate step-back (more general) query."""

        if not self._llm_client:
            return self._step_back_rule_based(query)

        prompt = f"""What are the general concepts or principles related to this question?
Generate a broader query that would retrieve relevant background information.

Question: {query}

General concepts:"""

        response = await self._llm_client.generate(prompt)
        return response.answer.strip()

    def _step_back_rule_based(self, query: str) -> str:
        """Generate step-back query without LLM."""

        # Extract key terms
        key_terms = re.findall(r'[\u0600-\u06FFa-zA-Z]+', query)

        # Remove question words
        question_words = [
            "ما", "ماذا", "من", "متى", "أين", "كيف", "لماذا", "هل",
            "who", "what", "when", "where", "how", "why",
        ]

        general_terms = [t for t in key_terms if t.lower() not in question_words]

        return " ".join(general_terms[:5])


# ==================== Main Query Transformer ====================


class QueryTransformer:
    """
    Complete query transformation pipeline.

    Combines all transformation techniques:
    - Classification
    - Rewriting
    - Decomposition
    - HyDE
    - Step-back
    """

    def __init__(self, config: Optional[QueryTransformationConfig] = None):
        self.config = config or QueryTransformationConfig()

        # Initialize components
        self.classifier = QueryClassifier()
        self.rewriter = QueryRewriter(self.config)
        self.decomposer = QueryDecomposer(self.config)
        self.hyde = HyDEGenerator(self.config)
        self.step_back = StepBackPrompter(self.config)

    async def transform(
        self,
        query: str,
        enable_all: bool = True,
    ) -> TransformedQuery:
        """
        Apply all transformation techniques to query.

        Args:
            query: Original query
            enable_all: Enable all transformations

        Returns:
            TransformedQuery with all transformations
        """

        # Step 1: Classify
        query_type = self.classifier.classify(query)

        # Step 2: Rewrite
        rewritten_query = query
        if self.config.enable_rewriting:
            rewritten_query = self.rewriter.rewrite(query, query_type)

        # Step 3: Decompose
        sub_queries = [query]
        if (
            self.config.enable_decomposition
            and query_type == QueryType.MULTI_HOP
        ):
            sub_queries = self.decomposer.decompose(query)

        # Step 4: Generate hypothetical document
        hypothetical_doc = None
        if self.config.enable_hyde:
            hypothetical_doc = await self.hyde.generate(query, query_type)

        # Step 5: Step-back query
        step_back_query = None
        if self.config.enable_step_back:
            step_back_query = await self.step_back.step_back(query)

        # Step 6: Term expansion
        expanded_terms = []
        if self.config.enable_expansion:
            expanded_terms = self._expand_terms(query, query_type)

        return TransformedQuery(
            original_query=query,
            rewritten_query=rewritten_query,
            query_type=query_type,
            sub_queries=sub_queries,
            hypothetical_document=hypothetical_doc,
            expanded_terms=expanded_terms,
            confidence=1.0,
            metadata={
                "step_back_query": step_back_query,
            },
        )

    def _expand_terms(self, query: str, query_type: QueryType) -> List[str]:
        """Expand query with related terms."""

        expanded = []

        # Add Arabic synonyms
        if self.config.arabic_expansion:
            expanded.extend(self._get_arabic_synonyms(query))

        # Add Islamic terms
        if self.config.islamic_term_expansion:
            expanded.extend(self._get_islamic_synonyms(query))

        return expanded[:self.config.max_expanded_terms]

    def _get_arabic_synonyms(self, query: str) -> List[str]:
        """Get Arabic synonyms for query terms."""

        synonyms = {
            "الله": ["رب", "خالق", "بارئ"],
            "النبي": ["رسول", "محمد", "أحمد"],
            "الإسلام": ["مسلم", "دين"],
            "القرآن": ["كتاب", "وحي", "تنزيل"],
            "الحديث": ["سنة", "أثر"],
        }

        results = []
        for term, syns in synonyms.items():
            if term in query:
                results.extend(syns)

        return results

    def _get_islamic_synonyms(self, query: str) -> List[str]:
        """Get Islamic terminology synonyms."""

        return self.rewriter._expand_islamic_terms(query).split()


# ==================== Factory Functions ====================


def create_query_transformer(
    enable_hyde: bool = True,
    enable_decomposition: bool = True,
    enable_step_back: bool = True,
    **kwargs,
) -> QueryTransformer:
    """
    Create query transformer.

    Args:
        enable_hyde: Enable HyDE generation
        enable_decomposition: Enable query decomposition
        enable_step_back: Enable step-back prompting
        **kwargs: Additional config options

    Returns:
        QueryTransformer instance
    """

    config = QueryTransformationConfig(
        enable_hyde=enable_hyde,
        enable_decomposition=enable_decomposition,
        enable_step_back=enable_step_back,
        **kwargs,
    )

    return QueryTransformer(config)


if __name__ == "__main__":
    import asyncio

    async def main():
        """Demo query transformation."""

        print("Query Transformer - Demo")
        print("=" * 50)

        transformer = create_query_transformer()

        # Test queries
        queries = [
            "ما هو التوحيد في الإسلام؟",
            "Compare the four madhhabs on prayer",
            "What is the ruling on zakat and how is it calculated?",
        ]

        for query in queries:
            print(f"\nOriginal: {query}")
            print("-" * 40)

            result = await transformer.transform(query)

            print(f"Type: {result.query_type.value}")
            print(f"Rewritten: {result.rewritten_query}")
            print(f"Sub-queries: {result.sub_queries}")
            print(f"Expanded terms: {result.expanded_terms[:5]}")

    asyncio.run(main())
