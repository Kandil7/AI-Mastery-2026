"""
Search Enhancement Services
=========================
Services for auto-suggest and faceted search.

خدمات تحسين البحث
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

log = logging.getLogger(__name__)


class FacetType(str, Enum):
    """Types of search facets.

    أنواع جوانب البحث
    """

    STATUS = "status"
    CONTENT_TYPE = "content_type"
    DATE_RANGE = "date_range"
    SIZE_RANGE = "size_range"


@dataclass
class SearchFacet:
    """A search facet with count.

    جانب البحث مع عدد
    """

    facet_type: FacetType
    name: str
    count: int
    value: Any | None = None


@dataclass
class SearchSuggestion:
    """Auto-suggest result.

    نتيجة الاقتراح التلقائي
    """

    text: str
    type: str  # document, query, topic
    relevance_score: float


@dataclass
class AutoSuggestRequest:
    """Request for auto-suggest.

    طلب للاقتراح التلقائي
    """

    query: str
    limit: int = 5
    types: List[str] = None  # document, query, topic


class QueryExpansionService:
    """
    Expand queries with related terms and synonyms.

    خدمة توسيع الاستعلامات
    """

    def __init__(self, llm_port, document_repo):
        """
        Initialize query expansion service.

        Args:
            llm_port: LLM adapter for expansion
            document_repo: Document repository for context
        """
        self._llm = llm_port
        self._repo = document_repo

    def expand_query(
        self,
        query: str,
        tenant_id: str,
        num_expansions: int = 3,
    ) -> List[str]:
        """
        Expand query with related terms.

        Args:
            query: Original query
            tenant_id: Tenant ID for context
            num_expansions: Number of expansions to generate

        Returns:
            List of expanded queries

        توسيع الاستعلام بمصطلحات ذات صلة
        """
        if not query or not query.strip():
            return [query]

        # Get top 5 similar documents for context
        similar_docs = self._repo.find_similar_docs(
            tenant_id=tenant_id,
            query=query,
            k=5,
        )

        # Build context from document filenames
        context = "\n".join([d.filename for d in similar_docs])

        # Build prompt for query expansion
        prompt = f"""You are an expert query expansion assistant. 

Original query: {query}

Context from similar documents:
{context}

Task: Generate {num_expansions} alternative search queries that would retrieve relevant documents. The queries should:
1. Be semantically similar but different wording
2. Use synonyms and related terms
3. Cover different aspects of the topic
4. Be concise (2-5 words each)

Return ONLY the {num_expansions} queries, one per line:"""

        try:
            response = self._llm.generate(prompt, temperature=0.7)

            # Parse response
            expansions = [q.strip() for q in response.split("\n") if q.strip()][:num_expansions]

            if not expansions:
                log.warning("Query expansion returned no results", query=query)
                return [query]

            log.info("Query expanded", query=query, expansions=expansions)
            return [query] + expansions

        except Exception as e:
            log.error("Failed to expand query", query=query, error=str(e))
            return [query]

    async def expand_query_async(
        self,
        query: str,
        tenant_id: str,
        num_expansions: int = 3,
    ) -> List[str]:
        """Async version of expand_query()."""
        import asyncio

        return await asyncio.run(self.expand_query(query, tenant_id, num_expansions))


class FacetedSearchService:
    """
    Compute search facets for document search results.

    خدمة البحث المجزوء
    """

    def __init__(self):
        """
        Initialize faceted search service.

        This service is stateless - all configuration is provided
        at method call time. No initialization required.
        """
        # Intentional no-op: service doesn't require state initialization
        pass

    def compute_facets(
        self,
        documents: List[Dict[str, Any]],
        requested_facets: List[FacetType] | None = None,
    ) -> List[SearchFacet]:
        """
        Compute facets for search results.

        Args:
            documents: List of documents from search
            requested_facets: Facets to compute (default: all)

        Returns:
            List of search facets

        حساب جوانب البحث
        """
        if not documents:
            return []

        facets: List[SearchFacet] = []

        # Status facet
        if requested_facets is None or FacetType.STATUS in requested_facets:
            status_counts = self._count_by_field(documents, "status")
            for status, count in status_counts.items():
                facets.append(
                    SearchFacet(
                        facet_type=FacetType.STATUS,
                        name=status,
                        count=count,
                    )
                )

        # Content type facet
        if requested_facets is None or FacetType.CONTENT_TYPE in requested_facets:
            content_counts = self._count_by_field(documents, "content_type")
            for content_type, count in content_counts.items():
                facets.append(
                    SearchFacet(
                        facet_type=FacetType.CONTENT_TYPE,
                        name=content_type,
                        count=count,
                    )
                )

        # Size range facet
        if requested_facets is None or FacetType.SIZE_RANGE in requested_facets:
            size_facets = self._compute_size_ranges(documents)
            facets.extend(size_facets)

        # Date range facet
        if requested_facets is None or FacetType.DATE_RANGE in requested_facets:
            date_facets = self._compute_date_ranges(documents)
            facets.extend(date_facets)

        return facets

    def _count_by_field(
        self,
        documents: List[Dict[str, Any]],
        field: str,
    ) -> Dict[str, int]:
        """Count documents by field value."""
        counts = {}
        for doc in documents:
            value = doc.get(field, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

    def _compute_size_ranges(self, documents: List[Dict[str, Any]]) -> List[SearchFacet]:
        """Compute size range facets."""
        sizes = [d.get("size_bytes", 0) for d in documents]

        # Define ranges (in KB)
        ranges = [
            ("0-100KB", 0, 100 * 1024),
            ("100KB-1MB", 100 * 1024, 1024 * 1024),
            ("1MB-10MB", 1024 * 1024, 10 * 1024 * 1024),
            ("10MB+", 10 * 1024 * 1024, float("inf")),
        ]

        facets = []
        for name, min_size, max_size in ranges:
            count = sum(1 for s in sizes if min_size <= s < max_size)
            facets.append(
                SearchFacet(
                    facet_type=FacetType.SIZE_RANGE,
                    name=name,
                    count=count,
                )
            )

        return facets

    def _compute_date_ranges(self, documents: List[Dict[str, Any]]) -> List[SearchFacet]:
        """Compute date range facets."""
        from datetime import datetime, timedelta

        dates = [d.get("created_at") for d in documents if d.get("created_at")]

        if not dates:
            return []

        # Define ranges (last 7 days, last 30 days, older)
        now = datetime.now()
        ranges = [
            ("Last 7 days", now - timedelta(days=7), now),
            ("Last 30 days", now - timedelta(days=30), now),
            ("Older than 30 days", None, now - timedelta(days=30)),
        ]

        facets = []
        for name, min_date, max_date in ranges:
            if min_date:
                count = sum(1 for d in dates if min_date <= d < max_date)
            else:
                count = sum(1 for d in dates if d < max_date)

            facets.append(
                SearchFacet(
                    facet_type=FacetType.DATE_RANGE,
                    name=name,
                    count=count,
                    value={"start": min_date.isoformat() if min_date else None},
                )
            )

        return facets


class AutoSuggestService:
    """
    Provide auto-suggest for search queries.

    خدمة الاقتراح التلقائي للاستعلامات
    """

    def __init__(self, llm_port, document_repo):
        """
        Initialize auto-suggest service.

        Args:
            llm_port: LLM adapter
            document_repo: Document repository
        """
        self._llm = llm_port
        self._repo = document_repo

    def get_suggestions(
        self,
        request: AutoSuggestRequest,
        tenant_id: str,
    ) -> List[SearchSuggestion]:
        """
        Get search suggestions.

        Args:
            request: Auto-suggest request
            tenant_id: Tenant ID for filtering

        Returns:
            List of suggestions with relevance scores

        الحصول على اقتراحات البحث
        """
        suggestions = []

        if not request.query or not request.query.strip():
            return suggestions

        query_lower = request.query.lower()

        # 1. Query suggestions (query expansion)
        if not request.types or "query" in request.types:
            expanded = self._expand_query_suggestions(request.query, limit=request.limit // 2)
            suggestions.extend(expanded)

        # 2. Document name suggestions
        if not request.types or "document" in request.types:
            docs = self._repo.search_by_prefix(
                tenant_id=tenant_id,
                prefix=query_lower,
                limit=request.limit // 2,
            )
            for i, doc in enumerate(docs):
                relevance = 1.0 - (i * 0.1)  # Decrease relevance for lower matches
                suggestions.append(
                    SearchSuggestion(
                        text=doc["filename"],
                        type="document",
                        relevance_score=relevance,
                    )
                )

        # 3. Topic suggestions (LLM-generated)
        if not request.types or "topic" in request.types:
            topics = self._generate_topic_suggestions(request.query, limit=request.limit // 2)
            suggestions.extend(topics)

        # Sort by relevance and limit
        suggestions.sort(key=lambda s: s.relevance_score, reverse=True)

        return suggestions[: request.limit]

    def _expand_query_suggestions(
        self,
        query: str,
        limit: int,
    ) -> List[SearchSuggestion]:
        """Generate query expansion suggestions."""
        # Use simple expansion logic
        # In production, use QueryExpansionService

        # Placeholder expansions
        return [
            SearchSuggestion(text=f"{query} tutorial", type="query", relevance_score=0.9),
            SearchSuggestion(text=f"how to {query}", type="query", relevance_score=0.8),
        ]

    def _generate_topic_suggestions(
        self,
        query: str,
        limit: int,
    ) -> List[SearchSuggestion]:
        """Generate topic suggestions using LLM."""
        try:
            prompt = f"""Generate {limit} topic suggestions related to: {query}

Each suggestion should:
- Be a concise topic name (2-4 words)
- Be semantically relevant
- Be a potential search topic

Return ONLY {limit} topics, one per line:"""

            response = self._llm.generate(prompt, temperature=0.7)
            topics = [t.strip() for t in response.split("\n") if t.strip()][:limit]

            return [SearchSuggestion(text=t, type="topic", relevance_score=0.7) for t in topics]

        except Exception as e:
            log.error("Failed to generate topic suggestions", query=query, error=str(e))
            return []


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def build_facet_response(
    query: str,
    documents: List[Dict[str, Any]],
    facets: List[SearchFacet],
) -> Dict[str, Any]:
    """
    Build search response with facets.

    بناء استجابة البحث مع الجوانب
    """
    return {
        "query": query,
        "results": documents,
        "facets": {
            "status": [f for f in facets if f.facet_type == FacetType.STATUS],
            "content_type": [f for f in facets if f.facet_type == FacetType.CONTENT_TYPE],
            "size_ranges": [f for f in facets if f.facet_type == FacetType.SIZE_RANGE],
            "date_ranges": [f for f in facets if f.facet_type == FacetType.DATE_RANGE],
        },
        "total": len(documents),
    }


if __name__ == "__main__":
    from unittest.mock import Mock

    # Test search enhancements
    llm_mock = Mock()
    repo_mock = Mock()

    # Query expansion
    expansion_service = QueryExpansionService(llm_mock, repo_mock)
    expansions = expansion_service.expand_query("RAG", "tenant-123", num_expansions=3)
    print(f"Expanded queries: {expansions}")

    # Faceted search
    faceted_service = FacetedSearchService()
    sample_docs = [
        {
            "status": "indexed",
            "content_type": "application/pdf",
            "size_bytes": 1024 * 100,
            "created_at": "2024-01-01",
        },
        {
            "status": "indexed",
            "content_type": "text/plain",
            "size_bytes": 1024 * 500,
            "created_at": "2024-01-15",
        },
        {
            "status": "failed",
            "content_type": "application/pdf",
            "size_bytes": 1024 * 2000,
            "created_at": "2024-01-10",
        },
    ]
    facets = faceted_service.compute_facets(sample_docs)
    print(f"Facets: {len(facets)} facets")

    # Auto-suggest
    suggest_service = AutoSuggestService(llm_mock, repo_mock)
    request = AutoSuggestRequest(query="vector", limit=5)
    suggestions = suggest_service.get_suggestions(request, "tenant-123")
    print(f"Suggestions: {len(suggestions)} suggestions")
    for s in suggestions:
        print(f"  - {s.text} ({s.type}, relevance: {s.relevance_score})")
