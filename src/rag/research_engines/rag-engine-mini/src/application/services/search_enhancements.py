"""
Search Enhancement Services
=========================
Services for auto-suggest and faceted search.

خدمات تحسين البحث
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging
from collections import defaultdict
from datetime import datetime, timedelta

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
        self._query_expansion_service = QueryExpansionService(llm_port, document_repo)
        self._doc_trie = Trie()
        self._topic_cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_trie_build = None

    def _build_document_trie(self, tenant_id: str) -> None:
        """
        Build trie from document names for fast autocomplete.

        Build trie periodically or when cache expires.

        بناء trie من أسماء المستندات للإكمال التلقائي السريع
        """
        now = datetime.now()
        if self._last_trie_build:
            elapsed = (now - self._last_trie_build).total_seconds()
            if elapsed < self._cache_ttl:
                return

        try:
            docs = self._repo.list_documents(
                tenant_id=tenant_id,
                limit=500,  # Limit to recent 500 docs for performance
            )

            # Clear and rebuild trie
            self._doc_trie = Trie()
            for doc in docs:
                filename = doc.filename
                score = 1.0
                # Boost score for recently accessed docs
                if hasattr(doc, "updated_at") and doc.updated_at:
                    days_since_update = (now - doc.updated_at).total_seconds() / 86400
                    score = max(0.1, 1.0 - (days_since_update / 365))

                self._doc_trie.insert(filename, score)

            self._last_trie_build = now
            log.info(
                "Document trie built", doc_count=len(docs), trie_size=self._doc_trie.get_size()
            )
        except Exception as e:
            log.error("Failed to build document trie", error=str(e))

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

        # Build trie if needed
        self._build_document_trie(tenant_id)

        # 1. Query suggestions (query expansion)
        if not request.types or "query" in request.types:
            expanded = self._query_expansion_service.expand_query(
                query=request.query,
                tenant_id=tenant_id,
                num_expansions=request.limit // 2,
            )
            # Convert to suggestions with higher relevance
            for exp_query in expanded[1:]:  # Skip first (original query)
                suggestions.append(
                    SearchSuggestion(
                        text=exp_query,
                        type="query",
                        relevance_score=0.85,
                    )
                )

        # 2. Document name suggestions (using trie for faster autocomplete)
        if not request.types or "document" in request.types:
            trie_matches = self._doc_trie.autocomplete(
                prefix=query_lower,
                limit=request.limit,
                min_score=0.1,
            )
            for doc_name, score in trie_matches:
                suggestions.append(
                    SearchSuggestion(
                        text=doc_name,
                        type="document",
                        relevance_score=score,
                    )
                )

        # 3. Topic suggestions (LLM-generated, with caching)
        if not request.types or "topic" in request.types:
            cache_key = f"topic:{query_lower}:{tenant_id}"
            now = datetime.now()

            # Check cache
            if cache_key in self._topic_cache:
                cached_time, cached_topics = self._topic_cache[cache_key]
                if (now - cached_time).total_seconds() < self._cache_ttl:
                    suggestions.extend(cached_topics)
                else:
                    del self._topic_cache[cache_key]

            # Generate new if not in cache
            if cache_key not in self._topic_cache:
                topics = self._generate_topic_suggestions(request.query, limit=request.limit // 2)
                self._topic_cache[cache_key] = (now, topics)
                suggestions.extend(topics)

        # Sort by relevance and limit
        suggestions.sort(key=lambda s: (s.relevance_score, s.text.lower()), reverse=True)

        return suggestions[: request.limit]

    def _generate_topic_suggestions(
        self,
        query: str,
        limit: int,
    ) -> List[SearchSuggestion]:
        """
        Generate topic suggestions using LLM.

        Uses improved prompt engineering for better results.

        توليد اقتراحات المواضيع باستخدام LLM
        """
        try:
            prompt = f"""You are a search suggestion assistant for a document retrieval system.

Original query: "{query}"

Task: Generate {limit} topic suggestions that would help users find relevant documents.

Requirements:
1. Each topic must be 2-4 words long
2. Topics should be semantically related to the query
3. Topics should represent potential search intents
4. Focus on concepts and themes, not variations of the same query
5. Avoid repeating the original query

Examples:
Query: "machine learning"
Topics: deep learning, neural networks, data science, model training

Query: "API design"
Topics: REST architecture, GraphQL, endpoint design, API documentation

Generate {limit} topics related to: "{query}"
Return ONLY the topics, one per line:"""

            response = self._llm.generate(prompt, temperature=0.7)
            topics = [t.strip() for t in response.split("\n") if t.strip()][:limit]

            return [SearchSuggestion(text=t, type="topic", relevance_score=0.7) for t in topics]

        except Exception as e:
            log.error("Failed to generate topic suggestions", query=query, error=str(e))
            return []


# -----------------------------------------------------------------------------
# Trie Data Structure for Auto-Suggest
# -----------------------------------------------------------------------------


class TrieNode:
    """Node in a trie data structure for efficient prefix search.

    عقدة في هيكل البيانات trie للبحث المسبع
    """

    __slots__ = ["children", "is_end", "score"]

    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end = False
        self.score = 0.0


class Trie:
    """
    Trie data structure for efficient prefix-based auto-suggest.

    هيكل بيانات trie للاقتراح التلقائي الفعال بالبادئات
    """

    def __init__(self):
        self.root = TrieNode()
        self._size = 0

    def insert(self, word: str, score: float = 1.0) -> None:
        """
        Insert a word into the trie with optional relevance score.

        أدخل كلمة في trie مع درجة صلة اختيارية
        """
        node = self.root
        for char in word.lower():
            node = node.children[char]
        node.is_end = True
        node.score = score
        self._size += 1

    def search_prefix(self, prefix: str, limit: int = 10) -> List[tuple[str, float]]:
        """
        Search for all words starting with prefix.

        البحث عن جميع الكلمات التي تبدأ بالبادئة
        """
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]

        # DFS to collect all words from this node
        results = []

        def dfs(current_node: TrieNode, current_word: str):
            if len(results) >= limit:
                return
            if current_node.is_end:
                results.append((current_word, current_node.score))
            for char, child_node in current_node.children.items():
                dfs(child_node, current_word + char)

        dfs(node, prefix)
        return results

    def autocomplete(
        self, prefix: str, limit: int = 10, min_score: float = 0.0
    ) -> List[tuple[str, float]]:
        """
        Get top autocomplete suggestions for a prefix.

        الحصول على أفضل اقتراحات الإكمال التلقائي لبادئة
        """
        suggestions = self.search_prefix(prefix, limit * 2)
        filtered = [(w, s) for w, s in suggestions if s >= min_score]
        filtered.sort(key=lambda x: (-x[1], x[0]))  # Sort by score desc, name asc
        return filtered[:limit]

    def get_size(self) -> int:
        """Get total number of words in trie."""
        return self._size


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
