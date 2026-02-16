# Search Enhancements Guide
# دليل تحسينات البحث

## Table of Contents / جدول المحتويات

1. [Introduction / مقدمة](#introduction)
2. [Auto-Suggest / الاقتراح التلقائي](#auto-suggest)
3. [Query Expansion / توسيع الاستعلامات](#query-expansion)
4. [Faceted Search / البحث المجزوء](#faceted-search)
5. [Trie Data Structure / هيكل البيانات Trie](#trie-data-structure)
6. [Best Practices / أفضل الممارسات](#best-practices)
7. [Summary / الملخص](#summary)

---

## Introduction / مقدمة

### Why Search Enhancements? / لماذا تحسينات البحث؟

Search enhancements improve the user experience by:

**1. Auto-Suggest / الاقتراح التلقائي**
- Predictive text completion as users type
- Reduces typing effort
- Guides users to relevant content

**2. Query Expansion / توسيع الاستعلامات**
- Suggests related search terms
- Handles synonyms and alternative phrasings
- Improves recall (finds more relevant results)

**3. Faceted Search / البحث المجزوء**
- Organizes results by categories
- Enables filtering by attributes
- Improves navigation and discovery

**تحسينات البحث تحسن تجربة المستخدم من خلال:**

---

## Auto-Suggest / الاقتراح التلقائي

### Types of Suggestions / أنواع الاقتراحات

**1. Document Name Suggestions / اقتراحات أسماء المستندات**

```python
# Example: User types "proj"
Suggestions:
- "project_plan.pdf"
- "project_specs.docx"
- "project_timeline.pdf"
- "proj_summary.md"
```

**2. Query Expansion Suggestions / اقتراحات توسيع الاستعلامات**

```python
# Example: User types "rag"
Suggestions:
- "retrieval augmented generation"
- "rag architecture"
- "hybrid search"
- "vector database"
```

**3. Topic Suggestions / اقتراحات المواضيع**

```python
# Example: User types "machine learning"
Suggestions:
- "deep learning"
- "neural networks"
- "data science"
- "model training"
```

### أنواع الاقتراحات

**1. اقتراحات أسماء المستندات**

**2. اقتراحات توسيع الاستعلامات**

**3. اقتراحات المواضيع**

---

## Query Expansion / توسيع الاستعلامات

### Strategies / استراتيجيات

**1. LLM-Based Expansion / التوسيع المستند إلى LLM**

```python
def expand_query(query: str, context: List[Document]) -> List[str]:
    """
    Use LLM to generate semantically related queries.

    استخدام LLM لتوليد استعلامات مرتبطة دلاليًا
    """
    # Build context from similar documents
    doc_context = "\n".join([d.filename for d in context[:5]])

    prompt = f"""Original query: "{query}"

Context from similar documents:
{doc_context}

Task: Generate 3 alternative search queries that would retrieve relevant documents.

Requirements:
- Use synonyms and related terms
- Cover different aspects of the topic
- Be concise (2-5 words each)

Return ONLY 3 queries, one per line:"""

    expanded = llm.generate(prompt, temperature=0.7)
    return [q.strip() for q in expanded.split("\n") if q.strip()]
```

**2. Rule-Based Expansion / التوسيع القائم على القواعد**

```python
# Synonym mapping
SYNONYMS = {
    "rag": ["retrieval augmented generation", "retrieval-augmented"],
    "vector": ["embedding", "representation", "feature vector"],
    "search": ["query", "find", "retrieve"],
    "document": ["file", "record", "entry"],
}

def expand_with_synonyms(query: str) -> List[str]:
    """Expand query using synonym dictionary."""
    expanded = [query]
    words = query.lower().split()

    for word in words:
        if word in SYNONYMS:
            for synonym in SYNONYMS[word]:
                expanded.append(" ".join(synonym if w != word else w for w in words))

    return list(set(expanded))
```

### استراتيجيات

**1. التوسيع المستند إلى LLM**

**2. التوسيع القائم على القواعد**

---

## Faceted Search / البحث المجزوء

### Facet Types / أنواع الجوانب

**1. Status Facet / جانب الحالة**

```json
{
  "status": [
    {"name": "indexed", "count": 150},
    {"name": "processing", "count": 12},
    {"name": "failed", "count": 3}
  ]
}
```

**2. Content Type Facet / جانب نوع المحتوى**

```json
{
  "content_type": [
    {"name": "application/pdf", "count": 100},
    {"name": "text/plain", "count": 45},
    {"name": "application/docx", "count": 20}
  ]
}
```

**3. Size Range Facet / جانب نطاق الحجم**

```python
# Size ranges (in KB)
RANGES = [
    ("0-100KB", 0, 100 * 1024),
    ("100KB-1MB", 100 * 1024, 1024 * 1024),
    ("1MB-10MB", 1024 * 1024, 10 * 1024 * 1024),
    ("10MB+", 10 * 1024 * 1024, float("inf")),
]

# Compute counts
size_counts = {"0-100KB": 50, "100KB-1MB": 80, "1MB-10MB": 30, "10MB+": 5}
```

**4. Date Range Facet / جانب نطاق التاريخ**

```python
from datetime import datetime, timedelta

# Define ranges
now = datetime.now()
RANGES = [
    ("Last 7 days", now - timedelta(days=7), now),
    ("Last 30 days", now - timedelta(days=30), now),
    ("Older than 30 days", None, now - timedelta(days=30)),
]

# Compute counts
date_counts = {
    "Last 7 days": 20,
    "Last 30 days": 120,
    "Older than 30 days": 25,
}
```

### أنواع الجوانب

**1. جانب الحالة**

**2. جانب نوع المحتوى**

**3. جانب نطاق الحجم**

**4. جانب نطاق التاريخ**

---

## Trie Data Structure / هيكل البيانات Trie

### What is a Trie? / ما هي Trie؟

A **Trie** (prefix tree) is a tree-like data structure used for efficient string prefix searches.

**Key properties:**
- Each node represents a character
- Paths from root to leaf represent strings
- Fast prefix lookup: O(m) where m is prefix length
- Memory efficient for shared prefixes

**Trie** هيكل شجري مستخدم للبحث المسبع الفعال بالبادئات.

**الخصائص الرئيسية:**
- كل عقدة تمثل حرفًا
- المسارات من الجذر إلى الأوراق تمثل سلاسل
- البحث المسبع السريع: O(m) حيث m هو طول البادئة
- استهلاك الذاكرة الفعال للبادئات المشتركة

### Trie Implementation / تنفيذ Trie

```python
from collections import defaultdict

class TrieNode:
    """Node in a trie data structure."""
    __slots__ = ['children', 'is_end', 'score']

    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end = False
        self.score = 0.0


class Trie:
    """Trie data structure for efficient prefix search."""

    def __init__(self):
        self.root = TrieNode()
        self._size = 0

    def insert(self, word: str, score: float = 1.0) -> None:
        """Insert a word into the trie."""
        node = self.root
        for char in word.lower():
            node = node.children[char]
        node.is_end = True
        node.score = score
        self._size += 1

    def autocomplete(self, prefix: str, limit: int = 10) -> List[tuple[str, float]]:
        """Get top autocomplete suggestions for a prefix."""
        # Find node at end of prefix
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]

        # DFS to collect all words from this node
        results = []

        def dfs(current_node, current_word):
            if len(results) >= limit:
                return
            if current_node.is_end:
                results.append((current_word, current_node.score))
            for char, child_node in current_node.children.items():
                dfs(child_node, current_word + char)

        dfs(node, prefix)
        return sorted(results, key=lambda x: (-x[1], x[0]))[:limit]
```

### ما هي Trie؟

### تنفيذ Trie

---

## Best Practices / أفضل الممارسات

### 1. Cache Results / تخزين النتائج مؤقتًا

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_facets(tenant_id: str, query: str):
    """Cache facet computation results."""
    # Only compute once per unique query
    return compute_facets(tenant_id, query)
```

### 2. Limit Results / تحديد النتائج

```python
def get_suggestions(query: str, limit: int = 10):
    """Always limit suggestions for performance."""
    suggestions = []

    # Limit each type
    for suggestion_type in ["document", "query", "topic"]:
        type_suggestions = get_type_suggestions(
            query, suggestion_type, limit // 3
        )
        suggestions.extend(type_suggestions)

    return suggestions[:limit]
```

### 3. Score by Relevance / الترتيب حسب الصلة

```python
def score_suggestions(suggestions: List[Suggestion]) -> List[Suggestion]:
    """
    Score and sort suggestions by relevance.

    Factors:
    - Prefix match quality
    - Recency/popularity
    - Click-through rate
    """
    for s in suggestions:
        # Higher score for exact prefix matches
        if s.text.startswith(query):
            s.relevance_score *= 1.5

        # Higher score for recently accessed
        if s.last_accessed_days_ago < 7:
            s.relevance_score *= 1.2

    return sorted(suggestions, key=lambda x: x.relevance_score, reverse=True)
```

### 1. تخزين النتائج مؤقتًا

### 2. تحديد النتائج

### 3. الترتيب حسب الصلة

---

## Summary / الملخص

### Key Takeaways / النقاط الرئيسية

1. **Auto-suggest** improves UX by predicting user intent
2. **Query expansion** uses synonyms and LLM for better recall
3. **Faceted search** organizes results for better navigation
4. **Trie data structure** enables efficient prefix-based autocomplete
5. **Best practices**: caching, limiting, relevance scoring

### النقاط الرئيسية

1. **الاقتراح التلقائي** يحسن تجربة المستخدم من خلال التنبؤ بالنية
2. **توسيع الاستعلامات** يستخدم المرادفات وLLM لاسترجاع أفضل
3. **البحث المجزوء** ينظم النتائج للتنقل الأفضل
4. **هيكل البيانات Trie** يسمح بالإكمال التلقائي الفعال بالبادئات
5. **أفضل الممارسات**: التخزين المؤقت، التحديد، الترتيب حسب الصلة

---

## Further Reading / قراءة إضافية

- [Elasticsearch Query DSL](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html)
- [Autocomplete Best Practices](https://www.algolia.com/doc/guides/overview/autocomplete)
- [Trie Data Structure](https://en.wikipedia.org/wiki/Trie)

---

## Arabic Summary / ملخص بالعربية

هذا الدليل يغطي تحسينات البحث في محرك RAG. تشمل المواضيع الرئيسية:

1. **الاقتراح التلقائي**: الإكمال التلقائي أثناء الكتابة
2. **توسيع الاستعلامات**: استخدام المرادفات وLLM لتحسين الاسترجاع
3. **البحث المجزوء**: تنظيم النتائج حسب الفئات
4. **هيكل البيانات Trie**: تنفيذ فعال للبحث بالبادئات
5. **أفضل الممارسات**: التخزين المؤقت، التحديد، الترتيب حسب الصلة

تساهم هذه التحسينات في تجربة بحث سلسة وسريعة.
