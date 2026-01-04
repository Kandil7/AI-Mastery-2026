# Notion AI: Production RAG Architecture

## Business Context

**Challenge**: Notion needed to provide AI-powered Q&A across millions of user workspaces, each containing unique documents, databases, and notes. Key requirements:
- Answer questions using only the user's own content
- Minimize hallucination (critical for business users)
- Keep costs manageable at scale

**Solution**: Enterprise RAG with semantic chunking, hybrid retrieval, and intelligent model routing.

---

## Technical Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Semantic   │───▶│   Hybrid     │───▶│    Model     │───▶│  LLM-as-Judge│
│   Chunking   │    │   Retrieval  │    │    Router    │    │  Evaluation  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
 ┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐
 │Hierarchical     │Dense+Sparse│      │Task-Based │      │Relevance  │
 │Structure  │      │RRF Fusion │      │Selection  │      │Accuracy   │
 │Preserved  │      │           │      │           │      │Scoring    │
 └───────────┘      └───────────┘      └───────────┘      └───────────┘
```

---

## Key Components

### 1. Semantic Chunking

Traditional chunking splits on character count. Notion's approach preserves document structure:

```python
class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"      # Basic: split every N chars
    SENTENCE = "sentence"          # Better: respect sentence boundaries
    PARAGRAPH = "paragraph"        # Good: preserve paragraph units
    SEMANTIC = "semantic"          # Best: detect headers, sections, lists
    HIERARCHICAL = "hierarchical"  # Advanced: create parent-child relationships
```

**Hierarchical Example**:
```
Document Summary (Level 0)
    ├── Section: "Getting Started" (Level 1)
    │       ├── Paragraph: "Installation steps..." (Level 2)
    │       └── Paragraph: "Configuration..." (Level 2)
    └── Section: "Advanced Usage" (Level 1)
            └── Paragraph: "Custom plugins..." (Level 2)
```

### 2. Hybrid Retrieval with RRF

Neither dense (embedding) nor sparse (BM25) is perfect alone:

| Approach | Strength | Weakness |
|----------|----------|----------|
| Dense | Semantic similarity | Misses exact keywords |
| Sparse | Exact keyword matching | No semantic understanding |
| **Hybrid** | Best of both | Requires score fusion |

**Reciprocal Rank Fusion (RRF)**:
```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    final_scores = {}
    for rank, doc in enumerate(dense_results):
        final_scores[doc.id] = final_scores.get(doc.id, 0) + 1/(k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        final_scores[doc.id] = final_scores.get(doc.id, 0) + 1/(k + rank + 1)
    return sorted(final_scores.items(), key=lambda x: -x[1])
```

### 3. Model Router

Not every query needs GPT-4:

| Query Type | Routed Model | Cost/1K tokens |
|------------|--------------|----------------|
| Simple extraction | Claude Instant | $0.0008 |
| General Q&A | GPT-3.5-turbo | $0.002 |
| Complex reasoning | GPT-4-turbo | $0.03 |

**Routing Logic**:
```python
def route(query, context_length, task_type):
    if task_type == "extraction" and context_length < 4000:
        return "claude-instant"  # Fast + cheap
    elif task_type == "reasoning" or requires_multi_step(query):
        return "gpt-4-turbo"  # Quality matters
    else:
        return "gpt-3.5-turbo"  # Balanced default
```

### 4. LLM-as-Judge Evaluation

Automated quality scoring without human labels:

```python
evaluation_criteria = [
    {"name": "relevance", "weight": 1.0},
    {"name": "accuracy", "weight": 1.5},   # Higher weight!
    {"name": "completeness", "weight": 1.0},
    {"name": "coherence", "weight": 0.5}
]

def evaluate(query, answer, context):
    prompt = f"""Rate this answer on each criterion (1-5):
    
    Question: {query}
    Context: {context}
    Answer: {answer}
    
    Criteria: {criteria}
    """
    return judge_llm.generate(prompt)
```

---

## Production Results

| Metric | Before (2023) | After (2024) | Change |
|--------|---------------|--------------|--------|
| Answer Accuracy | 72% | 89% | **+23.6%** |
| Hallucination Rate | 15% | 4% | **-73.3%** |
| Avg. Cost/Query | $0.018 | $0.007 | **-61%** |
| P95 Latency | 3.2s | 1.8s | -43.8% |
| User Satisfaction | 3.6/5 | 4.4/5 | **+22%** |

---

## Implementation in This Project

See: [`src/llm/advanced_rag.py`](file:///k:/learning/technical/ai-ml/AI-Mastery-2026/src/llm/advanced_rag.py)

**Key Classes**:
- `SemanticChunker`: Multiple chunking strategies
- `DenseRetriever` / `SparseRetriever`: Individual retrieval methods
- `HybridRetriever`: RRF-based fusion
- `ModelRouter`: Cost-optimized model selection
- `LLMJudge`: Automated evaluation
- `EnterpriseRAG`: Unified pipeline

---

## Code Example

```python
from src.llm.advanced_rag import EnterpriseRAG, Document, ChunkingStrategy

# Initialize with semantic chunking
rag = EnterpriseRAG(chunking_strategy=ChunkingStrategy.HIERARCHICAL)

# Add workspace documents
documents = [
    Document(id="doc1", content=page.content, metadata={"page_id": page.id})
    for page in workspace.pages
]
rag.add_documents(documents)

# Query with smart routing
result = rag.query(
    query="What's our refund policy?",
    query_embedding=embed(query),
    task_type="extraction"  # Will route to fast model
)

print(f"Answer: {result.answer}")
print(f"Model: {result.model_used}")
print(f"Sources: {[s.id for s in result.sources]}")
```

---

## Lessons Learned

1. **Chunking is critical**: Hierarchical chunks improved retrieval accuracy by 18%
2. **Hybrid > Pure Dense**: Adding BM25 caught 23% more keyword-heavy queries
3. **Route 80% to cheap models**: Most queries don't need GPT-4
4. **Evaluate automatically**: LLM-as-Judge scales better than human review

---

## References

- Notion AI Architecture Blog
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
