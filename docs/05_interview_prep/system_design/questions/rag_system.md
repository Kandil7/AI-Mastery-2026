# System Design Question: RAG System for Enterprise Search

## Problem Statement

Design a Retrieval-Augmented Generation (RAG) system for an enterprise with 10M documents and 50K daily users who need accurate, cited answers.

---

## Requirements

### Functional Requirements
1. Answer questions using company documents
2. Provide citations with every answer
3. Support follow-up questions (conversational)
4. Handle multi-modal documents (text, tables, images)
5. Allow per-user/team access control

### Non-Functional Requirements
1. P95 latency < 3 seconds end-to-end
2. Answer accuracy > 90% on benchmarks
3. Handle 100 concurrent requests
4. Support documents up to 200 pages
5. Update index within 1 hour of document change

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG SYSTEM                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    INGESTION PIPELINE                              │  │
│  │  Documents → Parse → Chunk → Embed → Index → Vector DB            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    QUERY PIPELINE                                  │  │
│  │  Query → Embed → Retrieve → Rerank → Generate → Citations         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Detailed Components

```
                              ┌─────────────────┐
                              │   User Query    │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │ Query Analysis  │
                              │ - Intent detect │
                              │ - Query rewrite │
                              └────────┬────────┘
                                       │
                     ┌─────────────────┼─────────────────┐
                     ▼                 ▼                 ▼
              ┌────────────┐   ┌────────────┐   ┌────────────┐
              │    BM25    │   │   Dense    │   │  Metadata  │
              │  Retrieval │   │  Retrieval │   │   Filter   │
              └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
                    └────────────────┼────────────────┘
                                     │
                            ┌────────▼────────┐
                            │ Reciprocal Rank │
                            │    Fusion       │
                            └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │   Cross-Encoder │
                            │    Reranking    │
                            └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │  LLM Generation │
                            │  + Citations    │
                            └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │ Answer + Sources│
                            └─────────────────┘
```

---

## Component Deep Dives

### 1. Document Ingestion Pipeline

```python
class IngestionPipeline:
    """
    Stages:
    1. Parse: Extract text from PDF/DOCX/HTML
    2. Clean: Remove headers, footers, boilerplate
    3. Chunk: Split into semantic units
    4. Embed: Generate dense vectors
    5. Index: Store in vector database
    """
    
    def process(self, document):
        # 1. Parse
        text = self.parser.extract(document)
        tables = self.parser.extract_tables(document)
        images = self.parser.extract_images(document)
        
        # 2. Clean
        text = self.cleaner.remove_boilerplate(text)
        
        # 3. Chunk with context
        chunks = self.chunker.split(
            text,
            chunk_size=512,
            overlap=50,
            preserve_sentences=True
        )
        
        # 4. Enrich chunks with metadata
        for chunk in chunks:
            chunk.metadata = {
                "doc_id": document.id,
                "page": chunk.page_number,
                "section": chunk.section_title,
                "timestamp": document.modified_at
            }
        
        # 5. Embed and index
        embeddings = self.embedder.encode(chunks)
        self.vector_db.upsert(chunks, embeddings)
```

### 2. Chunking Strategy

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| Fixed size | Simple | May split concepts | General text |
| Semantic | Preserves meaning | Complex | Technical docs |
| Recursive | Adaptive | May miss context | Varied content |
| Document-aware | Section-aware | Parser needed | Structured docs |

**Recommendation**: Semantic chunking with overlap

```
Document Section:
├── Chunk 1 (512 tokens) ──┐
│   └── Overlap (50)       │ Shared context
├── Chunk 2 (512 tokens) ──┘
│   └── Overlap (50)
└── Chunk 3 (400 tokens)
```

### 3. Retrieval Strategy

**Hybrid Retrieval** combines:
- BM25 for keyword matching
- Dense for semantic similarity
- Metadata filtering for access control

```python
def retrieve(query, user, k=20):
    # 1. Dense retrieval
    query_embedding = embed(query)
    dense_results = vector_db.search(
        query_embedding, 
        k=k*2,
        filter={"access_groups": {"$in": user.groups}}
    )
    
    # 2. Sparse retrieval (BM25)
    sparse_results = bm25_index.search(query, k=k*2)
    
    # 3. Fusion
    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=60
    )
    
    return fused[:k]
```

### 4. Reranking

Cross-encoder reranking for precision:

```python
def rerank(query, documents, k=5):
    """
    Cross-encoder scores query-document pairs jointly.
    Much more accurate than bi-encoder, but slower.
    
    Use for: Top 20-50 candidates → Top 5
    """
    pairs = [(query, doc.content) for doc in documents]
    scores = cross_encoder.predict(pairs)
    
    ranked = sorted(zip(documents, scores), key=lambda x: -x[1])
    return [doc for doc, score in ranked[:k]]
```

### 5. Generation with Citations

```python
GENERATION_PROMPT = """
Answer the question using ONLY the provided context.
Cite sources using [1], [2], etc.

Context:
{context}

Question: {question}

Answer with citations:
"""

def generate_answer(query, documents):
    # Build context with source numbers
    context = "\n\n".join([
        f"[{i+1}] {doc.content}"
        for i, doc in enumerate(documents)
    ])
    
    prompt = GENERATION_PROMPT.format(
        context=context,
        question=query
    )
    
    response = llm.generate(prompt)
    
    # Extract and validate citations
    citations = extract_citations(response)
    validated = validate_citations(citations, documents)
    
    return {
        "answer": response,
        "sources": [documents[i-1] for i in validated]
    }
```

---

## Vector Database Choice

| Database | Strengths | Considerations |
|----------|-----------|----------------|
| **Pinecone** | Managed, easy | Cost at scale |
| **Weaviate** | Hybrid search | Self-hosted complexity |
| **Qdrant** | Fast, filtering | Newer |
| **Milvus** | Distributed | Operational overhead |
| **pgvector** | PostgreSQL native | Performance limits |

**Recommendation**: 
- < 1M docs: pgvector or Qdrant
- 1M-10M docs: Qdrant or Pinecone
- > 10M docs: Milvus or Pinecone

---

## Scaling Considerations

### Ingestion Scale
- Batch processing with queues
- Parallel chunking and embedding
- Incremental updates vs full reindex

### Query Scale
- Cache frequent queries
- Pre-compute common retrievals
- Horizontal scaling of API

### Storage Scale
- Shard by document type or date
- Tiered storage (hot/cold)
- Index compression

---

## Evaluation Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Answer accuracy | > 90% | Human evaluation |
| Faithfulness | > 95% | Check claims vs sources |
| Citation accuracy | > 90% | Verify citations exist |
| Retrieval recall | > 85% | Hit rate at top-10 |
| Latency P95 | < 3s | Monitoring |

```python
def evaluate_rag(test_set):
    metrics = {
        "retrieval_recall": [],
        "answer_accuracy": [],
        "faithfulness": [],
        "citation_accuracy": []
    }
    
    for item in test_set:
        result = rag.query(item.question)
        
        # Retrieval
        retrieved_ids = [d.id for d in result.sources]
        relevant_ids = item.relevant_doc_ids
        recall = len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)
        metrics["retrieval_recall"].append(recall)
        
        # Answer quality (requires LLM judge)
        accuracy = llm_judge(result.answer, item.ground_truth)
        metrics["answer_accuracy"].append(accuracy)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

---

## Common Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Wrong answer | Bad retrieval | Improve chunking, add reranking |
| Hallucination | Missing context | Better prompting, faithfulness check |
| Missing info | Chunk too small | Larger chunks, parent-child |
| Slow response | Large context | Limit tokens, cache |
| Access violation | Filter bypass | Strict ACL enforcement |

---

## Interview Discussion Points

1. **How do you handle long documents?**
   - Hierarchical chunking
   - Summarization + detail retrieval
   - Parent-child chunk relationships

2. **How to ensure answer quality?**
   - Cross-encoder reranking
   - LLM self-consistency checks
   - Human feedback loop

3. **How to optimize latency?**
   - Pre-fetch common queries
   - Streaming generation
   - Approximate nearest neighbor

4. **How to handle multi-turn conversations?**
   - Query rewriting with context
   - Sliding window memory
   - Reference resolution

5. **How to measure success?**
   - A/B testing with user feedback
   - Automated evaluation benchmarks
   - Citation verification rate
