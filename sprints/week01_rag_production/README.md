# Week 1: Production RAG System

**Status**: 🚧 In Progress 
**Sprint**: Week 1 (LLM & RAG Mastery)  
**Goal**: Build a production-ready RAG system with hybrid retrieval and evaluation metrics.

---

## 🎯 Problem Statement

Most RAG tutorials stop at "chat with your PDF." This sprint focuses on the **Production** gap:
- How to retrieve accurately when keywords fail (Semantic Search)?
- How to retrieve specific terms like "Schema 1.2" (Keyword Search)?
- How to know if the answer is actually correct (Ragas Evaluation)?

---

## 🏗️ Architecture

```
┌──────────────┐      ┌───────────────┐      ┌───────────────┐
│ Streamlit UI │─────▶│ FastAPI Route │─────▶│ RAG Pipeline  │
└──────────────┘      └───────────────┘      └───────┬───────┘
                                                     │
                                        ┌────────────▼────────────┐
                                        │ Hybrid Retriever        │
                                        │ (ChromaDB + BM25)       │
                                        └────────────┬────────────┘
                                                     │
                                        ┌────────────▼────────────┐
                                        │      LLM (OpenAI/       │
                                        │      Local Llama)       │
                                        └─────────────────────────┘
```

---

## 🛠️ Sprint Tasks

- [x] **Day 1**: Implement `HybridRetriever` (Dense + Sparse) - [Demo Notebook](notebooks/day1_hybrid_demo.ipynb)
- [x] **Day 2**: Build Eval Pipeline (Context Recall, Faithfulness) with Ragas - [Eval Notebook](notebooks/day2_eval_pipeline.ipynb)
- [x] **Day 3**: Wrap in FastAPI + Streamlit Dashboard - [Backend](api.py) | [Frontend](ui.py)
- [x] **Day 4**: "Stress Test" - Index 100 complex docs and benchmark - [Benchmark Script](stress_test.py)

## Production Implementation

See `IMPLEMENTATION.md` for the notebook-to-production mapping, module layout,
and the staged plan to reach a 2026-grade RAG system.

---

## 📊 Success Metrics

| Metric | Target |
|--------|--------|
| Retrieval Recall @ 5 | > 85% |
| Generation Faithfulness | > 90% |
| Latency p95 | < 800ms |

---

## 🎤 Interview Prep

**Question**: "How do you improve RAG performance?"
- **Answer key**: Hybrid retrieval, HyDE, Reranking, Metadata filtering.

