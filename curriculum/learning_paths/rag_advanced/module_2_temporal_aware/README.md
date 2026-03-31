# Module 2: Temporal-Aware RAG Systems

## 📋 Module Overview

**Duration:** 2-3 weeks (15-20 hours)  
**Difficulty:** Advanced  
**Prerequisites:** Module 1 completion, Understanding of time-series data

This module teaches you to build RAG systems that understand and leverage temporal information for more accurate, time-aware retrieval and generation.

---

## 🎯 Learning Objectives (Bloom's Taxonomy)

### Remember
- Define temporal awareness in RAG context
- Identify temporal entities (dates, events, trends)
- Recall time-decay functions and recency boosting

### Understand
- Explain why temporal context matters for retrieval
- Describe time-aware chunking strategies
- Summarize temporal query understanding approaches

### Apply
- Implement temporal entity extraction
- Build time-decay scoring functions
- Create temporal-aware chunking pipelines

### Analyze
- Compare temporal vs. non-temporal retrieval quality
- Diagnose temporal drift in retrieved content
- Evaluate recency vs. relevance trade-offs

### Evaluate
- Assess when temporal awareness is critical
- Critique time-decay parameter choices
- Judge temporal grounding in generated answers

### Create
- Design end-to-end temporal RAG architectures
- Develop custom temporal scoring functions
- Build temporal drift detection systems

---

## 📚 Module Structure

```
module_2_temporal_aware/
├── README.md
├── theory/
│   ├── 01_temporal_foundations.md
│   ├── 02_time_aware_chunking.md
│   ├── 03_temporal_scoring.md
│   ├── 04_query_temporal_understanding.md
│   └── 05_temporal_drift_detection.md
├── labs/
│   ├── lab_1_temporal_extraction/
│   ├── lab_2_time_decay_scoring/
│   └── lab_3_temporal_rag_pipeline/
├── knowledge_checks/
├── coding_challenges/
├── solutions/
└── further_reading.md
```

---

## Key Concepts Preview

### Temporal Relevance Decay

```python
def time_decay_score(doc_timestamp: datetime, query_time: datetime, 
                     half_life_days: int = 30) -> float:
    """
    Calculate time-decay score for document.
    
    Uses exponential decay: score = 2^(-days_since_update / half_life)
    """
    days_old = (query_time - doc_timestamp).days
    return 2 ** (-days_old / half_life_days)
```

### Temporal Query Classification

```
Query: "What was the revenue last quarter?"
→ Temporal Intent: HISTORICAL
→ Time Range: Q-1 (previous quarter)
→ Recency Weight: 0.3 (historical accuracy > recency)

Query: "What is the current system status?"
→ Temporal Intent: CURRENT
→ Time Range: Now (real-time)
→ Recency Weight: 0.9 (recency critical)
```

---

## 🎓 Completion Criteria

1. ✅ Complete all theory sections
2. ✅ Implement temporal entity extraction
3. ✅ Build time-decay scoring system
4. ✅ Create temporal RAG pipeline
5. ✅ Score 80%+ on knowledge checks

---

*Last Updated: March 30, 2026*
