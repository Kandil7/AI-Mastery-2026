# 🧠 Learning Log & Changelog

> **Philosophy**: Fast learning = fast measurement. Log what changed in code, what broke, and what you can now build that you couldn't last week. This is your "Fast Review" substrate for interviews.

## 📝 Template
```markdown
### Week [N]: [Topic]
**Date**: YYYY-MM-DD

**🏗️ Implementation (What I Built)**
- [ ] Created `src/...`
- [ ] Merged ...

**🐛 Debugging (What Broke & How I Fixed It)**
- Error: ...
- Fix: ...

**💡 Realization (Theory -> Code connection)**
- I understood X when I saw it implemented as ...

**⏭️ Next Week's Focus**
- ...
```

---

## 📅 Logs

### Week 01: RAG Deep Dive
**Date**: 2026-01-02

**🏗️ Implementation (What I Built)**
- Verified `HybridRetriever` (BM25 + Semantic) in `src/retrieval`.
- Built `day1_hybrid_demo.ipynb` comparing Sparse vs Dense retrieval.
- Built `day2_eval_pipeline.ipynb` using Ragas for automated metrics (Recall, Faithfulness).
- Built Full Stack Demo: FastAPI (`api.py`) + Streamlit (`ui.py`).

**🐛 Debugging (What Broke & How I Fixed It)**
- *Placeholder for actual debugging notes during Day 4 stress test*

**💡 Realization**
- Realized that Dense retrieval fails on specific error codes, while Sparse handles them perfectly, validating the need for Hybrid.

