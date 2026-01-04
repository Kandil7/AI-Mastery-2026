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

---

### Week 01: System Design & Interview Prep
**Date**: 2026-01-04

**🏗️ Implementation (What I Built)**
- Completed 2 system design documents (100% of system designs done):
  - `docs/system_design_solutions/04_model_serving.md`: ML serving at 10K req/s with dynamic batching, deployment strategies (blue-green, canary), cost optimization (~$4.8K/month)
  - `docs/system_design_solutions/05_ab_testing.md`: A/B testing platform for 10M users with Thompson Sampling, statistical testing, sequential analysis (~$2K/month)
- Created `docs/CAPSTONE_DEMO_SCRIPT.md`: 7-part professional demo walkthrough (5-7 min)
- Updated project tracking: 28% complete (29/104 tasks)
- Created comprehensive implementation plan with 3 priority tiers

**🐛 Debugging (What Broke & How I Fixed It)**
- **Issue**: 104 tasks felt overwhelming, unclear priorities
- **Fix**: Organized into Tier 1 (Job-Ready, 2-3 days), Tier 2 (Competitive, 1-2 weeks), Tier 3 (Elite, 3-4 weeks)
- **Result**: Clear critical path for rapid job application readiness

**💡 Realization**
- **Dynamic Batching Math**: Batch of 32 gives **21x throughput** (10ms → 0.47ms per request). Trade-off is 5ms max queue delay.
- **Thompson Sampling**: Beta distribution elegantly balances exploration/exploitation without manual tuning. Tracks successes/failures, samples from posterior, picks best.
- **System Design Numbers**: Specific costs ($4.8K/month), latencies (p95 < 50ms), sample sizes (3,842 per variant) make designs interview-ready, not theoretical.
- **Production ML Trade-offs**: Canary > blue-green for model changes (gradual validation), Redis caching saves 40% cost at 40% hit rate, Triton for GPU models vs FastAPI for simple CPU models.

**⏭️ Next Focus**
- Tomorrow (Jan 5): Record capstone demo video, upload to YouTube/Loom
- Start Week 5 ResNet implementation (`src/ml/vision.py`)
- Practice explaining all 5 system designs out loud (10 min each)

