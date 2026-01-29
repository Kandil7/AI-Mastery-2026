# System Design Solution Map

Fast reference to the playbooks in `docs/system_design_solutions/`.

---

## 01_RAG_at_scale.md
- **Scenario:** 1M+ docs, 1000 QPS, p95 <500ms.
- **Patterns:** Hybrid retrieval (vector+BM25), reranker, multi-level cache, agentic planning,
  governance baked into retrieval.
- **Use it when:** You need tenant-aware RAG with low latency and cost controls.

## 02_recommendation_system.md
- **Scenario:** 100M users, 10M products, <100ms p95.
- **Patterns:** Ensemble (MF + content + two-tower), 3-tier caching, nightly precompute, bandits
  for A/B.
- **Use it when:** Building large-scale recs with cold-start coverage.

## 03_fraud_detection.md
- **Scenario:** Real-time credit-card fraud scoring.
- **Patterns:** Feature store + streaming, anomaly rules + ML ensemble, low-latency API with
  thresholds.
- **Use it when:** You need layered fraud defenses with explainability and SLAs.

## 04_ml_model_serving.md (and 04_model_serving.md)
- **Scenario:** Serving multiple models with autoscale and canary.
- **Patterns:** FastAPI/GRPC front, Triton/TorchServe, feature store fetch, shadow/canary deploys,
  observability budget.
- **Use it when:** Standing up general model serving with traffic shifting.

## 05_ab_testing.md / 05_ab_testing_framework.md
- **Scenario:** Experimentation at scale across services.
- **Patterns:** Experiment definitions, randomization service, metric guardrails, sequential
  testing, bandits.
- **Use it when:** You need consistent experimentation plumbing across teams.

## 06_churn_prediction.md
- **Scenario:** B2B churn predictor with dual-path serving.
- **Patterns:** Batch + real-time scoring, feature contracts, Salesforce/Email interventions,
  monitoring for drift.
- **Use it when:** Rolling out retention models tied to downstream actions.

## 07_experimentation_platform.md
- **Scenario:** 85M MAU, 600k QPS assignment, 120-200 concurrent experiments; p95 <5ms.
- **Patterns:** Stateless assignment + cached SDKs, versioned config snapshots,
  guardrails/kill-switch, mutual exclusions, sequential tests with CUPED, Thompson bandits,
  near-real-time metrics.
- **Use it when:** You need multi-tenant, low-latency experimentation with strong safety controls.
