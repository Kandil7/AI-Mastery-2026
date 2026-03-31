# Full-Stack Case Studies (Industry Patterns)

Curated examples from `case_studies/full_stack_ai` to reuse patterns quickly.

---

## Salesforce Einstein: Trust Layer for Enterprise AI

### Problem
Offer LLM-powered features without leaking PII, while meeting SOC2/HIPAA and zero-retention commitments.

### Solution
- Trust Layer wraps every LLM call with PII masking, jailbreak filtering, audit logging, and ZRA headers/TTL.
- Output validator blocks harmful content; logs every request/response for compliance.
- Works as a sidecar/gateway so app teams adopt without model changes.

**Architecture (text):**
```
App -> Trust Layer (PII mask -> safety -> audit -> ZRA) -> LLM Provider
                                   \
                                    -> Output validator -> App response
```

More: `case_studies/full_stack_ai/04_salesforce_trust_layer.md`.

---

## Pinterest: Multi-Stage Ranking Pipeline

### Problem
Rank billions of pins for 400M+ users with p95 <100ms while balancing personalization and diversity.

### Solution
- Four-stage funnel: candidate gen (kNN/pop) -> pre-rank MLP -> full-rank two-tower -> business re-rank.
- Latency budget per stage: 10ms / 20ms / 50ms / 10ms; drops 1M -> 25 items.
- Diversity and freshness rules in the final pass to reduce echo chambers.

**Architecture (text):**
```
1M candidates -> kNN/pop -> 10K -> light MLP -> 500 -> two-tower -> 50
-> business re-rank (diversity/freshness) -> Top 25 pins
```

**Key Metrics:** <100ms total p95; maintains diversity while improving engagement.

More: `case_studies/full_stack_ai/05_pinterest_ranking_pipeline.md`.

---

## DoorDash: Gigascale Feature Store

### Problem
Serve 10B feature reads/day for online models with sub-millisecond latency.

### Solution
- Dual write path (Kafka streams + batch) into shared feature storage.
- Redis cluster as low-latency serving layer; read path optimized for hot-feature access.
- Model serving fetches features via RPC; strong schema/versioning for backward compatibility.

**Architecture (text):**
```
Kafka + batch -> write path -> feature storage -> Redis cluster
Redis -> read path -> model serving -> predictions
```

**Operational Notes:** Handles bursty traffic with sharded Redis; background TTL eviction; schema registry prevents drift.

More: `case_studies/full_stack_ai/06_doordash_feature_store.md`.

