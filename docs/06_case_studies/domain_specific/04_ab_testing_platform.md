# Case Study 4: Experimentation Platform for a Global Marketplace

## Executive Summary

**Problem**: A rides & food marketplace ran 120+ concurrent experiments, but assignments were
inconsistent across web/iOS/Android, metric delays slowed decisions (3-5 days), and misconfigured
tests caused revenue dips.

**Solution**: Built a centralized experimentation platform with consistent hashing, config-driven
enrollment, guardrails, sequential testing, and bandits for long-tail optimizations.

**Impact**: Time-to-decision dropped from 4 days to same-day (p50 6 hours); incident rate from bad
splits fell from 3/month to <1/quarter; weekly GMV +1.8% via safer velocity of launches.

**System design snapshot**  
Full design: `docs/system_design_solutions/07_experimentation_platform.md`.
- SLOs: assignment latency <5 ms p95; exposure loss <0.1%; metrics freshness <15 minutes.
- Scale: 85M MAU, 600k QPS peak assignment, 150M events/day to the metrics pipeline.
- Safety: per-metric guardrails, auto kill-switch if error rate or GMV drops >1% over 15 mins.
- Multi-tenant: experiment namespaces per org/team; mutual exclusivity sets for collisions.

---

## Business Context

- **Industry**: Two-sided marketplace (rides + delivery)
- **Footprint**: 35 countries; web, iOS, Android
- **Experiment Volume**: 120-200 concurrent; many nested behind feature flags
- **Primary Goals**: Faster iteration without revenue regressions; unify telemetry for product + ML

### Key Challenges
1. Cross-platform consistency: mobile clients cached flags differently, causing arm leakage.
2. Metric latency: nightly batch jobs delayed go/no-go calls.
3. Blast radius: mis-scoped experiments overwrote each other in the same surface.
4. Governance: need HIPAA/SOC2 audit for medical-delivery vertical.

---

## Experiment Design & Data

- **Unit of randomization**: user_id (fallback to device_id for logged-out).
- **Traffic policy**: config defines buckets (e.g., 90/5/5) plus holdouts for long-term drift
  checks.
- **Exposures**: streamed to Kafka with experiment_id, variant, timestamp, hashed_subject, surface.
- **Metrics**: near-real-time aggregation in Druid; precomputed guardrail views (error rate, GMV,
  p50/p95 latency) refreshed every 5 minutes.
- **Exclusions**: mutual-exclusion sets per surface; global kill-switch topic consumed by SDKs.

---

## Architecture (text)

```
Client (web/iOS/Android)
    -> Assignment SDK (hash + cache; listens to kill-switch)
        -> Experiment Config Service (Redis hot cache; Postgres source)
    -> Event SDK -> Kafka -> Stream Processor (Flink)
        -> Metrics Store (Druid) -> Stats Engine -> Dashboards/Alerts
Control plane: Admin UI -> Config API -> Postgres -> Redis -> CDN edge snapshots
```

---

## Implementation Snapshot

```python
# assignment.py
from hashlib import sha256

def bucket(subject_id: str, experiment_id: str, splits: list[tuple[str, float]]) -> str:
    key = f"{experiment_id}:{subject_id}".encode()
    score = int(sha256(key).hexdigest(), 16) % 10_000 / 10_000
    cumulative = 0.0
    for variant, share in splits:
        cumulative += share
        if score < cumulative:
            return variant
    return "control"

# guardrail check (stream)
if metric_window.gmv_delta < -0.01 or metric_window.error_rate > 0.5:
    publish_kill_switch(experiment_id)
```

---

## Results

| Metric | Before | After |
|--------|--------|-------|
| Assignment p95 | ~22 ms (mobile cache misses) | 4.3 ms |
| Metrics freshness | Nightly batch | 12-15 minutes |
| Exposure loss | ~1.2% (client drops) | 0.08% |
| Incidents from bad splits | 3 / month | <1 / quarter |
| Time-to-decision | 3-5 days | Same day (p50 6h) |
| GMV impact | - | +1.8% weekly lift |

---

## Lessons Learned

- Ship kill-switch first; enforcement belongs in SDK + control plane, not just dashboards.
- Make mutual-exclusion first-class; collisions caused the majority of early incidents.
- Guardrails by default: block promotions if critical metrics degrade, even when primary metric
  wins.
- Streamed exposures + pre-aggregated guardrails are the biggest unlock for speed with safety.
