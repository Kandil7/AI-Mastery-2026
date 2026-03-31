# System Design: Experimentation Platform at Global Scale

## Scenario

- 85M MAU; 600k QPS assignment at peak; 120-200 concurrent experiments.
- Cross-platform (web/iOS/Android) with CDN edges; some experiments nested behind feature flags.
- SLOs: assignment p95 <5 ms; exposure loss <0.1%; metrics freshness <15 minutes; auto kill within
  5 minutes of guardrail breach.

Use when you need multi-tenant, low-latency A/B testing with strong guardrails across many teams.

---

## Architecture (text)

```
            Admin UI
               |
           Config API
               |
     Postgres (source of truth)
               |
      Config Publisher -> Redis -> CDN edge snapshots
               |
         Assignment Service (stateless; consistent hashing)
               |
         SDKs on clients/servers (cache + kill-switch listener)
               |
  Kafka (exposures + metrics) -> Flink/Beam processors
               |
        Metrics Store (Druid/ClickHouse) -> Stats Engine
               |
        Dashboards / Alerts / Kill-switch topic
```

---

## Components

- **Config Service**: Versioned experiment definitions (id, namespace, traffic splits, exclusions,
  start/end, metrics, guardrails). Publishes signed snapshots to Redis + CDN for low-latency reads.
- **Assignment Service**: Stateless API; consistent hashing on subject_id; respects namespaces and
  mutual-exclusion sets. Supports layered bucketing (global -> surface -> experiment).
- **SDKs**: Client + server SDKs that cache snapshots, emit exposures, listen to kill-switch topic,
  and fall back to baked defaults if config is stale.
- **Stream Processor**: Joins exposures with events; computes rolling guardrails (error rate, GMV,
  latency) in 5-minute windows; emits to metrics store.
- **Stats Engine**: Sequential tests (SPRT), CUPED/CUPAC for variance reduction, bandits (Thompson)
  for long-running or low-sensitivity experiments.
- **Governance**: RBAC on config API; approvals for high-blast surfaces; audit logs for SOC2/HIPAA.

---

## Data Model (simplified)

```sql
CREATE TABLE experiments (
  experiment_id   UUID PRIMARY KEY,
  namespace       TEXT,
  surface         TEXT,
  start_ts        TIMESTAMPTZ,
  end_ts          TIMESTAMPTZ,
  splits          JSONB,         -- [{"variant":"control","share":0.9}, ...]
  exclusions      TEXT[],        -- experiment_ids mutually exclusive
  metrics         JSONB,         -- primary + guardrail metric ids
  kill_switch     BOOLEAN DEFAULT FALSE
);

CREATE TABLE exposures (
  experiment_id UUID,
  variant       TEXT,
  subject_hash  BIGINT,
  ts            TIMESTAMPTZ,
  surface       TEXT
);
```

---

## Traffic & Safety Controls

- **Guardrails by default**: Block auto-promote if error rate, GMV, or latency degrades beyond
  thresholds; triggers kill-switch broadcast to SDKs.
- **Mutual exclusions**: Namespaces with collision rules (e.g., only one checkout experiment
  active).
- **Holdouts**: 1-5% long-term holdouts per surface to detect systemic drift.
- **Bandits**: Thompson sampling for many-arm or small-effect experiments; revert to fixed splits
  for regulatory/medical surfaces.
- **Progressive delivery**: Ramp 1% -> 10% -> 50% -> 100% with automatic pausing on alerts.

---

## Observability & Reliability

- Tracing: Assignment path instrumented; logs include experiment_id, variant, snapshot version.
- Metrics: p95 assignment latency, snapshot staleness, exposure drop rate, guardrail breaches.
- On-call runbooks: stale snapshot? fall back to baked config; Kafka lag? trigger partial kill; bad
  split? rotate config version and invalidate CDN cache.
- Disaster recovery: Config snapshots replicated cross-region; Kafka mirrored topics; cold path
  batch recompute of exposures to fix gaps.

---

## Capacity Notes

- Redis sized for 2x peak QPS with 1% miss tolerance; snapshot payload ~50-200 KB compressed.
- Kafka: plan for 2x experiment traffic; retention 7-14 days; compaction on config topic.
- Flink: autoscale by partitions (exposures + metrics topics each 64-128 partitions).

---

## Build vs Buy Checklist

- Buy if you need SOC2-ready platform fast and experimentation is not core.
- Build if you need tight coupling with feature flags, custom variance reduction, or on-device
  assignment for offline/edge scenarios.
