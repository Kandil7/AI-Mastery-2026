# 06 Churn Prediction System Design (SaaS)

## Scope and goals
- Predict B2B SaaS customer churn 30 days in advance; trigger interventions via CS tooling.
- Serve both daily batch scores (all tenants) and low-latency on-demand scores for UI/API.
- Priorities: recall of churners, explainability for CS, stable ops with tight freshness SLOs.

## Requirements
- Functional: ingest usage/billing/support data; build 47+ features; train/register/serve models;
  surface risk scores and top drivers; push tasks to Salesforce and email.
- Non-functional: p99 online latency <150 ms; feature freshness <30 minutes; batch SLA 06:00 UTC;
  availability 99.5%; RPO 24h; rollback within 15 minutes.
- Guardrails: data quality gates, drift detection, audit trails, per-tenant rate limits.

## Architecture (high level)
```
Event streams (Usage/Billing/Support) -> Raw lake (parquet, daily partitions)
                                            |
                               Airflow feature DAG (Great Expectations)
                                            |
                    Feature Store (Redis for hot, parquet for cold/backup)
                     |                                |
      Batch scorer (Airflow)               Online scorer (FastAPI + model server)
                     |                                |
        Scores to warehouse + S3            POST /predict_churn (UI/API)
                     |                                |
          CS CRM (Salesforce tasks)      Alerting + CS dashboard (Grafana/Metabase)
```

## Data model and contracts
- Tables: `usage_events` (tenant_id, user_id, event_type, ts, metadata json); `billing_events`
  (tenant_id, amount, status, ts); `support_tickets` (tenant_id, sentiment, channel, ts).
- Contracts: schemas versioned; additive only; partition by `date`; late data allowed to T-2 with
  watermark. Quality gates: null ratios, row deltas +/-15%, categorical domain checks.
- PII: tokens at ingestion; only hashed IDs in feature store; retention raw 180d, aggregates 2y.

## Feature store
- Hot path: Redis keyed by `tenant_id` with TTL 36h; cache warmers for top 5% accounts.
- Cold storage: parquet snapshots (daily) for reproducibility and backfills.
- Feature metadata: version, source, freshness timestamp, owner; SHAP top drivers stored per score.

## Training and retraining
- Time-series CV with 5 splits; optimized for recall at precision >=0.7.
- Weekly retrain on last 12 months; drift gate on AUC drop >5% or PSI >0.2 for top features.
- Model registry: stage -> shadow -> prod; promotion requires offline metrics + 24h shadow parity.
- Rollback: flip traffic to previous model; clear Redis keys that include new feature versions.

## Inference
- Batch: Airflow task pulls latest parquet features, scores all tenants, writes to warehouse + S3,
  and posts top drivers to CRM.
- Online: FastAPI service loads model and Redis client; if cache miss, compute features on the fly
  from last 30d slice in warehouse (bounded to 300 ms).

**API contract**
```http
POST /predict_churn
{
  "tenant_id": 12345,
  "force_refresh": false
}

Response 200
{
  "tenant_id": 12345,
  "churn_risk": 0.41,
  "top_drivers": [
    {"feature": "login_frequency_30d", "shap": -0.18},
    {"feature": "failed_payment_attempts", "shap": 0.12}
  ],
  "feature_freshness_min": 18
}
```

## Interventions
- Rules engine: thresholds (high >0.7 immediate AM call; medium 0.35-0.7 email + tips; low monitor).
- Outputs: Salesforce task with risk score + top drivers; SendGrid template for automated emails.
- Throttling: max 2 proactive touches per tenant per 14 days; dedupe within 24h window.

## Observability
- Metrics: AUC/PR by cohort, feature lag, batch completeness, API latency, intervention send rate,
  alert fatigue (tasks per CSM per day).
- Logs: structured JSON with request_id and model_version; traces via OpenTelemetry to Jaeger.
- Alerts: PagerDuty if p99 >150 ms for 15 minutes or freshness >30 minutes;
  Slack if batch SLA at risk.

## Scalability and cost
- Online service: HPA on CPU (target 60%) and p95 latency; 2-6 pods; autoscale Redis with max memory
  alarm at 70% to avoid evictions.
- Batch: windowed to finish in <90 minutes; spot instances for feature build; cost target < $65/day.
- Caching: top tenants pre-warmed; long-tail served from cold path with rate limiting 50 RPS.

## Testing and validation
- Contract tests for schemas and feature expectations; golden datasets for model output regression.
- Load tests for API (locust) to 200 RPS; failure budget used to set autoscale floor.
- Chaos drills: Redis eviction, late data, CRM outage; verified graceful degradation paths.

## Ops runbook (abridged)
- Late data: alert from row-count delta; rerun DAG with backfill flag; re-score affected tenants.
- Bad model deploy: drop in AUC_7d >5% triggers rollback and clears new feature keys.
- Redis cold start/evictions: warm caches; serve long-tail tenants from batch scores temporarily.
- Salesforce API limits: queue tasks in SQS, retry with jitter; if >1h backlog, page CS lead.

## Links
- Case study narrative: `case_studies/01_churn_prediction.md`
- Architecture diagram: `docs/architecture_diagrams/churn_prediction.md`.
