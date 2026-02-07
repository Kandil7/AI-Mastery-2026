# Churn Prediction Architecture (B2B SaaS)

```mermaid
flowchart LR
    subgraph Ingestion
        U[Usage events] --> DL[Data Lake (parquet, daily)]
        B[Billing events] --> DL
        S[Support tickets] --> DL
    end

    subgraph FeaturePipeline
        DL --> GE[Airflow + Great Expectations]
        GE --> FS[Feature Store\nRedis (hot) + Parquet (cold)]
    end

    subgraph Training
        FS --> TR[Weekly retrain\nXGBoost]
        TR --> MR[Model Registry]
    end

    subgraph Inference
        FS --> BS[Batch scoring (Airflow)]
        MR --> OS[Online Scoring (FastAPI)]
        FS --> OS
    end

    BS --> WH[Warehouse + S3]
    BS --> CRM[Salesforce tasks]
    OS --> CRM
    OS --> DASH[CS Dashboard]

    subgraph Observability
        OS --> MET[Metrics/Logs/Traces]
        BS --> MET
        MET --> PD[Alerts (PagerDuty/Slack)]
    end
```

Key SLOs: p99 online <150 ms; feature freshness <30 minutes; batch SLA 06:00 UTC.
