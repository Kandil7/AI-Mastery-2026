"""Generate example payloads for case study scenarios.

Usage:
  python research/week5-backend/week5_backend/scripts/case_study_payloads.py
"""

from __future__ import annotations

import json


def _print(title: str, payload: dict) -> None:
    print("\n" + title)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def main() -> None:
    _print(
        "Customer Support RAG",
        {
            "tenant_id": "tenant-123",
            "question": "How do I reset MFA for a user?",
            "filters": {"plan_tier": "enterprise"},
            "top_k": 8,
            "mode": "hybrid",
        },
    )
    _print(
        "Compliance & Legal RAG",
        {
            "tenant_id": "legal-team",
            "question": "What is the data retention policy for EU customers?",
            "filters": {"doc_type": "policy"},
            "top_k": 6,
            "mode": "vector",
        },
    )
    _print(
        "Engineering Knowledge Base",
        {
            "tenant_id": "eng",
            "question": "How do I use the new AuthClient in v2?",
            "filters": {"doc_type": "rfc"},
            "top_k": 10,
            "mode": "hybrid",
        },
    )
    _print(
        "Sales Enablement",
        {
            "tenant_id": "sales",
            "question": "What are the key objections for mid-market retail?",
            "filters": {"segment": "mid-market"},
            "top_k": 5,
            "mode": "vector",
        },
    )
    _print(
        "Incident Response",
        {
            "tenant_id": "sre",
            "question": "Database latency spike: what runbook should I follow?",
            "filters": {"doc_type": "runbook"},
            "top_k": 6,
            "mode": "hybrid",
        },
    )
    _print(
        "HR Policy",
        {
            "tenant_id": "hr",
            "question": "What is the parental leave policy for 2025 hires?",
            "filters": {"doc_type": "policy"},
            "top_k": 6,
            "mode": "vector",
        },
    )


if __name__ == "__main__":
    main()
