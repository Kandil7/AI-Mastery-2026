from __future__ import annotations

import uuid

from evaluation.datasets import load_dataset
from evaluation.metrics import compute_metrics
from pipelines.online_query import run_query_pipeline


def run_evaluation(dataset_id: str, mode: str) -> str:
    dataset = load_dataset(dataset_id)
    exact_matches = 0
    for example in dataset:
        result = run_query_pipeline(
            tenant_id="eval",
            question=example.question,
            filters={},
            top_k=5,
            mode=mode,
        )
        if result["answer"].strip().lower() == example.answer.strip().lower():
            exact_matches += 1
    _ = compute_metrics(exact_matches=exact_matches, total=len(dataset))
    return str(uuid.uuid4())
