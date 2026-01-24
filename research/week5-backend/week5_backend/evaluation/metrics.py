from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    faithfulness: float
    citation_coverage: float


def compute_metrics(exact_matches: int, total: int) -> EvalResult:
    if total == 0:
        return EvalResult(accuracy=0.0, faithfulness=0.0, citation_coverage=0.0)
    accuracy = exact_matches / total
    return EvalResult(accuracy=accuracy, faithfulness=0.0, citation_coverage=0.0)
