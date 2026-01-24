from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Trace:
    trace_id: str
    attributes: Dict[str, str]


def start_trace(trace_id: str, attributes: Dict[str, str] | None = None) -> Trace:
    return Trace(trace_id=trace_id, attributes=attributes or {})
