from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PlanStep:
    tool: str
    input: str


class Planner:
    def plan(self, question: str) -> List[PlanStep]:
        # Minimal heuristic: always start with RAG.
        return [PlanStep(tool="rag", input=question)]
