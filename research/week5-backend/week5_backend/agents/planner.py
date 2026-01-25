from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PlanStep:
    tool: str
    input: str


class Planner:
    def plan(self, question: str) -> List[PlanStep]:
        question_lower = question.lower()
        steps: List[PlanStep] = []
        if "sql:" in question_lower or "database" in question_lower:
            steps.append(PlanStep(tool="sql", input=question))
        if "web:" in question_lower or "external" in question_lower:
            steps.append(PlanStep(tool="web", input=question))
        steps.append(PlanStep(tool="rag", input=question))
        return steps
