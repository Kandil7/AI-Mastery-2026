from __future__ import annotations

from typing import List

from agents.planner import Planner
from agents.tools import ToolRegistry


class AgentExecutor:
    def __init__(self, planner: Planner, tools: ToolRegistry) -> None:
        self._planner = planner
        self._tools = tools

    def run(self, question: str) -> str:
        steps = self._planner.plan(question)
        outputs: List[str] = []
        for step in steps:
            outputs.append(self._tools.run(step.tool, step.input))
        return "\n".join(outputs)
