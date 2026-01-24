from __future__ import annotations

from dataclasses import dataclass
from typing import List

from agents.planner import Planner
from agents.tools import ToolRegistry, ToolResult


@dataclass(frozen=True)
class AgentResult:
    output: str
    citations: List[dict]


class AgentExecutor:
    def __init__(self, planner: Planner, tools: ToolRegistry) -> None:
        self._planner = planner
        self._tools = tools

    def run(self, question: str) -> AgentResult:
        steps = self._planner.plan(question)
        outputs: List[str] = []
        citations: List[dict] = []
        for step in steps:
            result: ToolResult = self._tools.run(step.tool, step.input)
            outputs.append(result.output)
            citations.extend(result.citations)
        return AgentResult(output="\n".join(outputs), citations=citations)
