from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class Tool:
    name: str
    handler: Callable[[str], "ToolResult"]


@dataclass(frozen=True)
class ToolResult:
    output: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def run(self, name: str, input_text: str) -> ToolResult:
        tool = self._tools[name]
        return tool.handler(input_text)
