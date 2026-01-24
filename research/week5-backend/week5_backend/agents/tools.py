from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass(frozen=True)
class Tool:
    name: str
    handler: Callable[[str], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def run(self, name: str, input_text: str) -> str:
        tool = self._tools[name]
        return tool.handler(input_text)
