from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingPolicy:
    default_provider: str
    fallback_provider: str

    def choose(self, task: str) -> str:
        task = task.lower()
        if "summary" in task or "summarize" in task:
            return self.default_provider
        if "extract" in task or "classification" in task:
            return self.default_provider
        return self.fallback_provider or self.default_provider
