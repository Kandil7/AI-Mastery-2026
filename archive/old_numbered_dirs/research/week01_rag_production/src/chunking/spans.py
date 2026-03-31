# src/chunking/spans.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TextSpan:
    """
    Represents a slice from the original text by character offsets.
    Offsets are ALWAYS in original document characters.
    """
    start: int
    end: int

    def clamp(self, n: int) -> "TextSpan":
        s = max(0, min(self.start, n))
        e = max(0, min(self.end, n))
        if e < s:
            e = s
        return TextSpan(s, e)


@dataclass(frozen=True)
class ChunkSpan:
    """
    A chunk payload with exact source offsets.
    """
    text: str
    span: TextSpan