# src/chunking/tokenizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, List, Optional


class TokenCounter(Protocol):
    """
    Abstraction to count units and to slice by unit boundaries if possible.
    Implementations:
      - TiktokenCounter (accurate for OpenAI models if tiktoken installed)
      - CharCounter (fallback)
    """
    def count(self, text: str) -> int: ...
    def slice_by_units(self, text: str, start_unit: int, end_unit: int) -> str: ...


@dataclass(frozen=True)
class CharCounter:
    def count(self, text: str) -> int:
        return len(text)

    def slice_by_units(self, text: str, start_unit: int, end_unit: int) -> str:
        return text[start_unit:end_unit]


class TiktokenCounter:
    """
    Optional accurate tokenization if `tiktoken` is available.
    """
    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken  # type: ignore
        except Exception as e:
            raise RuntimeError("tiktoken is not available; install it or use CharCounter") from e
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def slice_by_units(self, text: str, start_unit: int, end_unit: int) -> str:
        toks = self._enc.encode(text)
        sub = toks[start_unit:end_unit]
        return self._enc.decode(sub)


def build_counter(prefer_tiktoken: bool = True, encoding_name: str = "cl100k_base") -> TokenCounter:
    if prefer_tiktoken:
        try:
            return TiktokenCounter(encoding_name=encoding_name)
        except Exception:
            return CharCounter()
    return CharCounter()
