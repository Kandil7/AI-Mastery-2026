# src/chunking/analyze.py
from __future__ import annotations

import re
from typing import Optional

from src.retrieval import Document
from .config import ChunkingStrategy


_CODE_FENCE = re.compile(r"```|~~~")
_MD_HEADERS = re.compile(r"(?m)^\s*#{1,6}\s+")
_MD_TABLE = re.compile(r"(?m)^\s*\|.+\|\s*$")
_PY_HINTS = re.compile(r"(?m)^\s*(def|class)\s+\w+|^\s*from\s+\w+|^\s*import\s+\w+")

_ARABIC_Q = "؟"


def choose_strategy(document: Document) -> ChunkingStrategy:
    content = (document.content or "")
    doc_type = (document.doc_type or "").lower()

    if "code" in doc_type or "program" in doc_type:
        return ChunkingStrategy.CODE
    if "markdown" in doc_type or doc_type.endswith(".md") or doc_type == "md":
        return ChunkingStrategy.MARKDOWN

    # Content-based signals
    lower = content.lower()
    code_score = 0
    md_score = 0

    if _CODE_FENCE.search(content):
        md_score += 3
        code_score += 2  # fenced code is both markdown and code-ish

    if _MD_HEADERS.search(content):
        md_score += 3
    if _MD_TABLE.search(content):
        md_score += 2

    # Lightweight code detection
    if _PY_HINTS.search(content):
        code_score += 3
    for kw in ("if __name__", "public ", "private ", "const ", "let ", "function "):
        if kw in lower:
            code_score += 1

    # Paragraph signal
    para_breaks = content.count("\n\n")
    word_count = max(1, len(content.split()))
    para_density = para_breaks / max(1, word_count / 120)

    # Decide
    if code_score >= 4 and code_score >= md_score:
        return ChunkingStrategy.CODE
    if md_score >= 4 and md_score >= code_score:
        return ChunkingStrategy.MARKDOWN
    if para_density >= 1.0:
        return ChunkingStrategy.PARAGRAPH

    return ChunkingStrategy.SEMANTIC