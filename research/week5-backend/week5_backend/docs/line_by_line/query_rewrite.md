# Line-by-line explanation: query_rewrite.py

File: `research/week5-backend/week5_backend/rag/query_rewrite.py`

- L1: `from __future__ import annotations` -> defers evaluation of type hints.
- L2: (blank) -> visual separation.
- L3: `from typing import Optional` -> optional type hint.
- L4: (blank) -> spacing before local import.
- L5: `from providers.llm_base import LLMProvider` -> LLM provider interface.
- L6: (blank) -> spacing before class.
- L7: (blank) -> extra spacing.
- L8: `class QueryRewriter:` -> encapsulates query rewrite logic.
- L9: `def __init__(...)` -> constructor.
- L10: `self._provider = provider` -> store optional provider.
- L11: (blank) -> spacing before method.
- L12: `def rewrite(self, question: str) -> str:` -> rewrite a question to a search query.
- L13: `if not self._provider:` -> if rewrite disabled.
- L14: `return question` -> no rewrite, return original question.
- L15: `prompt = (` -> build rewrite prompt.
- L16: string line -> instruct to keep intent and key entities.
- L17: string line -> ask for rewritten query only.
- L18: `f"Question: {question}\n"` -> include original question.
- L19: `"Query:"` -> prompt expects a query output.
- L20: `)` -> close prompt.
- L21: `rewritten = self._provider.generate(prompt).strip()` -> call LLM and trim output.
- L22: `return rewritten or question` -> fallback to original if empty.
