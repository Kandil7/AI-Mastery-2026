# src/chunking/sanitize.py
from __future__ import annotations

import re


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_NULL_BYTES = re.compile(r"\x00+")


def sanitize_text(
    text: str,
    *,
    max_chars: int,
    strip_control_chars: bool = True,
    normalize_newlines: bool = True,
) -> str:
    """
    Basic hardening:
      - cap size to prevent memory blowups
      - strip control chars (optional)
      - normalize newlines to '\n' (optional)
      - remove null bytes
    """
    if not isinstance(text, str):
        text = str(text)

    # Cap size early (memory safety)
    if len(text) > max_chars:
        text = text[:max_chars]

    # Normalize newlines
    if normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null bytes
    text = _NULL_BYTES.sub("", text)

    # Strip control chars
    if strip_control_chars:
        text = _CONTROL_CHARS.sub("", text)

    return text