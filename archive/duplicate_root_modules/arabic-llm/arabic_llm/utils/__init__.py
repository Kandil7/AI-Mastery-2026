"""
Arabic LLM - Utilities

This subpackage contains utility functions:
- Logging configuration
- I/O utilities
- Text processing utilities
- Arabic-specific utilities
"""

from .logging import (
    setup_logging,
    get_logger,
)

from .io import (
    read_jsonl,
    write_jsonl,
    read_json,
    write_json,
    read_yaml,
    read_yaml_config,
)

from .text import (
    normalize_whitespace,
    remove_control_chars,
    truncate_text,
)

from .arabic import (
    count_arabic_chars,
    count_diacritics,
    get_arabic_ratio,
    normalize_arabic_text,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    # I/O
    "read_jsonl",
    "write_jsonl",
    "read_json",
    "write_json",
    "read_yaml",
    "read_yaml_config",
    # Text
    "normalize_whitespace",
    "remove_control_chars",
    "truncate_text",
    # Arabic
    "count_arabic_chars",
    "count_diacritics",
    "get_arabic_ratio",
    "normalize_arabic_text",
]
