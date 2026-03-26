"""
Arabic LLM - Arabic Text Utilities

Utilities for processing Arabic text:
- Character counting
- Diacritics detection
- Text normalization
"""

import re
from typing import Tuple


def count_arabic_chars(text: str) -> int:
    """
    Count Arabic characters in text (Unicode range 0600-06FF).

    Args:
        text: Input text

    Returns:
        Count of Arabic characters
    """
    return sum(1 for c in text if "\u0600" <= c <= "\u06FF")


def count_diacritics(text: str) -> int:
    """
    Count Arabic diacritics (Tashkeel) in text.

    Args:
        text: Input text

    Returns:
        Count of diacritics
    """
    diacritics_range = [
        ("\u064B", "\u0652"),  # Fathatan to Sukun
        ("\u0670", "\u0670"),  # Superscript Alef
    ]

    count = 0
    for c in text:
        for start, end in diacritics_range:
            if start <= c <= end:
                count += 1
                break
    return count


def get_arabic_ratio(text: str) -> float:
    """
    Calculate ratio of Arabic characters to total characters.

    Args:
        text: Input text

    Returns:
        Ratio (0.0 to 1.0)
    """
    if not text:
        return 0.0

    arabic_count = count_arabic_chars(text)
    total_count = len(text)

    return arabic_count / total_count


def get_diacritics_ratio(text: str) -> float:
    """
    Calculate ratio of diacritics to Arabic characters.

    Args:
        text: Input text

    Returns:
        Ratio (0.0 to 1.0)
    """
    arabic_count = count_arabic_chars(text)
    if arabic_count == 0:
        return 0.0

    diacritics_count = count_diacritics(text)
    return diacritics_count / arabic_count


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text:
    - Unify Alif forms (أ, إ, آ → ا)
    - Unify Alif Maqsura (ى → ي)
    - Unify Ta Marbuta (ة → ه)
    - Unify Hamza combinations

    Args:
        text: Input Arabic text

    Returns:
        Normalized Arabic text
    """
    # Unify Alif forms
    text = re.sub(r"[أإآ]", "ا", text)

    # Unify Alif Maqsura
    text = re.sub(r"ى", "ي", text)

    # Unify Ta Marbuta
    text = re.sub(r"ة", "ه", text)

    # Unify Hamza on Waw/Ya
    text = re.sub(r"ؤ", "ءو", text)
    text = re.sub(r"ئ", "ءي", text)

    return text


def remove_tashkeel(text: str) -> str:
    """
    Remove all diacritics (Tashkeel) from Arabic text.

    Args:
        text: Input Arabic text

    Returns:
        Text without diacritics
    """
    tashkeel_pattern = r"[\u064B-\u0652\u0670]"
    return re.sub(tashkeel_pattern, "", text)


def is_arabic_text(text: str, threshold: float = 0.7) -> bool:
    """
    Check if text is primarily Arabic.

    Args:
        text: Input text
        threshold: Minimum Arabic ratio (default: 0.7)

    Returns:
        True if text is primarily Arabic
    """
    return get_arabic_ratio(text) >= threshold


def extract_arabic_text(text: str) -> str:
    """
    Extract only Arabic characters from text.

    Args:
        text: Input text

    Returns:
        Arabic characters only
    """
    return "".join(c for c in text if "\u0600" <= c <= "\u06FF")
