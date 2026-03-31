"""
Arabic LLM - Text Processing Utilities

General text processing utilities for all text operations.
"""

import re
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text:
    - Multiple spaces → single space
    - Tabs → space
    - Remove leading/trailing whitespace

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def remove_control_chars(text: str, keep_newlines: bool = True) -> str:
    """
    Remove control characters from text.

    Args:
        text: Input text
        keep_newlines: Keep \n, \r, \t

    Returns:
        Text without control characters
    """
    if keep_newlines:
        # Keep \n, \r, \t
        return "".join(
            c for c in text 
            if c in "\n\r\t" or not c.isspace() or c.isprintable()
        )
    else:
        # Remove all control characters
        return "".join(c for c in text if c.isprintable())


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
    truncate_from_start: bool = False,
) -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length (including suffix)
        suffix: Suffix to add when truncated
        truncate_from_start: If True, truncate from start instead of end

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    if truncate_from_start:
        # Keep end, truncate from start
        return suffix + text[-(max_length - len(suffix)):]
    else:
        # Keep start, truncate from end
        return text[:max_length - len(suffix)] + suffix


def split_into_sentences(text: str) -> list:
    """
    Split Arabic text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Arabic sentence delimiters
    delimiters = r"[.!?۔]\s*"
    
    sentences = re.split(delimiters, text)
    
    # Filter empty sentences
    return [s.strip() for s in sentences if s.strip()]


def count_words(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Input text

    Returns:
        Word count
    """
    # Split on whitespace and filter empty
    words = [w for w in text.split() if w.strip()]
    return len(words)


def count_lines(text: str) -> int:
    """
    Count non-empty lines in text.

    Args:
        text: Input text

    Returns:
        Line count
    """
    lines = [line for line in text.split("\n") if line.strip()]
    return len(lines)


def extract_urls(text: str) -> list:
    """
    Extract URLs from text.

    Args:
        text: Input text

    Returns:
        List of URLs
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def remove_urls(text: str, replacement: str = "") -> str:
    """
    Remove URLs from text.

    Args:
        text: Input text
        replacement: Replacement string

    Returns:
        Text without URLs
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.sub(url_pattern, replacement, text)


def clean_text_for_training(
    text: str,
    min_length: int = 10,
    max_length: int = 2000,
    remove_urls_flag: bool = True,
) -> Optional[str]:
    """
    Clean text for training data:
    - Remove URLs
    - Normalize whitespace
    - Remove control characters
    - Check length constraints

    Args:
        text: Input text
        min_length: Minimum length
        max_length: Maximum length
        remove_urls_flag: Remove URLs

    Returns:
        Cleaned text or None if too short/long
    """
    # Remove URLs
    if remove_urls_flag:
        text = remove_urls(text)
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    # Remove control characters
    text = remove_control_chars(text)
    
    # Check length
    if len(text) < min_length or len(text) > max_length:
        return None
    
    return text
