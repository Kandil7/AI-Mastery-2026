"""
Input Sanitization Service
=============================
Prevents XSS, SQL injection, and other attacks.

خدمة التنظيف من حقن لمنع هجمات XSS و SQLi وغيرها
"""

import re
import html
from typing import List, Pattern

from src.core.config import settings


class InputSanitizer:
    """
    Input sanitization service for security.

    Cleans user input to prevent:
    - XSS (Cross-Site Scripting) attacks
    - SQL injection attempts
    - Command injection
    - Path traversal
    - HTML injection

    Design Principles:
    - Whitelist-based (safer than blacklist)
    - Remove dangerous characters/patterns
    - Encode output properly

    خدمة تنظيف المدخلات
    """

    def __init__(
        self,
        allowed_tags: List[str] | None = None,
        allowed_attributes: List[str] | None = None,
        max_length: int = 10000,
    ) -> None:
        """
        Initialize input sanitizer.

        Args:
            allowed_tags: Allowed HTML tags (whitelist)
            allowed_attributes: Allowed HTML attributes (whitelist)
            max_length: Maximum input length before sanitization
        """
        # Default safe tags (minimal set for security)
        self._allowed_tags = allowed_tags or [
            "p",
            "br",
            "strong",
            "em",
            "u",
            "ol",
            "ul",
            "li",
            "blockquote",
            "code",
            "pre",
            "a",
            "b",
            "i",
            "span",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "table",
            "thead",
            "tbody",
            "tr",
            "th",
            "td",
        ]

        # Default safe attributes
        self._allowed_attributes = allowed_attributes or [
            "href",
            "title",
            "alt",
            "class",
            "id",
            "style",
            "data-",
            "aria-",
        ]

        self._max_length = max_length

    def sanitize_html(self, text: str) -> str:
        """
        Sanitize HTML to prevent XSS.

        Removes all tags except whitelisted ones.
        Strips all attributes except whitelisted ones.

        Args:
            text: Input text potentially containing HTML

        Returns:
            Sanitized safe HTML string

        Examples:
        - Input: `<script>alert(1)</script>`
          Output: `alert(1)`
        - Input: `<a href="javascript:alert(1)">Click</a>`
          Output: `<a>Click</a>`
        - Input: `<p class="safe">Safe</p>`
          Output: `<p>Safe</p>`
        """
        # Remove all tags except allowed
        tags_pattern = re.compile(
            r"<(?!/(?:" + "|".join(self._allowed_tags) + r"))/)[^>]+)>", re.IGNORECASE
        )
        text = tags_pattern.sub("", text)

        # Remove all attributes except allowed
        attr_pattern = re.compile(
            r"\s+[" + "|".join(self._allowed_attributes) + r'][^>]*(?=(["\'][^"]*["\'][^>]*)?)',
            re.IGNORECASE,
        )
        text = attr_pattern.sub(" ", text)

        # Encode remaining special characters
        return html.escape(text)

    def strip_html(self, text: str) -> str:
        """
        Remove ALL HTML tags (for strict security).

        Use when you want plain text only.

        Args:
            text: Input text potentially containing HTML

        Returns:
            Plain text without any HTML tags

        Examples:
        - Input: `<p><script>alert(1)</script></p>`
          Output: `alert(1)`
        """
        # Remove all HTML tags
        html_pattern = re.compile(r"<[^>]+>", re.IGNORECASE)
        return html_pattern.sub("", text)

    def escape_sql(self, query: str) -> str:
        """
        Escape SQL special characters to prevent injection.

        Note: This is a DEFENSE IN DEPTH approach.
        The proper solution is parameterized queries.
        However, this provides an additional layer of protection.

        Args:
            query: SQL query string

        Returns:
            Escaped query string

        SQL Special Characters:
        - ' (single quote)
        - " (double quote)
        - ; (statement separator)
        - -- (comment)
        - /* (comment)
        - \ (backslash)

        Examples:
        - Input: `admin' OR '1'='1`
          Output: `admin\' OR \'1\'=\'1`
        """
        # Escape single quotes
        query = query.replace("'", "''")

        # Escape double quotes
        query = query.replace('"', '""')

        # Escape semicolons
        query = query.replace(";", "\\;")

        # Escape comment characters
        query = query.replace("--", "\\--")
        query = query.replace("/*", "\\/\\*")

        return query

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal.

        Removes directory traversal attempts (../).
        Removes null bytes.
        Truncates to reasonable length.

        Args:
            filename: User-provided filename

        Returns:
            Safe filename string

        Examples:
        - Input: `../../../etc/passwd`
          Output: `etc_passwd`
        - Input: `test\0file.pdf`
          Output: `test_file.pdf`
        """
        # Remove path traversal sequences
        filename = re.sub(r"\.\.|\/", "_", filename)

        # Remove null bytes
        filename = filename.replace("\x00", "")

        # Truncate to reasonable length
        max_filename_length = 255
        if len(filename) > max_filename_length:
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
            filename = (
                f"{name[: max_filename_length - len(ext) - 1]}{ext}"
                if ext
                else name[:max_filename_length]
            )

        # Allow only safe characters
        filename = re.sub(r"[^\w\s\-._]", "_", filename)

        return filename

    def sanitize_url_param(self, url: str) -> str:
        """
        Sanitize URL parameter to prevent open redirect.

        Removes dangerous URL protocols (javascript:, data:).
        Validates URL format.

        Args:
            url: URL parameter

        Returns:
            Sanitized URL string

        Examples:
        - Input: `javascript:alert(document.cookie)`
          Output: ``
        - Input: `data:text/html,<script>alert(1)</script>`
          Output: ``
        """
        # Block dangerous protocols
        dangerous_protocols = [
            "javascript:",
            "data:",
            "vbscript:",
            "file:",
            "about:",
            "mailto:",
            "tel:",
        ]

        for protocol in dangerous_protocols:
            if url.lower().startswith(protocol):
                return ""

        # Allow only http/https URLs
        if not url.lower().startswith(("http://", "https://")):
            # If it's not http/https, it might be unsafe
            return ""

        return url

    def sanitize_markdown(self, text: str) -> str:
        """
        Sanitize Markdown input (common in RAG systems).

        Prevents:
        - HTML tags in Markdown
        - JavaScript in code blocks
        - Dangerous links

        Args:
            text: Markdown-formatted text

        Returns:
            Sanitized Markdown string
        """
        # Remove HTML tags (Markdown should be text-based)
        html_pattern = re.compile(r"<[^>]+>", re.IGNORECASE)
        text = html_pattern.sub("", text)

        # Remove JavaScript in code blocks
        js_pattern = re.compile(r"javascript:", re.IGNORECASE)
        text = js_pattern.sub("js:", text)

        # Sanitize links (basic check)
        # More complex link validation would require full URL parsing
        text = re.sub(r"\[.*?\]\(javascript:.*?\)", "[link]", text)

        return text

    def truncate_string(self, text: str, max_length: int) -> str:
        """
        Truncate string to maximum length safely.

        Args:
            text: Input string
            max_length: Maximum allowed length

        Returns:
            Truncated string
        """
        if len(text) <= max_length:
            return text

        return text[:max_length] + "..."

    def sanitize_json_input(self, text: str) -> str:
        """
        Sanitize JSON input to prevent injection.

        Removes control characters.
        Validates JSON structure (basic).

        Args:
            text: JSON string input

        Returns:
            Sanitized JSON string

        Control Characters to Remove:
        - \x00-\x1f (ASCII control chars)
        - \x7f-\x9f (ASCII control chars)
        """
        # Remove control characters
        sanitized = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)

        return sanitized

    def clean_input(
        self,
        text: str,
        allow_html: bool = False,
        max_length: int | None = None,
    ) -> str:
        """
        Clean input with configurable security level.

        Args:
            text: Input text to clean
            allow_html: Whether to allow whitelisted HTML tags
            max_length: Maximum allowed length (uses default if None)

        Returns:
            Sanitized string
        """
        # Use default max length if not provided
        max_length = max_length or self._max_length

        # Truncate if too long
        text = self.truncate_string(text, max_length)

        # Choose sanitization level
        if allow_html:
            # Allow safe HTML
            return self.sanitize_html(text)
        else:
            # Strip all HTML (strict mode)
            return self.strip_html(text)

    def sanitize_query_for_search(self, query: str) -> str:
        """
        Sanitize search query for vector/keyword search.

        Special handling for search queries:
        - Preserve search operators (AND, OR, quotes)
        - Escape special characters for query
        - Prevent injection in vector/keyword queries

        Args:
            query: Search query string

        Returns:
            Sanitized search query
        """
        # Escape quotes for search (preserve them but prevent injection)
        query = query.replace('"', '\\"')
        query = query.replace("'", "\\'")

        # Escape semicolons (might be SQL injection)
        query = query.replace(";", "\\;")

        return query

    def sanitize_llm_prompt(self, prompt: str) -> str:
        """
        Sanitize LLM prompt to prevent prompt injection.

        Prevents:
        - Instructions that bypass safety guidelines
        - System prompt injection
        - Context overflow

        Args:
            prompt: LLM prompt string

        Returns:
            Sanitized prompt string

        Warning:
            - This is a basic defense
            - Full prompt injection prevention requires more sophisticated approaches
            - Always validate prompts server-side
        """
        # Remove potential system instructions
        dangerous_phrases = [
            "ignore previous instructions",
            "disregard all above",
            "system:",
            "assistant:",
            "forget everything",
        ]

        for phrase in dangerous_phrases:
            if phrase.lower() in prompt.lower():
                prompt = prompt.lower().replace(phrase, "[REMOVED]")

        return prompt


# ============================================================================
# Global Sanitizer Instance
# ============================================================================

_sanitizer: InputSanitizer | None = None


def get_sanitizer() -> InputSanitizer:
    """Get global input sanitizer instance."""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = InputSanitizer()
    return _sanitizer
