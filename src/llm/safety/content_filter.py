"""
Content Filter Module
=====================

Content filtering for LLM outputs.

Classes:
    ContentCategory: Categories of content to filter
    ContentRating: Content safety rating
    FilterConfig: Configuration for content filtering
    ContentFilter: Main content filtering class

Author: AI-Mastery-2026
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ContentCategory(Enum):
    """Categories of content to filter."""

    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    HARASSMENT = "harassment"
    EXTREME = "extreme"
    ILLEGAL = "illegal"
    SPAM = "spam"
    PII = "pii"


class ContentRating(Enum):
    """Content safety rating."""

    SAFE = "safe"
    ADVISORY = "advisory"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class FilterConfig:
    """Configuration for content filtering."""

    # Enable specific categories
    check_violence: bool = True
    check_hate_speech: bool = True
    check_sexual: bool = True
    check_self_harm: bool = True
    check_harassment: bool = True
    check_illegal: bool = True
    check_pii: bool = True

    # Action on detection
    block_on_detection: bool = True
    replace_with_placeholder: bool = False
    placeholder_text: str = "[FILTERED]"

    # Sensitivity (0.0 to 1.0)
    sensitivity: float = 0.7

    # Custom keywords to block
    custom_blocked: Set[str] = field(default_factory=set)
    custom_warning: Set[str] = field(default_factory=set)


@dataclass
class FilterResult:
    """
    Result of content filtering.

    Attributes:
        is_safe: Whether content passed all filters
        rating: Content safety rating
        categories: Triggered content categories
        confidence: Confidence score (0-1)
        filtered_content: Filtered content if modified
        details: Additional details
    """

    is_safe: bool
    rating: ContentRating = ContentRating.SAFE
    categories: List[ContentCategory] = field(default_factory=list)
    confidence: float = 1.0
    filtered_content: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set rating based on categories."""
        if not self.is_safe:
            if any(
                c in [ContentCategory.ILLEGAL, ContentCategory.SELF_HARM]
                for c in self.categories
            ):
                self.rating = ContentRating.BLOCKED
            elif any(
                c in [ContentCategory.VIOLENCE, ContentCategory.HATE_SPEECH]
                for c in self.categories
            ):
                self.rating = ContentRating.WARNING
            else:
                self.rating = ContentRating.ADVISORY


class ContentFilter:
    """
    Content filter for LLM outputs.

    Filters various categories of potentially harmful or unwanted content.

    Example:
        >>> filter = ContentFilter()
        >>> result = filter.check("This is a test message")
        >>> print(result.is_safe)  # True
    """

    # Keyword patterns for each category
    PATTERNS: Dict[ContentCategory, List[str]] = {
        ContentCategory.VIOLENCE: [
            r"\bkill\b.*\bperson\b",
            r"\bmurder\b",
            r"\battack\b.*\bweapon\b",
            r"\bbomb\b",
            r"\bterrorist\b",
            r"\bdestroy\b.*\bhuman\b",
        ],
        ContentCategory.HATE_SPEECH: [
            r"\bhate\b.*\b(black|white|asian|jew|muslim|christian|gay|lesbian)\b",
            r"\bnigger\b",
            r"\bfaggot\b",
            r"\bwhite\s+supremac",
            r"\bnazi\b",
            r"\bgenocide\b",
        ],
        ContentCategory.SEXUAL: [
            r"\bsex\b.*\b(oral|anal|positional)\b",
            r"\bporn\b",
            r"\bxxx\b",
            r"\bnude\b",
            r"\bexplicit\b.*\bsexual\b",
        ],
        ContentCategory.SELF_HARM: [
            r"\bsuicide\b",
            r"\bself[-\s]harm\b",
            r"\bkill\s+(?:yourself|myself|oneself)\b",
            r"\bcut\s+(?:yourself|myself)\b",
        ],
        ContentCategory.HARASSMENT: [
            r"\bstupid\b.*\bidiot\b",
            r"\bdie\b.*\b(you|everyone)\b",
            r"\bcurse\b.*\bword\b",
        ],
        ContentCategory.ILLEGAL: [
            r"\bdrug\b.*\b(sell|buy|deal)\b",
            r"\bfraud\b",
            r"\bscam\b",
            r"\bhack\b.*\bpassword\b",
            r"\bstolen\b.*\bcredit\b",
        ],
        ContentCategory.SPAM: [
            r"\b(click\s+here|win\s+now|free\s+money)\b",
            r"\bbuy\s+now\b.*\blimited\b",
            r"\bact\s+now\b.*\boffer\b",
        ],
        ContentCategory.PII: [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+\d{3,4}\b",  # Name + SSN-like
        ],
    }

    def __init__(self, config: Optional[FilterConfig] = None):
        """Initialize content filter."""
        self.config = config or FilterConfig()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._compiled_patterns: Dict[ContentCategory, List[re.Pattern]] = {}

        for category, patterns in self.PATTERNS.items():
            compiled = []
            for p in patterns:
                try:
                    compiled.append(re.compile(p, re.IGNORECASE | re.MULTILINE))
                except re.error as e:
                    logger.warning(f"Invalid pattern in {category}: {e}")
            self._compiled_patterns[category] = compiled

    def check(self, text: str) -> FilterResult:
        """
        Check content for violations.

        Args:
            text: Content to check

        Returns:
            FilterResult with filtering details
        """
        categories = []
        details = {}
        filtered = text
        confidence = 1.0

        # Check each category
        if self.config.check_violence:
            result = self._check_category(text, ContentCategory.VIOLENCE)
            if result["detected"]:
                categories.append(ContentCategory.VIOLENCE)
                details["violence"] = result
                confidence = min(confidence, 0.9)

        if self.config.check_hate_speech:
            result = self._check_category(text, ContentCategory.HATE_SPEECH)
            if result["detected"]:
                categories.append(ContentCategory.HATE_SPEECH)
                details["hate_speech"] = result
                confidence = min(confidence, 0.95)

        if self.config.check_sexual:
            result = self._check_category(text, ContentCategory.SEXUAL)
            if result["detected"]:
                categories.append(ContentCategory.SEXUAL)
                details["sexual"] = result
                confidence = min(confidence, 0.85)

        if self.config.check_self_harm:
            result = self._check_category(text, ContentCategory.SELF_HARM)
            if result["detected"]:
                categories.append(ContentCategory.SELF_HARM)
                details["self_harm"] = result
                confidence = min(confidence, 0.95)

        if self.config.check_harassment:
            result = self._check_category(text, ContentCategory.HARASSMENT)
            if result["detected"]:
                categories.append(ContentCategory.HARASSMENT)
                details["harassment"] = result
                confidence = min(confidence, 0.8)

        if self.config.check_illegal:
            result = self._check_category(text, ContentCategory.ILLEGAL)
            if result["detected"]:
                categories.append(ContentCategory.ILLEGAL)
                details["illegal"] = result
                confidence = min(confidence, 0.9)

        if self.config.check_pii:
            result = self._check_category(text, ContentCategory.PII)
            if result["detected"]:
                categories.append(ContentCategory.PII)
                details["pii"] = result
                confidence = min(confidence, 0.95)

        # Apply sensitivity adjustment
        confidence *= self.config.sensitivity

        # Determine if content is safe
        is_safe = len(categories) == 0
        if is_safe and self.config.block_on_detection:
            is_safe = True

        # Apply filtering if configured
        if not is_safe and self.config.replace_with_placeholder:
            filtered = self._apply_filter(text, categories)

        return FilterResult(
            is_safe=is_safe,
            rating=ContentRating.SAFE if is_safe else ContentRating.WARNING,
            categories=categories,
            confidence=confidence,
            filtered_content=filtered if filtered != text else None,
            details=details,
        )

    def _check_category(self, text: str, category: ContentCategory) -> Dict[str, Any]:
        """Check text for a specific category."""
        patterns = self._compiled_patterns.get(category, [])
        matches = []

        for pattern in patterns:
            match = pattern.search(text)
            if match:
                matches.append(match.group())

        return {
            "detected": len(matches) > 0,
            "matches": matches,
            "match_count": len(matches),
        }

    def _apply_filter(self, text: str, categories: List[ContentCategory]) -> str:
        """Apply filtering by replacing problematic content."""
        # For now, replace entire text with placeholder
        # More sophisticated filtering could target specific matches
        return self.config.placeholder_text

    def check_batch(self, texts: List[str]) -> List[FilterResult]:
        """Check multiple texts."""
        return [self.check(text) for text in texts]


# Quick filter function
def quick_filter(text: str) -> bool:
    """
    Quick content filter.

    Args:
        text: Content to check

    Returns:
        True if safe, False if blocked
    """
    filter = ContentFilter()
    result = filter.check(text)
    return result.is_safe
