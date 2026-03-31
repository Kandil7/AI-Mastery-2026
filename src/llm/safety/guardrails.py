"""
Guardrails Module
=================

Input and output guardrails for LLM safety.

Classes:
    SafetyConfig: Configuration for safety checks
    SafetyResult: Result of safety evaluation
    InputGuardrails: Input validation and filtering
    OutputGuardrails: Output validation and filtering

Author: AI-Mastery-2026
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety level classification."""

    SAFE = "safe"
    WARNING = "warning"
    BLOCK = "block"


@dataclass
class SafetyConfig:
    """Configuration for safety guardrails."""

    # Input checks
    check_prompt_injection: bool = True
    check_sensitive_data: bool = True
    max_input_length: int = 10000

    # Output checks
    check_harmful_content: bool = True
    check_pii: bool = True
    block_on_safety: bool = True

    # Patterns
    blocked_patterns: List[str] = field(default_factory=list)
    warning_patterns: List[str] = field(default_factory=list)

    # Custom words
    blocked_words: Set[str] = field(default_factory=set)
    warning_words: Set[str] = field(default_factory=set)


@dataclass
class SafetyResult:
    """
    Result of safety evaluation.

    Attributes:
        is_safe: Whether the content passed safety checks
        confidence: Confidence score (0-1)
        categories: List of triggered safety categories
        filtered_content: If content was modified, the filtered version
        safety_level: Overall safety level
        details: Additional details about checks performed
    """

    is_safe: bool
    confidence: float = 1.0
    categories: List[str] = field(default_factory=list)
    filtered_content: Optional[str] = None
    safety_level: SafetyLevel = SafetyLevel.SAFE
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set safety level based on is_safe."""
        if not self.is_safe:
            self.safety_level = SafetyLevel.BLOCK


class InputGuardrails:
    """
    Input validation and filtering for LLM inputs.

    Provides:
    - Prompt injection detection
    - Sensitive data detection
    - Length validation
    - Pattern-based blocking

    Example:
        >>> guardrails = InputGuardrails()
        >>> result = guardrails.check("Ignore previous instructions...")
        >>> if not result.is_safe:
        ...     print(f"Blocked: {result.categories}")
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        """Initialize input guardrails with config."""
        self.config = config or SafetyConfig()
        self._blocked_patterns = self._compile_patterns(self.config.blocked_patterns)
        self._warning_patterns = self._compile_patterns(self.config.warning_patterns)

    def _compile_patterns(self, patterns: List[str]) -> List[re.Pattern]:
        """Compile regex patterns."""
        compiled = []
        for p in patterns:
            try:
                compiled.append(re.compile(p, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{p}': {e}")
        return compiled

    def check(self, text: str) -> SafetyResult:
        """
        Check input text for safety issues.

        Args:
            text: Input text to check

        Returns:
            SafetyResult with evaluation details
        """
        categories = []
        details = {}
        filtered = text
        is_safe = True
        confidence = 1.0

        # Length check
        if len(text) > self.config.max_input_length:
            categories.append("length_exceeded")
            is_safe = False
            details["max_length"] = self.config.max_input_length
            details["actual_length"] = len(text)

        # Check for prompt injection patterns
        if self.config.check_prompt_injection:
            injection_result = self._check_prompt_injection(text)
            if injection_result["detected"]:
                categories.append("prompt_injection")
                confidence = min(confidence, injection_result["confidence"])
                if injection_result["severity"] == "high":
                    is_safe = False
                details["injection"] = injection_result

        # Check blocked patterns
        for pattern in self._blocked_patterns:
            match = pattern.search(text)
            if match:
                categories.append("blocked_pattern")
                is_safe = False
                details["matched_pattern"] = match.group()
                break

        # Check warning patterns
        if is_safe:
            for pattern in self._warning_patterns:
                if pattern.search(text):
                    categories.append("warning_pattern")
                    confidence = min(confidence, 0.8)

        # Check blocked words
        text_lower = text.lower()
        for word in self.config.blocked_words:
            if word.lower() in text_lower:
                categories.append("blocked_word")
                is_safe = False
                details["blocked_word"] = word
                break

        # Check sensitive data patterns
        if self.config.check_sensitive_data:
            sensitive_result = self._check_sensitive_data(text)
            if sensitive_result["detected"]:
                categories.append("sensitive_data")
                details["sensitive_types"] = sensitive_result["types"]

        return SafetyResult(
            is_safe=is_safe,
            confidence=confidence,
            categories=categories,
            filtered_content=filtered if filtered != text else None,
            details=details,
        )

    def _check_prompt_injection(self, text: str) -> Dict[str, Any]:
        """Check for prompt injection patterns."""
        # Common prompt injection patterns
        injection_patterns = [
            r"ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|rules?)",
            r"(?:forget|disregard)\s+(?:all\s+)?(?:previous\s+)?instructions?",
            r"system:\s*ignore",
            r"you\s+(?:are\s+)?(?:now|free\s+to)\s+ignore",
            r"new\s+instruction[s]?:",
            r"override\s+(?:your\s+)?(?:system\s+)?(?:instructions?|rules?)",
            r"disable\s+(?:your\s+)?safety",
            r"\\n\\n(?:human|user):",
        ]

        detected = False
        severity = "low"
        confidence = 0.9

        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected = True
                if "disable" in pattern or "ignore all" in pattern:
                    severity = "high"
                    confidence = 0.95
                break

        return {
            "detected": detected,
            "severity": severity,
            "confidence": confidence,
        }

    def _check_sensitive_data(self, text: str) -> Dict[str, Any]:
        """Check for sensitive data patterns."""
        patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "api_key": r"(?:api[_-]?key|apikey|secret)[\\s:=]+[\\w-]{20,}",
        }

        detected_types = []
        for data_type, pattern in patterns.items():
            if re.search(pattern, text):
                detected_types.append(data_type)

        return {
            "detected": len(detected_types) > 0,
            "types": detected_types,
        }


class OutputGuardrails:
    """
    Output validation and filtering for LLM outputs.

    Provides:
    - Harmful content detection
    - PII detection in outputs
    - Safety level classification
    - Content filtering
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        """Initialize output guardrails with config."""
        self.config = config or SafetyConfig()

    def check(self, text: str) -> SafetyResult:
        """
        Check output text for safety issues.

        Args:
            text: Output text to check

        Returns:
            SafetyResult with evaluation details
        """
        categories = []
        details = {}
        filtered = text
        is_safe = True
        confidence = 1.0

        # Check harmful content patterns
        if self.config.check_harmful_content:
            harmful_result = self._check_harmful_content(text)
            if harmful_result["detected"]:
                categories.extend(harmful_result["categories"])
                is_safe = False
                confidence = min(confidence, harmful_result["confidence"])
                details["harmful"] = harmful_result

        # Check for PII in output
        if self.config.check_pii:
            pii_result = self._check_pii(text)
            if pii_result["detected"]:
                categories.append("pii_detected")
                details["pii"] = pii_result
                # Don't block, but warn
                confidence = min(confidence, 0.7)

        return SafetyResult(
            is_safe=is_safe,
            confidence=confidence,
            categories=categories,
            filtered_content=filtered if filtered != text else None,
            details=details,
        )

    def _check_harmful_content(self, text: str) -> Dict[str, Any]:
        """Check for harmful content."""
        # This is a simplified example - in production, use more sophisticated methods
        harmful_patterns = {
            "violence": [
                r"kill\s+(?:\w+\s+){0,3}(?:\w+|person|people|human)",
                r"attack\s+(?:\w+\s+){0,3}(?:\w+|person|people|human)",
            ],
            "self_harm": [
                r"(?:suicide|self[-\s]harm|kill\s+(?:yourself|myself))",
            ],
            "hate_speech": [
                r"(?:hate|kill|attack)\s+(?:\w+\s+)?(?:women|men|racial|ethnic)",
            ],
        }

        detected_categories = []
        severity = "low"
        confidence = 0.8

        for category, patterns in harmful_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected_categories.append(category)
                    if category in ["violence", "self_harm"]:
                        severity = "high"
                        confidence = 0.95
                    break

        return {
            "detected": len(detected_categories) > 0,
            "categories": detected_categories,
            "severity": severity,
            "confidence": confidence,
        }

    def _check_pii(self, text: str) -> Dict[str, Any]:
        """Check for PII in output."""
        # Similar to input guardrails - check for SSN, credit cards, etc.
        pii_patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        }

        detected_types = []
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, text):
                detected_types.append(pii_type)

        return {
            "detected": len(detected_types) > 0,
            "types": detected_types,
        }
