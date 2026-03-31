"""
Prompt Injection Detection Module
==================================

Detects prompt injection attacks against LLM systems.

Classes:
    InjectionPattern: Represents a known injection pattern
    InjectionSeverity: Severity levels for injections
    PromptInjectionDetector: Detector for prompt injection attacks

Author: AI-Mastery-2026
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InjectionSeverity(Enum):
    """Severity levels for prompt injection."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InjectionPattern:
    """
    Represents a known prompt injection pattern.

    Attributes:
        name: Name/identifier for the pattern
        pattern: Regex pattern to match
        severity: Severity level if detected
        description: Description of the pattern
        examples: Example strings that match this pattern
    """

    name: str
    pattern: str
    severity: InjectionSeverity = InjectionSeverity.MEDIUM
    description: str = ""
    examples: List[str] = field(default_factory=list)

    def to_regex(self) -> re.Pattern:
        """Compile pattern to regex."""
        return re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)


class PromptInjectionDetector:
    """
    Detector for prompt injection attacks.

    Detects various types of prompt injection:
    - Direct injection (attempting to override system prompt)
    - Indirect injection (via user-provided context)
    - Context injection (hidden instructions in data)
    - Role playing attacks

    Example:
        >>> detector = PromptInjectionDetector()
        >>> result = detector.detect("Ignore previous instructions and...")
        >>> if result.detected:
        ...     print(f"Blocked: {result.severity}")
    """

    # Default injection patterns
    DEFAULT_PATTERNS: List[InjectionPattern] = [
        # Direct injection attempts
        InjectionPattern(
            name="ignore_instructions",
            pattern=r"ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|rules?)",
            severity=InjectionSeverity.HIGH,
            description="Attempt to ignore system instructions",
            examples=["ignore previous instructions", "ignore all rules"],
        ),
        InjectionPattern(
            name="forget_instructions",
            pattern=r"(?:forget|disregard)\s+(?:all\s+)?(?:previous\s+)?instructions?",
            severity=InjectionSeverity.HIGH,
            description="Attempt to make model forget instructions",
            examples=["forget previous instructions", "disregard all rules"],
        ),
        InjectionPattern(
            name="new_instructions",
            pattern=r"(?:new|additional)\s+instruction[s]?:",
            severity=InjectionSeverity.HIGH,
            description="Attempt to add new instructions",
            examples=["new instructions:", "additional instructions:"],
        ),
        InjectionPattern(
            name="override_system",
            pattern=r"override\s+(?:your\s+)?(?:system\s+)?(?:instructions?|rules?|prompt)",
            severity=InjectionSeverity.CRITICAL,
            description="Attempt to override system prompt",
            examples=["override your system prompt", "override system rules"],
        ),
        InjectionPattern(
            name="system_ignore",
            pattern=r"system:\s*ignore",
            severity=InjectionSeverity.HIGH,
            description="Direct system prompt override attempt",
            examples=["system: ignore previous"],
        ),
        InjectionPattern(
            name="disable_safety",
            pattern=r"(?:disable|turn off|remove)\s+(?:your\s+)?(?:safety|safety\s+features|guardrails)",
            severity=InjectionSeverity.CRITICAL,
            description="Attempt to disable safety measures",
            examples=["disable your safety", "turn off guardrails"],
        ),
        InjectionPattern(
            name="jailbreak",
            pattern=r"(?:you\s+are\s+)?(?:now\s+)?(?:free\s+to|can\s+)(?:ignore|do\s+anything)",
            severity=InjectionSeverity.CRITICAL,
            description="Classic jailbreak attempt",
            examples=["you are now free to ignore", "you can do anything"],
        ),
        InjectionPattern(
            name="role_play_begin",
            pattern=r"(?:pretend\s+to\s+be|act\s+as|扮演|你现在是一个)",
            severity=InjectionSeverity.MEDIUM,
            description="Role play attempt to bypass restrictions",
            examples=["pretend to be", "act as", "你现在是一个"],
        ),
        InjectionPattern(
            name="developer_mode",
            pattern=r"(?:developer|dev\s+mode|god\s+mode|supervisor\s+mode)",
            severity=InjectionSeverity.HIGH,
            description="Developer mode jailbreak attempt",
            examples=["developer mode", "dev mode", "god mode"],
        ),
        InjectionPattern(
            name="context_ignore",
            pattern=r"(?:for the purpose of|from now on|as of now)\s+,\s*(?:you are|respond as)",
            severity=InjectionSeverity.MEDIUM,
            description="Context manipulation attempt",
            examples=["for the purpose of testing, respond as"],
        ),
    ]

    def __init__(
        self,
        patterns: Optional[List[InjectionPattern]] = None,
        threshold: InjectionSeverity = InjectionSeverity.MEDIUM,
    ):
        """
        Initialize detector.

        Args:
            patterns: Custom patterns (uses defaults if not provided)
            threshold: Minimum severity to trigger detection
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self.threshold = threshold
        self._compiled_patterns = [p.to_regex() for p in self.patterns]

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect prompt injection in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with detection results:
            - detected: bool
            - severity: InjectionSeverity
            - matched_patterns: List[str]
            - confidence: float
            - details: Dict
        """
        matched_patterns = []
        highest_severity = InjectionSeverity.LOW

        for pattern, compiled in zip(self.patterns, self._compiled_patterns):
            if compiled.search(text):
                matched_patterns.append(pattern.name)
                if pattern.severity.value > highest_severity.value:
                    highest_severity = pattern.severity

        # Determine if detected based on threshold
        detected = highest_severity.value >= self.threshold.value

        # Calculate confidence
        confidence = self._calculate_confidence(matched_patterns, len(text))

        return {
            "detected": detected,
            "severity": highest_severity,
            "matched_patterns": matched_patterns,
            "confidence": confidence,
            "should_block": detected
            and highest_severity.value >= InjectionSeverity.HIGH.value,
            "details": {
                "match_count": len(matched_patterns),
                "text_length": len(text),
            },
        }

    def _calculate_confidence(
        self, matched_patterns: List[str], text_length: int
    ) -> float:
        """Calculate confidence score."""
        if not matched_patterns:
            return 0.0

        # Base confidence increases with more patterns
        base_confidence = min(0.5 + (len(matched_patterns) * 0.15), 0.95)

        # Reduce confidence for very long texts (more likely false positive)
        if text_length > 5000:
            base_confidence *= 0.9

        return base_confidence

    def detect_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect injection in multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of detection results
        """
        return [self.detect(text) for text in texts]

    def add_pattern(self, pattern: InjectionPattern) -> None:
        """Add a custom pattern to the detector."""
        self.patterns.append(pattern)
        self._compiled_patterns.append(pattern.to_regex())
        logger.info(f"Added pattern: {pattern.name}")

    def remove_pattern(self, name: str) -> bool:
        """Remove a pattern by name."""
        for i, p in enumerate(self.patterns):
            if p.name == name:
                self.patterns.pop(i)
                self._compiled_patterns.pop(i)
                return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "total_patterns": len(self.patterns),
            "patterns_by_severity": {
                s.value: sum(1 for p in self.patterns if p.severity == s)
                for s in InjectionSeverity
            },
            "threshold": self.threshold.value,
        }


# Utility function for quick detection
def quick_detect(text: str) -> Tuple[bool, InjectionSeverity]:
    """
    Quick detection function for simple use cases.

    Args:
        text: Text to check

    Returns:
        Tuple of (is_injection_detected, severity)
    """
    detector = PromptInjectionDetector()
    result = detector.detect(text)
    return result["detected"], result["severity"]
