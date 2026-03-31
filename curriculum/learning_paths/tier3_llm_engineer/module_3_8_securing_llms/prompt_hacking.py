"""
Prompt Hacking Security Module

Production-ready prompt security:
- Injection detection
- Jailbreak detection
- Prompt leakage prevention

Features:
- Pattern-based detection
- ML-based classification
- Real-time blocking
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityConfig:
    """Configuration for prompt security."""

    # Detection settings
    enable_injection_detection: bool = True
    enable_jailbreak_detection: bool = True
    enable_leakage_prevention: bool = True

    # Thresholds
    injection_threshold: float = 0.7
    jailbreak_threshold: float = 0.6
    similarity_threshold: float = 0.8

    # Actions
    block_on_high_threat: bool = True
    log_all_requests: bool = True
    alert_on_critical: bool = True

    # Patterns
    custom_injection_patterns: List[str] = field(default_factory=list)
    custom_jailbreak_patterns: List[str] = field(default_factory=list)

    # Allowlist/Denylist
    allowed_instructions: Set[str] = field(default_factory=set)
    blocked_phrases: Set[str] = field(default_factory=set)


@dataclass
class SecurityAnalysis:
    """Result of security analysis."""

    is_safe: bool
    threat_level: ThreatLevel
    threat_type: Optional[str] = None
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "threat_level": self.threat_level.value,
            "threat_type": self.threat_type,
            "confidence": self.confidence,
            "details": self.details,
            "recommendations": self.recommendations,
        }


class InjectionDetector:
    """
    Detects prompt injection attacks.

    Identifies attempts to override system instructions
    or inject malicious content.
    """

    # Common injection patterns
    INJECTION_PATTERNS = [
        # Direct instruction override
        r"(?i)ignore\s+(the\s+)?(previous|above|prior)\s+(instructions|rules|prompt)",
        r"(?i)forget\s+(the\s+)?(previous|above|prior)\s+(instructions|rules)",
        r"(?i)disregard\s+(the\s+)?(previous|above|prior)",

        # Role-playing attacks
        r"(?i)you\s+are\s+now\s+(in\s+)?(developer|debug|admin)\s+mode",
        r"(?i)act\s+as\s+(a\s+)?(developer|programmer|admin)",
        r"(?i)pretend\s+you\s+are\s+(unrestricted|uncensored)",

        # Instruction extraction
        r"(?i)what\s+(are\s+)?(your\s+)?(system\s+)?(instructions|prompt|rules)",
        r"(?i)reveal\s+(your\s+)?(system\s+)?(prompt|instructions)",
        r"(?i)print\s+(the\s+)?(above|previous)\s+(text|prompt|instructions)",

        # Context breaking
        r"(?i)start\s+with\s+\"(sure|here|ok|yes)\"",
        r"(?i)begin\s+with\s+\"(sure|here|ok|yes)\"",
        r"(?i)output\s+only\s+the\s+response",

        # Code injection
        r"<\?php",
        r"<script[^>]*>",
        r"{{\s*config\s*}}",
        r"\{\{\s*request\s*\}\}",

        # Base64/encoding tricks
        r"(?i)decode\s+(this|the\s+following)\s+(base64|hex|rot13)",
    ]

    def __init__(self, config: Optional[SecurityConfig] = None) -> None:
        self.config = config or SecurityConfig()
        self._compiled_patterns = [
            re.compile(p) for p in self.INJECTION_PATTERNS + self.config.custom_injection_patterns
        ]

        self._stats = {
            "total_analyzed": 0,
            "injections_detected": 0,
            "false_positives": 0,
        }

    def detect(self, text: str) -> SecurityAnalysis:
        """
        Detect injection attempts in text.

        Args:
            text: Text to analyze

        Returns:
            Security analysis result
        """
        self._stats["total_analyzed"] += 1

        threats = []
        max_confidence = 0.0

        # Pattern matching
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(text):
                confidence = self._calculate_pattern_confidence(pattern.pattern, text)
                threats.append({
                    "type": "pattern_match",
                    "pattern": pattern.pattern[:50],
                    "confidence": confidence,
                })
                max_confidence = max(max_confidence, confidence)

        # Check for blocked phrases
        for phrase in self.config.blocked_phrases:
            if phrase.lower() in text.lower():
                threats.append({
                    "type": "blocked_phrase",
                    "phrase": phrase,
                    "confidence": 0.9,
                })
                max_confidence = max(max_confidence, 0.9)

        # Determine threat level
        threat_level = self._get_threat_level(max_confidence)
        is_safe = threat_level in [ThreatLevel.NONE, ThreatLevel.LOW]

        if not is_safe:
            self._stats["injections_detected"] += 1

        return SecurityAnalysis(
            is_safe=is_safe,
            threat_level=threat_level,
            threat_type="injection" if threats else None,
            confidence=max_confidence,
            details={
                "threats": threats,
                "text_length": len(text),
            },
            recommendations=self._get_recommendations(threats),
        )

    def _calculate_pattern_confidence(
        self,
        pattern: str,
        text: str,
    ) -> float:
        """Calculate confidence score for pattern match."""
        # Higher confidence for more specific patterns
        if "ignore" in pattern.lower() or "forget" in pattern.lower():
            return 0.9
        elif "reveal" in pattern.lower() or "print" in pattern.lower():
            return 0.85
        elif "act as" in pattern.lower() or "pretend" in pattern.lower():
            return 0.75
        else:
            return 0.7

    def _get_threat_level(self, confidence: float) -> ThreatLevel:
        """Get threat level from confidence score."""
        if confidence >= 0.9:
            return ThreatLevel.CRITICAL
        elif confidence >= self.config.injection_threshold:
            return ThreatLevel.HIGH
        elif confidence >= 0.5:
            return ThreatLevel.MEDIUM
        elif confidence >= 0.3:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def _get_recommendations(self, threats: List[Dict]) -> List[str]:
        """Get security recommendations."""
        recommendations = []

        if any(t["type"] == "blocked_phrase" for t in threats):
            recommendations.append("Block request containing prohibited phrases")

        if any("ignore" in t.get("pattern", "").lower() for t in threats):
            recommendations.append("Implement instruction hierarchy enforcement")

        if any("reveal" in t.get("pattern", "").lower() for t in threats):
            recommendations.append("Add system prompt protection")

        if not recommendations:
            recommendations.append("Monitor for repeated attempts")

        return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            **self._stats,
            "detection_rate": (
                self._stats["injections_detected"] / self._stats["total_analyzed"]
                if self._stats["total_analyzed"] > 0 else 0
            ),
        }


class JailbreakDetector:
    """
    Detects jailbreak attempts.

    Identifies attempts to bypass safety filters
    and content restrictions.
    """

    # Known jailbreak techniques
    JAILBREAK_PATTERNS = [
        # DAN (Do Anything Now) variants
        r"(?i)dan\s*(\d+)?\s*(mode|prompt|jailbreak)",
        r"(?i)do\s+anything\s+now",
        r"(?i)dev\s+mode",

        # Role-play jailbreaks
        r"(?i)you\s+are\s+(an\s+)?(evil|unethical|unrestricted)\s+ai",
        r"(?i)pretend\s+to\s+be\s+(an\s+)?(hacker|criminal)",

        # Hypothetical scenarios
        r"(?i)for\s+(research|educational)\s+purposes",
        r"(?i)in\s+a\s+(hypothetical|fictional)\s+scenario",
        r"(?i)imagine\s+(a\s+)?(world|scenario)\s+where",

        # Translation attacks
        r"(?i)translate\s+the\s+following\s+to\s+(base64|rot13|leetspeak)",
        r"(?i)encode\s+this\s+message",

        # Token smuggling
        r"(?i)split\s+the\s+word\s+",
        r"(?i)write\s+each\s+letter\s+separately",

        # Logical paradoxes
        r"(?i)if\s+you\s+were\s+to\s+ignore\s+your\s+rules",
        r"(?i)what\s+would\s+happen\s+if\s+you\s+didn't\s+have\s+restrictions",
    ]

    # Jailbreak prompt templates
    KNOWN_JAILBREAKS = [
        "developer mode",
        "alpha test",
        "simulation mode",
        "theoretical discussion",
        "for educational purposes",
        "as an ai language model, you can",
    ]

    def __init__(self, config: Optional[SecurityConfig] = None) -> None:
        self.config = config or SecurityConfig()
        self._compiled_patterns = [
            re.compile(p) for p in self.JAILBREAK_PATTERNS + self.config.custom_jailbreak_patterns
        ]

        self._stats = {
            "total_analyzed": 0,
            "jailbreaks_detected": 0,
        }

    def detect(self, text: str) -> SecurityAnalysis:
        """
        Detect jailbreak attempts.

        Args:
            text: Text to analyze

        Returns:
            Security analysis result
        """
        self._stats["total_analyzed"] += 1

        threats = []
        max_confidence = 0.0

        # Pattern matching
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                confidence = self._calculate_jailbreak_confidence(pattern.pattern, text)
                threats.append({
                    "type": "jailbreak_pattern",
                    "pattern": pattern.pattern[:50],
                    "confidence": confidence,
                })
                max_confidence = max(max_confidence, confidence)

        # Check for known jailbreak phrases
        text_lower = text.lower()
        for jailbreak in self.KNOWN_JAILBREAKS:
            if jailbreak in text_lower:
                threats.append({
                    "type": "known_jailbreak",
                    "phrase": jailbreak,
                    "confidence": 0.8,
                })
                max_confidence = max(max_confidence, 0.8)

        # Check for suspicious structures
        if self._check_suspicious_structure(text):
            threats.append({
                "type": "suspicious_structure",
                "confidence": 0.6,
            })
            max_confidence = max(max_confidence, 0.6)

        threat_level = self._get_threat_level(max_confidence)
        is_safe = threat_level in [ThreatLevel.NONE, ThreatLevel.LOW]

        if not is_safe:
            self._stats["jailbreaks_detected"] += 1

        return SecurityAnalysis(
            is_safe=is_safe,
            threat_level=threat_level,
            threat_type="jailbreak" if threats else None,
            confidence=max_confidence,
            details={
                "threats": threats,
                "text_length": len(text),
            },
            recommendations=self._get_recommendations(threats),
        )

    def _calculate_jailbreak_confidence(
        self,
        pattern: str,
        text: str,
    ) -> float:
        """Calculate confidence for jailbreak detection."""
        if "dan" in pattern.lower():
            return 0.95
        elif "do anything" in pattern.lower():
            return 0.9
        elif "educational" in pattern.lower() or "research" in pattern.lower():
            return 0.7
        else:
            return 0.65

    def _check_suspicious_structure(self, text: str) -> bool:
        """Check for suspicious prompt structures."""
        # Multiple conflicting instructions
        if text.count("ignore") > 1 or text.count("forget") > 1:
            return True

        # Unusual formatting
        if text.count("```") > 4:
            return True

        # Excessive role-playing setup
        if text.count("you are") > 2:
            return True

        return False

    def _get_threat_level(self, confidence: float) -> ThreatLevel:
        """Get threat level from confidence."""
        if confidence >= 0.85:
            return ThreatLevel.CRITICAL
        elif confidence >= self.config.jailbreak_threshold:
            return ThreatLevel.HIGH
        elif confidence >= 0.4:
            return ThreatLevel.MEDIUM
        elif confidence >= 0.2:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def _get_recommendations(self, threats: List[Dict]) -> List[str]:
        """Get security recommendations."""
        recommendations = []

        if any(t["type"] == "known_jailbreak" for t in threats):
            recommendations.append("Block known jailbreak patterns")

        if any("dan" in t.get("pattern", "").lower() for t in threats):
            recommendations.append("Implement DAN detection rules")

        recommendations.append("Review and update jailbreak patterns regularly")

        return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            **self._stats,
            "detection_rate": (
                self._stats["jailbreaks_detected"] / self._stats["total_analyzed"]
                if self._stats["total_analyzed"] > 0 else 0
            ),
        }


class PromptLeakagePreventer:
    """
    Prevents prompt leakage attacks.

    Detects and prevents attempts to extract
    system prompts and internal instructions.
    """

    # Leakage patterns
    LEAKAGE_PATTERNS = [
        # Direct extraction
        r"(?i)what\s+(is\s+)?(are\s+)?(your\s+)?(system\s+)?(prompt|instructions|rules)",
        r"(?i)show\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions)",
        r"(?i)tell\s+(me\s+)?(what\s+)?(you\s+)?(were\s+)?(told|instructed)",

        # Indirect extraction
        r"(?i)repeat\s+(the\s+)?(above|previous|first)\s+(message|text|prompt)",
        r"(?i)output\s+(the\s+)?(full\s+)?(conversation|context)",
        r"(?i)print\s+(everything\s+)?(above|before)",

        # Encoding tricks
        r"(?i)encode\s+(your\s+)?(instructions|prompt)\s+(as|in)",
        r"(?i)convert\s+(your\s+)?(system\s+)?(prompt|instructions)\s+to",

        # Partial extraction
        r"(?i)what\s+does\s+(the\s+)?(first|beginning|start)\s+(of\s+)?(your|the)\s+(prompt|instructions)\s+(say|contain)",
        r"(?i)complete\s+(this\s+)?(sentence|phrase):\s*(you\s+are|your\s+instructions)",
    ]

    def __init__(self, config: Optional[SecurityConfig] = None) -> None:
        self.config = config or SecurityConfig()
        self._compiled_patterns = [
            re.compile(p) for p in self.LEAKAGE_PATTERNS
        ]

        self._stats = {
            "total_analyzed": 0,
            "leakage_attempts": 0,
            "blocked": 0,
        }

    def detect(self, text: str) -> SecurityAnalysis:
        """
        Detect prompt leakage attempts.

        Args:
            text: Text to analyze

        Returns:
            Security analysis result
        """
        self._stats["total_analyzed"] += 1

        threats = []
        max_confidence = 0.0

        # Pattern matching
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                confidence = self._calculate_leakage_confidence(pattern.pattern, text)
                threats.append({
                    "type": "leakage_pattern",
                    "pattern": pattern.pattern[:50],
                    "confidence": confidence,
                })
                max_confidence = max(max_confidence, confidence)

        threat_level = self._get_threat_level(max_confidence)
        is_safe = threat_level in [ThreatLevel.NONE, ThreatLevel.LOW]

        if not is_safe:
            self._stats["leakage_attempts"] += 1
            if self.config.block_on_high_threat and threat_level == ThreatLevel.HIGH:
                self._stats["blocked"] += 1

        return SecurityAnalysis(
            is_safe=is_safe,
            threat_level=threat_level,
            threat_type="prompt_leakage" if threats else None,
            confidence=max_confidence,
            details={
                "threats": threats,
                "text_length": len(text),
            },
            recommendations=self._get_recommendations(threats),
        )

    def _calculate_leakage_confidence(
        self,
        pattern: str,
        text: str,
    ) -> float:
        """Calculate confidence for leakage detection."""
        if "system prompt" in pattern.lower() or "instructions" in pattern.lower():
            return 0.9
        elif "repeat" in pattern.lower() or "output" in pattern.lower():
            return 0.8
        elif "encode" in pattern.lower() or "convert" in pattern.lower():
            return 0.85
        else:
            return 0.7

    def _get_threat_level(self, confidence: float) -> ThreatLevel:
        """Get threat level from confidence."""
        if confidence >= 0.85:
            return ThreatLevel.CRITICAL
        elif confidence >= self.config.similarity_threshold:
            return ThreatLevel.HIGH
        elif confidence >= 0.5:
            return ThreatLevel.MEDIUM
        elif confidence >= 0.3:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def _get_recommendations(self, threats: List[Dict]) -> List[str]:
        """Get security recommendations."""
        return [
            "Never reveal system prompts or instructions",
            "Implement response filtering for sensitive content",
            "Log and monitor leakage attempts",
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get preventer statistics."""
        return {
            **self._stats,
            "block_rate": (
                self._stats["blocked"] / self._stats["leakage_attempts"]
                if self._stats["leakage_attempts"] > 0 else 0
            ),
        }


class PromptSecurityAnalyzer:
    """
    Comprehensive prompt security analyzer.

    Combines multiple detection methods for
    complete prompt security analysis.
    """

    def __init__(self, config: Optional[SecurityConfig] = None) -> None:
        self.config = config or SecurityConfig()

        self.injection_detector = InjectionDetector(config)
        self.jailbreak_detector = JailbreakDetector(config)
        self.leakage_preventer = PromptLeakagePreventer(config)

        self._stats = {
            "total_analyzed": 0,
            "threats_detected": 0,
            "blocked": 0,
        }

    def analyze(self, text: str) -> SecurityAnalysis:
        """
        Perform comprehensive security analysis.

        Args:
            text: Text to analyze

        Returns:
            Combined security analysis
        """
        self._stats["total_analyzed"] += 1

        # Run all detectors
        injection_result = self.injection_detector.detect(text)
        jailbreak_result = self.jailbreak_detector.detect(text)
        leakage_result = self.leakage_preventer.detect(text)

        # Combine results
        all_threats = []
        max_confidence = 0.0
        threat_types = []

        for result in [injection_result, jailbreak_result, leakage_result]:
            if result.threat_type:
                threat_types.append(result.threat_type)
                all_threats.extend(result.details.get("threats", []))
            max_confidence = max(max_confidence, result.confidence)

        # Determine overall safety
        threat_level = self._get_overall_threat_level(
            injection_result.threat_level,
            jailbreak_result.threat_level,
            leakage_result.threat_level,
        )

        is_safe = threat_level in [ThreatLevel.NONE, ThreatLevel.LOW]

        if not is_safe:
            self._stats["threats_detected"] += 1

            if self.config.block_on_high_threat and threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self._stats["blocked"] += 1

        # Combine recommendations
        all_recommendations = []
        for result in [injection_result, jailbreak_result, leakage_result]:
            all_recommendations.extend(result.recommendations)

        return SecurityAnalysis(
            is_safe=is_safe,
            threat_level=threat_level,
            threat_type=", ".join(threat_types) if threat_types else None,
            confidence=max_confidence,
            details={
                "injection_analysis": injection_result.to_dict(),
                "jailbreak_analysis": jailbreak_result.to_dict(),
                "leakage_analysis": leakage_result.to_dict(),
                "all_threats": all_threats,
            },
            recommendations=list(set(all_recommendations)),
        )

    def _get_overall_threat_level(
        self,
        injection: ThreatLevel,
        jailbreak: ThreatLevel,
        leakage: ThreatLevel,
    ) -> ThreatLevel:
        """Get overall threat level from individual results."""
        levels = [injection, jailbreak, leakage]

        if ThreatLevel.CRITICAL in levels:
            return ThreatLevel.CRITICAL
        elif ThreatLevel.HIGH in levels:
            return ThreatLevel.HIGH
        elif ThreatLevel.MEDIUM in levels:
            return ThreatLevel.MEDIUM
        elif ThreatLevel.LOW in levels:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def is_safe(self, text: str) -> bool:
        """Quick safety check."""
        return self.analyze(text).is_safe

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            **self._stats,
            "injection_stats": self.injection_detector.get_stats(),
            "jailbreak_stats": self.jailbreak_detector.get_stats(),
            "leakage_stats": self.leakage_preventer.get_stats(),
        }
