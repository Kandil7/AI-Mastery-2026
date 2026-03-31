"""
Defense Layer Module

Production-ready LLM defense mechanisms:
- Input sanitization
- Output filtering
- Rate limiting
- Access control

Features:
- Multi-layer defense
- Configurable policies
- Real-time protection
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DefenseAction(str, Enum):
    """Actions that can be taken by defense layer."""

    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"
    FLAG = "flag"
    QUARANTINE = "quarantine"


@dataclass
class DefenseConfig:
    """Configuration for defense layer."""

    # Input sanitization
    enable_input_sanitization: bool = True
    max_input_length: int = 10000
    allowed_content_types: Set[str] = field(default_factory=lambda: {"text/plain"})

    # Output filtering
    enable_output_filtering: bool = True
    max_output_length: int = 4096
    filter_pii: bool = True
    filter_secrets: bool = True

    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    tokens_per_minute: int = 100000

    # Access control
    enable_access_control: bool = True
    allowed_api_keys: Optional[Set[str]] = None
    allowed_ips: Optional[Set[str]] = None

    # Logging
    log_blocked_requests: bool = True
    log_all_requests: bool = False


@dataclass
class DefenseResult:
    """Result of defense processing."""

    action: DefenseAction
    is_allowed: bool
    sanitized_input: Optional[str] = None
    filtered_output: Optional[str] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "is_allowed": self.is_allowed,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class InputSanitizer:
    """
    Sanitizes input before processing.

    Removes malicious content and ensures
    input meets security requirements.
    """

    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        # Code injection
        (r"<script[^>]*>.*?</script>", "script_injection"),
        (r"javascript:", "javascript_protocol"),
        (r"on\w+\s*=", "event_handler"),

        # SQL injection
        (r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b)", "sql_keyword"),

        # Path traversal
        (r"\.\.[\\/]", "path_traversal"),

        # Command injection
        (r"[;&|`$]", "command_separator"),
    ]

    # PII patterns
    PII_PATTERNS = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
        (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "credit_card"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    ]

    def __init__(self, config: Optional[DefenseConfig] = None) -> None:
        self.config = config or DefenseConfig()
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.DANGEROUS_PATTERNS
        ]

        self._stats = {
            "total_inputs": 0,
            "sanitized": 0,
            "blocked": 0,
        }

    def sanitize(self, text: str) -> DefenseResult:
        """
        Sanitize input text.

        Args:
            text: Input text to sanitize

        Returns:
            Defense result with sanitized text
        """
        self._stats["total_inputs"] += 1

        # Check length
        if len(text) > self.config.max_input_length:
            self._stats["blocked"] += 1
            return DefenseResult(
                action=DefenseAction.BLOCK,
                is_allowed=False,
                reason=f"Input exceeds max length ({len(text)} > {self.config.max_input_length})",
            )

        # Check for dangerous patterns
        dangerous_found = []
        sanitized = text

        for pattern, name in self._compiled_patterns:
            if pattern.search(sanitized):
                dangerous_found.append(name)
                sanitized = pattern.sub("", sanitized)

        if dangerous_found:
            self._stats["sanitized"] += 1

            if len(dangerous_found) >= 3:
                # Too many dangerous patterns - block
                self._stats["blocked"] += 1
                return DefenseResult(
                    action=DefenseAction.BLOCK,
                    is_allowed=False,
                    reason=f"Multiple dangerous patterns detected: {dangerous_found}",
                    metadata={"patterns": dangerous_found},
                )

            return DefenseResult(
                action=DefenseAction.SANITIZE,
                is_allowed=True,
                sanitized_input=sanitized,
                reason=f"Removed dangerous patterns: {dangerous_found}",
                metadata={"patterns": dangerous_found},
            )

        return DefenseResult(
            action=DefenseAction.ALLOW,
            is_allowed=True,
            sanitized_input=text,
        )

    def remove_pii(self, text: str) -> str:
        """Remove PII from text."""
        if not self.config.filter_pii:
            return text

        sanitized = text
        for pattern, pii_type in self.PII_PATTERNS:
            sanitized = re.sub(pattern, f"[{pii_type}_REDACTED]", sanitized)

        return sanitized

    def get_stats(self) -> Dict[str, Any]:
        """Get sanitizer statistics."""
        return {
            **self._stats,
            "sanitization_rate": (
                self._stats["sanitized"] / self._stats["total_inputs"]
                if self._stats["total_inputs"] > 0 else 0
            ),
        }


class OutputFilter:
    """
    Filters model output for sensitive content.

    Prevents leakage of PII, secrets, and
    inappropriate content.
    """

    # Secret patterns
    SECRET_PATTERNS = [
        (r"sk-[A-Za-z0-9]{32,}", "api_key"),
        (r"ghp_[A-Za-z0-9]{36}", "github_token"),
        (r"AKIA[0-9A-Z]{16}", "aws_key"),
        (r"-----BEGIN (RSA |EC )?PRIVATE KEY-----", "private_key"),
        (r"password\s*[=:]\s*\S+", "password"),
    ]

    # Inappropriate content patterns
    INAPPROPRIATE_PATTERNS = [
        (r"\b(hate|racist|discrimination)\b", "hate_speech"),
        (r"\b(violence|harm|kill)\b", "violence"),
        (r"\b(self-harm|suicide)\b", "self_harm"),
    ]

    def __init__(self, config: Optional[DefenseConfig] = None) -> None:
        self.config = config or DefenseConfig()

        self._compiled_secrets = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.SECRET_PATTERNS
        ]

        self._compiled_inappropriate = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.INAPPROPRIATE_PATTERNS
        ]

        self._stats = {
            "total_outputs": 0,
            "filtered": 0,
            "blocked": 0,
        }

    def filter(self, text: str) -> DefenseResult:
        """
        Filter output text.

        Args:
            text: Output text to filter

        Returns:
            Defense result with filtered text
        """
        self._stats["total_outputs"] += 1

        # Check length
        if len(text) > self.config.max_output_length:
            text = text[:self.config.max_output_length] + "..."

        # Check for secrets
        secrets_found = []
        filtered = text

        for pattern, name in self._compiled_secrets:
            if pattern.search(filtered):
                secrets_found.append(name)
                if self.config.filter_secrets:
                    filtered = pattern.sub(f"[{name}_REDACTED]", filtered)

        # Check for inappropriate content
        inappropriate_found = []
        for pattern, name in self._compiled_inappropriate:
            if pattern.search(filtered):
                inappropriate_found.append(name)

        # Determine action
        if secrets_found and inappropriate_found:
            self._stats["blocked"] += 1
            return DefenseResult(
                action=DefenseAction.BLOCK,
                is_allowed=False,
                reason=f"Contains secrets and inappropriate content",
                metadata={"secrets": secrets_found, "inappropriate": inappropriate_found},
            )

        if secrets_found:
            self._stats["filtered"] += 1
            return DefenseResult(
                action=DefenseAction.SANITIZE,
                is_allowed=True,
                filtered_output=filtered,
                reason=f"Redacted secrets: {secrets_found}",
                metadata={"secrets": secrets_found},
            )

        if inappropriate_found:
            self._stats["flagged"] = self._stats.get("flagged", 0) + 1
            return DefenseResult(
                action=DefenseAction.FLAG,
                is_allowed=True,
                filtered_output=filtered,
                reason=f"Flagged for review: {inappropriate_found}",
                metadata={"inappropriate": inappropriate_found},
            )

        return DefenseResult(
            action=DefenseAction.ALLOW,
            is_allowed=True,
            filtered_output=text,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        return {
            **self._stats,
            "filter_rate": (
                self._stats["filtered"] / self._stats["total_outputs"]
                if self._stats["total_outputs"] > 0 else 0
            ),
        }


class RateLimiter:
    """
    Rate limiter for API requests.

    Implements token bucket algorithm for
    flexible rate limiting.
    """

    def __init__(self, config: Optional[DefenseConfig] = None) -> None:
        self.config = config or DefenseConfig()

        # Token buckets
        self._minute_bucket: Dict[str, List[float]] = defaultdict(list)
        self._hour_bucket: Dict[str, List[float]] = defaultdict(list)
        self._token_bucket: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

        self._stats = {
            "total_requests": 0,
            "rate_limited": 0,
        }

    async def check_rate_limit(
        self,
        client_id: str,
        tokens: int = 1,
    ) -> DefenseResult:
        """
        Check if request is within rate limits.

        Args:
            client_id: Client identifier
            tokens: Number of tokens for this request

        Returns:
            Defense result
        """
        self._stats["total_requests"] += 1

        current_time = time.time()

        # Clean old entries
        self._clean_bucket(self._minute_bucket[client_id], 60)
        self._clean_bucket(self._hour_bucket[client_id], 3600)
        self._clean_token_bucket(self._token_bucket[client_id], 60)

        # Check requests per minute
        if len(self._minute_bucket[client_id]) >= self.config.requests_per_minute:
            self._stats["rate_limited"] += 1
            return DefenseResult(
                action=DefenseAction.BLOCK,
                is_allowed=False,
                reason="Rate limit exceeded (requests/minute)",
                metadata={"retry_after": 60},
            )

        # Check requests per hour
        if len(self._hour_bucket[client_id]) >= self.config.requests_per_hour:
            self._stats["rate_limited"] += 1
            return DefenseResult(
                action=DefenseAction.BLOCK,
                is_allowed=False,
                reason="Rate limit exceeded (requests/hour)",
                metadata={"retry_after": 3600},
            )

        # Check tokens per minute
        recent_tokens = sum(
            t for t, ts in self._token_bucket[client_id]
            if current_time - ts < 60
        )
        if recent_tokens + tokens > self.config.tokens_per_minute:
            self._stats["rate_limited"] += 1
            return DefenseResult(
                action=DefenseAction.BLOCK,
                is_allowed=False,
                reason="Token limit exceeded",
                metadata={"retry_after": 60},
            )

        # Record request
        self._minute_bucket[client_id].append(current_time)
        self._hour_bucket[client_id].append(current_time)
        self._token_bucket[client_id].append((current_time, tokens))

        return DefenseResult(
            action=DefenseAction.ALLOW,
            is_allowed=True,
            metadata={
                "remaining_requests_minute": self.config.requests_per_minute - len(self._minute_bucket[client_id]),
                "remaining_tokens": self.config.tokens_per_minute - recent_tokens - tokens,
            },
        )

    def _clean_bucket(self, bucket: List[float], max_age: float) -> None:
        """Remove old entries from bucket."""
        current_time = time.time()
        cutoff = current_time - max_age
        bucket[:] = [t for t in bucket if t > cutoff]

    def _clean_token_bucket(
        self,
        bucket: List[Tuple[float, int]],
        max_age: float,
    ) -> None:
        """Remove old entries from token bucket."""
        current_time = time.time()
        cutoff = current_time - max_age
        bucket[:] = [(t, n) for t, n in bucket if t > cutoff]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            **self._stats,
            "rate_limit_rate": (
                self._stats["rate_limited"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
        }


class AccessControl:
    """
    Access control for API.

    Manages authentication and authorization
    for API access.
    """

    def __init__(self, config: Optional[DefenseConfig] = None) -> None:
        self.config = config or DefenseConfig()

        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._ip_whitelist: Set[str] = set()
        self._ip_blacklist: Set[str] = set()

        self._stats = {
            "total_requests": 0,
            "authorized": 0,
            "denied": 0,
        }

        # Initialize from config
        if self.config.allowed_api_keys:
            for key in self.config.allowed_api_keys:
                self._api_keys[key] = {
                    "created": time.time(),
                    "requests": 0,
                }

        if self.config.allowed_ips:
            self._ip_whitelist = self.config.allowed_ips.copy()

    def register_api_key(
        self,
        key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an API key."""
        self._api_keys[key] = {
            "created": time.time(),
            "metadata": metadata or {},
            "requests": 0,
        }

    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key."""
        if key in self._api_keys:
            del self._api_keys[key]
            return True
        return False

    def add_to_whitelist(self, ip: str) -> None:
        """Add IP to whitelist."""
        self._ip_whitelist.add(ip)

    def add_to_blacklist(self, ip: str) -> None:
        """Add IP to blacklist."""
        self._ip_blacklist.add(ip)

    def check_access(
        self,
        api_key: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> DefenseResult:
        """
        Check if access is allowed.

        Args:
            api_key: API key for authentication
            ip: Client IP address

        Returns:
            Defense result
        """
        self._stats["total_requests"] += 1

        # Check IP blacklist
        if ip and ip in self._ip_blacklist:
            self._stats["denied"] += 1
            return DefenseResult(
                action=DefenseAction.BLOCK,
                is_allowed=False,
                reason="IP address blacklisted",
                metadata={"ip": ip},
            )

        # Check IP whitelist (if enabled)
        if self._ip_whitelist and ip and ip not in self._ip_whitelist:
            self._stats["denied"] += 1
            return DefenseResult(
                action=DefenseAction.BLOCK,
                is_allowed=False,
                reason="IP address not whitelisted",
                metadata={"ip": ip},
            )

        # Check API key
        if self.config.enable_access_control and self._api_keys:
            if not api_key:
                self._stats["denied"] += 1
                return DefenseResult(
                    action=DefenseAction.BLOCK,
                    is_allowed=False,
                    reason="API key required",
                )

            if api_key not in self._api_keys:
                self._stats["denied"] += 1
                return DefenseResult(
                    action=DefenseAction.BLOCK,
                    is_allowed=False,
                    reason="Invalid API key",
                )

            # Update key stats
            self._api_keys[api_key]["requests"] += 1

        self._stats["authorized"] += 1

        return DefenseResult(
            action=DefenseAction.ALLOW,
            is_allowed=True,
            metadata={"api_key_valid": bool(api_key and api_key in self._api_keys)},
        )

    def generate_api_key(self, prefix: str = "sk") -> str:
        """Generate a new API key."""
        import secrets
        token = secrets.token_urlsafe(32)
        return f"{prefix}-{token}"

    def get_stats(self) -> Dict[str, Any]:
        """Get access control statistics."""
        return {
            **self._stats,
            "authorization_rate": (
                self._stats["authorized"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "registered_keys": len(self._api_keys),
        }


class DefenseLayer:
    """
    Complete defense layer for LLM applications.

    Combines all defense mechanisms into a
    unified security layer.
    """

    def __init__(self, config: Optional[DefenseConfig] = None) -> None:
        self.config = config or DefenseConfig()

        self.input_sanitizer = InputSanitizer(config)
        self.output_filter = OutputFilter(config)
        self.rate_limiter = RateLimiter(config)
        self.access_control = AccessControl(config)

        self._stats = {
            "total_requests": 0,
            "allowed": 0,
            "blocked": 0,
        }

    async def process_input(
        self,
        text: str,
        client_id: str,
        api_key: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> DefenseResult:
        """
        Process input through all defense layers.

        Args:
            text: Input text
            client_id: Client identifier
            api_key: API key
            ip: Client IP

        Returns:
            Defense result
        """
        self._stats["total_requests"] += 1

        # 1. Access control
        access_result = self.access_control.check_access(api_key, ip)
        if not access_result.is_allowed:
            self._stats["blocked"] += 1
            return access_result

        # 2. Rate limiting
        rate_result = await self.rate_limiter.check_rate_limit(client_id)
        if not rate_result.is_allowed:
            self._stats["blocked"] += 1
            return rate_result

        # 3. Input sanitization
        if self.config.enable_input_sanitization:
            sanitize_result = self.input_sanitizer.sanitize(text)
            if not sanitize_result.is_allowed:
                self._stats["blocked"] += 1
                return sanitize_result

            text = sanitize_result.sanitized_input or text

        self._stats["allowed"] += 1

        return DefenseResult(
            action=DefenseAction.ALLOW,
            is_allowed=True,
            sanitized_input=text,
        )

    def process_output(self, text: str) -> DefenseResult:
        """
        Process output through defense layers.

        Args:
            text: Output text

        Returns:
            Defense result
        """
        if self.config.enable_output_filtering:
            return self.output_filter.filter(text)

        return DefenseResult(
            action=DefenseAction.ALLOW,
            is_allowed=True,
            filtered_output=text,
        )

    async def process_request(
        self,
        input_text: str,
        client_id: str,
        api_key: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> Tuple[DefenseResult, Optional[str]]:
        """
        Process complete request.

        Args:
            input_text: Input text
            client_id: Client identifier
            api_key: API key
            ip: Client IP

        Returns:
            Tuple of (input result, sanitized input)
        """
        input_result = await self.process_input(input_text, client_id, api_key, ip)

        if not input_result.is_allowed:
            return input_result, None

        return input_result, input_result.sanitized_input

    def get_stats(self) -> Dict[str, Any]:
        """Get defense layer statistics."""
        return {
            **self._stats,
            "block_rate": (
                self._stats["blocked"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "input_stats": self.input_sanitizer.get_stats(),
            "output_stats": self.output_filter.get_stats(),
            "rate_limit_stats": self.rate_limiter.get_stats(),
            "access_stats": self.access_control.get_stats(),
        }
