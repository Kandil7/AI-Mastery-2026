"""
Medical Diagnosis Agent - PII Filter
=====================================
Privacy protection for medical data.

Features:
- Named entity recognition for PII
- Pattern-based detection (SSN, phone, email)
- Configurable redaction vs masking
- Audit logging

Author: AI-Mastery-2026
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# PII TYPES
# ============================================================

class PIIType(Enum):
    """Types of personally identifiable information."""
    NAME = "name"
    SSN = "ssn"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    DOB = "date_of_birth"
    MRN = "medical_record_number"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """A detected PII match."""
    pii_type: PIIType
    original_text: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    redacted_text: str = "[REDACTED]"


@dataclass
class FilterResult:
    """Result of PII filtering."""
    original_text: str
    filtered_text: str
    pii_found: List[PIIMatch]
    was_modified: bool
    processing_time_ms: float = 0.0


# ============================================================
# PII PATTERNS
# ============================================================

class PIIPatterns:
    """Regex patterns for PII detection."""
    
    # Social Security Number
    SSN = r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
    
    # Phone numbers (US format)
    PHONE = r'\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    
    # Email addresses
    EMAIL = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Credit card numbers
    CREDIT_CARD = r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    
    # IP addresses
    IP_ADDRESS = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    
    # Medical record numbers (common formats)
    MRN = r'\b(?:MRN|mrn|Medical Record)[-:\s]?\d{6,10}\b'
    
    # Date of birth patterns
    DOB = r'\b(?:DOB|dob|Date of Birth|Birthday)[-:\s]?\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
    
    # Street addresses (simplified)
    ADDRESS = r'\b\d{1,5}\s+(?:[A-Za-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)\b'


# ============================================================
# NAME DETECTION
# ============================================================

class NameDetector:
    """
    Detect person names using patterns and heuristics.
    
    Note: In production, use a proper NER model.
    """
    
    # Common name prefixes and suffixes
    PREFIXES = {'mr', 'mrs', 'ms', 'dr', 'prof', 'patient'}
    SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'md', 'phd', 'rn'}
    
    # Pattern for names following prefixes
    NAME_PATTERN = r'\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|Patient)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'
    
    def __init__(self, common_names: Optional[Set[str]] = None):
        """
        Args:
            common_names: Set of common first/last names for matching
        """
        self.common_names = common_names or self._default_names()
    
    def _default_names(self) -> Set[str]:
        """Return a small set of common names for demo."""
        return {
            'john', 'jane', 'michael', 'sarah', 'david', 'lisa',
            'robert', 'jennifer', 'william', 'mary', 'james', 'patricia',
            'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia',
            'miller', 'davis', 'rodriguez', 'martinez', 'wilson', 'anderson'
        }
    
    def detect(self, text: str) -> List[PIIMatch]:
        """Detect names in text."""
        matches = []
        
        # Pattern-based detection
        for match in re.finditer(self.NAME_PATTERN, text, re.IGNORECASE):
            name = match.group(1)
            matches.append(PIIMatch(
                pii_type=PIIType.NAME,
                original_text=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.9
            ))
        
        # Common name detection (case-sensitive for proper nouns)
        words = text.split()
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word).lower()
            if clean_word in self.common_names:
                # Check if it looks like a proper noun
                if word[0].isupper() and len(word) > 2:
                    start = text.find(word)
                    if start >= 0:
                        matches.append(PIIMatch(
                            pii_type=PIIType.NAME,
                            original_text=word,
                            start_pos=start,
                            end_pos=start + len(word),
                            confidence=0.6
                        ))
        
        return matches


# ============================================================
# PII FILTER
# ============================================================

class PIIFilter:
    """
    Complete PII filtering system.
    
    Features:
        - Multiple detection methods (regex, NER)
        - Configurable redaction
        - Audit logging
        - Whitelist support
    
    Example:
        >>> filter = PIIFilter()
        >>> result = filter.filter("Patient John Smith, SSN 123-45-6789")
        >>> print(result.filtered_text)
        "Patient [NAME], SSN [SSN]"
    """
    
    def __init__(
        self,
        detect_types: Optional[List[PIIType]] = None,
        redaction_char: str = "[{type}]",
        mask_char: str = "*",
        use_masking: bool = False,
        min_confidence: float = 0.5,
        whitelist: Optional[Set[str]] = None,
        log_detections: bool = True
    ):
        """
        Args:
            detect_types: PII types to detect (None = all)
            redaction_char: Format string for redaction
            mask_char: Character for masking
            use_masking: If True, mask with characters instead of redacting
            min_confidence: Minimum confidence threshold
            whitelist: Terms to never redact
            log_detections: Whether to log detections
        """
        self.detect_types = detect_types or list(PIIType)
        self.redaction_char = redaction_char
        self.mask_char = mask_char
        self.use_masking = use_masking
        self.min_confidence = min_confidence
        self.whitelist = whitelist or set()
        self.log_detections = log_detections
        
        self.name_detector = NameDetector()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.patterns = {
            PIIType.SSN: re.compile(PIIPatterns.SSN),
            PIIType.PHONE: re.compile(PIIPatterns.PHONE),
            PIIType.EMAIL: re.compile(PIIPatterns.EMAIL),
            PIIType.CREDIT_CARD: re.compile(PIIPatterns.CREDIT_CARD),
            PIIType.IP_ADDRESS: re.compile(PIIPatterns.IP_ADDRESS),
            PIIType.MRN: re.compile(PIIPatterns.MRN),
            PIIType.DOB: re.compile(PIIPatterns.DOB),
            PIIType.ADDRESS: re.compile(PIIPatterns.ADDRESS),
        }
    
    def _detect_pattern(
        self, 
        text: str, 
        pii_type: PIIType, 
        pattern: re.Pattern
    ) -> List[PIIMatch]:
        """Detect PII using regex pattern."""
        matches = []
        
        for match in pattern.finditer(text):
            if match.group() not in self.whitelist:
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    original_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=1.0  # Pattern matches are high confidence
                ))
        
        return matches
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect all PII in text.
        
        Args:
            text: Input text
        
        Returns:
            List of PII matches
        """
        all_matches = []
        
        # Pattern-based detection
        for pii_type in self.detect_types:
            if pii_type in self.patterns:
                matches = self._detect_pattern(
                    text, pii_type, self.patterns[pii_type]
                )
                all_matches.extend(matches)
        
        # Name detection
        if PIIType.NAME in self.detect_types:
            name_matches = self.name_detector.detect(text)
            all_matches.extend([
                m for m in name_matches 
                if m.original_text not in self.whitelist
            ])
        
        # Filter by confidence
        all_matches = [
            m for m in all_matches 
            if m.confidence >= self.min_confidence
        ]
        
        # Sort by position (for proper replacement)
        all_matches.sort(key=lambda m: m.start_pos, reverse=True)
        
        return all_matches
    
    def _get_replacement(self, match: PIIMatch) -> str:
        """Get replacement text for a PII match."""
        if self.use_masking:
            # Mask with characters
            return self.mask_char * len(match.original_text)
        else:
            # Replace with type tag
            return self.redaction_char.format(type=match.pii_type.value.upper())
    
    def filter(self, text: str) -> FilterResult:
        """
        Filter PII from text.
        
        Args:
            text: Input text
        
        Returns:
            FilterResult with filtered text and details
        """
        import time
        start_time = time.time()
        
        # Detect PII
        matches = self.detect(text)
        
        # Apply redaction
        filtered_text = text
        for match in matches:
            replacement = self._get_replacement(match)
            match.redacted_text = replacement
            filtered_text = (
                filtered_text[:match.start_pos] + 
                replacement + 
                filtered_text[match.end_pos:]
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log if enabled
        if self.log_detections and matches:
            logger.info(f"Filtered {len(matches)} PII instances")
            for match in matches:
                logger.debug(f"  {match.pii_type.value}: {match.original_text[:10]}...")
        
        return FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            pii_found=matches,
            was_modified=len(matches) > 0,
            processing_time_ms=processing_time
        )
    
    def filter_batch(self, texts: List[str]) -> List[FilterResult]:
        """Filter multiple texts."""
        return [self.filter(text) for text in texts]


# ============================================================
# AUDIT LOGGING
# ============================================================

@dataclass
class PIIAuditEntry:
    """Audit log entry for PII detection."""
    timestamp: str
    action: str
    pii_types_found: List[str]
    original_length: int
    filtered_length: int
    processing_time_ms: float


class PIIAuditLogger:
    """Audit logger for PII filtering operations."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.entries: List[PIIAuditEntry] = []
        self.log_file = log_file
    
    def log(self, result: FilterResult, action: str = "filter"):
        """Log a filtering operation."""
        entry = PIIAuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            pii_types_found=[m.pii_type.value for m in result.pii_found],
            original_length=len(result.original_text),
            filtered_length=len(result.filtered_text),
            processing_time_ms=result.processing_time_ms
        )
        
        self.entries.append(entry)
        
        if self.log_file:
            import json
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry.__dict__) + '\n')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        if not self.entries:
            return {}
        
        total_pii = sum(len(e.pii_types_found) for e in self.entries)
        pii_by_type: Dict[str, int] = {}
        
        for entry in self.entries:
            for pii_type in entry.pii_types_found:
                pii_by_type[pii_type] = pii_by_type.get(pii_type, 0) + 1
        
        return {
            "total_operations": len(self.entries),
            "total_pii_found": total_pii,
            "pii_by_type": pii_by_type,
            "avg_processing_time_ms": sum(e.processing_time_ms for e in self.entries) / len(self.entries)
        }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'PIIType', 'PIIMatch', 'FilterResult',
    'PIIPatterns', 'NameDetector', 'PIIFilter',
    'PIIAuditEntry', 'PIIAuditLogger',
]
