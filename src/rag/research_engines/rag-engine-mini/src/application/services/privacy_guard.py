"""
Privacy Guard Service
=====================
Redacts PII (Personally Identifiable Information) before sending data to LLMs.

خدمة حماية الخصوصية لإخفاء المعلومات الحساسة قبل إرسالها للذكاء الاصطناعي
"""

import re
import structlog
from typing import Dict, Tuple

log = structlog.get_logger()

class PrivacyGuardService:
    """
    Redacts and restores sensitive data.
    
    قرار التصميم: استخدام Regex لحماية البيانات الشخصية لضمان عدم تسربها لمزودي الخدمة الخارجيين
    """

    def __init__(self):
        # Basic PII patterns
        self._patterns = {
            "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "PHONE": r"\+?\d{10,15}",
            "SSN": r"\d{3}-\d{2}-\d{4}",
        }
        self._redaction_map: Dict[str, str] = {}

    def redact(self, text: str) -> str:
        """
        Replace sensitive patterns with placeholders.
        """
        redacted_text = text
        for label, pattern in self._patterns.items():
            matches = re.findall(pattern, redacted_text)
            for i, match in enumerate(matches):
                placeholder = f"<{label}_{i}>"
                self._redaction_map[placeholder] = match
                redacted_text = redacted_text.replace(match, placeholder)
        
        if self._redaction_map:
            log.info("pii_redacted", count=len(self._redaction_map))
        return redacted_text

    def restore(self, text: str) -> str:
        """
        Restore original values from placeholders.
        """
        restored_text = text
        for placeholder, original in self._redaction_map.items():
            restored_text = restored_text.replace(placeholder, original)
        
        return restored_text

    def clear(self):
        """Clear the redaction mapping for a new session."""
        self._redaction_map.clear()
