"""
i18n Service
=============
Internationalization service for multi-language support.

خدمة التدويل لدعم اللغات المتعددة
"""

from typing import Dict, Optional
from enum import Enum


class Language(str, Enum):
    """Supported languages."""

    ENGLISH = "en"
    ARABIC = "ar"


class i18nService:
    """Internationalization service."""

    def __init__(self):
        # Translations for common messages
        self._translations = {
            Language.ENGLISH: {
                "welcome": "Welcome to RAG Engine",
                "document_uploaded": "Document uploaded successfully",
                "error_occurred": "An error occurred",
                "invalid_input": "Invalid input",
            },
            Language.ARABIC: {
                "welcome": "مرحبًا بك في محرك RAG",
                "document_uploaded": "تم تحميل المستند بنجاح",
                "error_occurred": "حدث خطأ",
                "invalid_input": "إدخال غير صحيح",
            },
        }

    def get_translation(
        self,
        language: Language,
        key: str,
        **kwargs,
    ) -> str:
        """
        Get translated message.

        Args:
            language: Target language
            key: Translation key
            **kwargs: Interpolation variables

        Returns:
            Translated message
        """
        template = self._translations.get(language, {}).get(key, key)
        return template.format(**kwargs)

    def detect_language(
        self,
        headers: Dict[str, str],
    ) -> Language:
        """
        Detect language from HTTP headers.

        Args:
            headers: HTTP request headers

        Returns:
            Detected language (default: English)
        """
        # Check Accept-Language header
        accept_language = headers.get("Accept-Language", "")
        if "ar" in accept_language.lower():
            return Language.ARABIC
        return Language.ENGLISH

    def is_rtl(self, language: Language) -> bool:
        """
        Check if language is right-to-left.

        Args:
            language: Language code

        Returns:
            True if RTL, False if LTR
        """
        return language == Language.ARABIC
