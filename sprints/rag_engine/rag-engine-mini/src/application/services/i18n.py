"""
i18n Service
============
Internationalization service for multi-language support.

خدمة التدويل لدعم اللغات المتعددة
"""

from typing import Dict, Optional, List
from enum import Enum
import logging

log = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported languages."""

    ENGLISH = "en"
    ARABIC = "ar"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"


class MessageCategory(str, Enum):
    """Message categories for better organization."""

    COMMON = "common"
    DOCUMENTS = "documents"
    CHAT = "chat"
    SEARCH = "search"
    EXPORT = "export"
    ERRORS = "errors"
    VALIDATION = "validation"
    SUCCESS = "success"


class i18nService:
    """
    Internationalization service with comprehensive translation support.

    خدمة التدويل مع دعم ترجمة شاملة.
    """

    RTL_LANGUAGES = {Language.ARABIC, Language.HEBREW, Language.FARSI}

    def __init__(self):
        """
        Initialize i18n service with all translations.

        تهيئة خدمة التدويل بجميع الترجمات.
        """
        self._translations = self._build_translations()

    def _build_translations(self) -> Dict[Language, Dict[str, str]]:
        """
        Build complete translation dictionary for all languages.

        بناء قاموس ترجمة كامل لجميع اللغات.
        """
        return {
            Language.ENGLISH: self._get_english_translations(),
            Language.ARABIC: self._get_arabic_translations(),
            Language.SPANISH: self._get_spanish_translations(),
            Language.FRENCH: self._get_french_translations(),
            Language.GERMAN: self._get_german_translations(),
        }

    def _get_english_translations(self) -> Dict[str, str]:
        """English translations (base language)."""
        return {
            # Common messages / رسائل شائعة
            "welcome": "Welcome to RAG Engine",
            "loading": "Loading...",
            "success": "Success",
            "error": "An error occurred",
            "not_found": "Not found",
            "unauthorized": "Unauthorized access",
            "forbidden": "Access forbidden",
            "server_error": "Internal server error",
            "maintenance": "System under maintenance",
            # Document messages / رسائل المستندات
            "document_uploaded": "Document uploaded successfully",
            "document_uploading": "Uploading document...",
            "document_indexing": "Indexing document...",
            "document_indexed": "Document indexed successfully",
            "document_failed": "Document processing failed",
            "document_deleted": "Document deleted successfully",
            "document_not_found": "Document not found",
            "document_too_large": "Document too large (max {max_mb}MB)",
            "invalid_file_type": "Invalid file type. Allowed: {allowed_types}",
            # Chat messages / رسائل المحادثة
            "chat_created": "Chat session created",
            "chat_deleted": "Chat session deleted",
            "chat_turn_added": "Message added",
            "title_generated": "Chat title generated",
            "session_summarized": "Session summarized",
            "no_messages": "No messages in session",
            # Search messages / رسائل البحث
            "search_no_results": "No results found",
            "search_results_count": "{count} results found",
            "suggestion": "Did you mean?",
            "expanding_query": "Expanding query...",
            # Export messages / رسائل التصدير
            "export_started": "Export started",
            "export_completed": "Export completed",
            "export_failed": "Export failed",
            "downloading": "Downloading {filename}...",
            # Validation errors / أخطاء التحقق
            "validation_required": "This field is required",
            "validation_invalid_format": "Invalid format",
            "validation_invalid_email": "Invalid email format",
            "validation_password_too_short": "Password must be at least 8 characters",
            "validation_password_mismatch": "Passwords do not match",
            # Error details / تفاصيل الأخطاء
            "contact_support": "Please contact support if the problem persists",
            "try_again": "Please try again",
            "operation_timeout": "Operation timed out",
            "connection_failed": "Connection failed",
        }

    def _get_arabic_translations(self) -> Dict[str, str]:
        """Arabic translations."""
        return {
            # Common messages / رسائل شائعة
            "welcome": "مرحبًا بك في محرك RAG",
            "loading": "جاري التحميل...",
            "success": "نجح",
            "error": "حدث خطأ",
            "not_found": "غير موجود",
            "unauthorized": "غير مصرح بالوصول",
            "forbidden": "الوصول ممنوع",
            "server_error": "خطأ في الخادم",
            "maintenance": "النظام تحت الصيانة",
            # Document messages / رسائل المستندات
            "document_uploaded": "تم تحميل المستند بنجاح",
            "document_uploading": "جاري رفع المستند...",
            "document_indexing": "جاري فهرسة المستند...",
            "document_indexed": "تم فهرسة المستند بنجاح",
            "document_failed": "فشل معالجة المستند",
            "document_deleted": "تم حذف المستند بنجاح",
            "document_not_found": "المستند غير موجود",
            "document_too_large": "المستند كبير جدًا (الحد الأقصى {max_mb} ميجابايت)",
            "invalid_file_type": "نوع ملف غير صالح. المسموح: {allowed_types}",
            # Chat messages / رسائل المحادثة
            "chat_created": "تم إنشاء جلسة المحادثة",
            "chat_deleted": "تم حذف جلسة المحادثة",
            "chat_turn_added": "تم إضافة الرسالة",
            "title_generated": "تم توليد عنوان المحادثة",
            "session_summarized": "تم تلخيص الجلسة",
            "no_messages": "لا توجد رسائل في الجلسة",
            # Search messages / رسائل البحث
            "search_no_results": "لم يتم العثور على نتائج",
            "search_results_count": "تم العثور على {count} نتيجة",
            "suggestion": "هل تقصد؟",
            "expanding_query": "جاري توسيع الاستعلام...",
            # Export messages / رسائل التصدير
            "export_started": "بدأ التصدير",
            "export_completed": "اكتمل التصدير",
            "export_failed": "فشل التصدير",
            "downloading": "جاري تنزيل {filename}...",
            # Validation errors / أخطاء التحقق
            "validation_required": "هذا الحقل مطلوب",
            "validation_invalid_format": "تنسيق غير صالح",
            "validation_invalid_email": "تنسيق البريد الإلكتروني غير صالح",
            "validation_password_too_short": "يجب أن تكون كلمة المرور 8 أحرف على الأقل",
            "validation_password_mismatch": "كلمات المرور غير متطابقة",
            # Error details / تفاصيل الأخطاء
            "contact_support": "يرجى التواصل مع الدعم إذا استمرت المشكلة",
            "try_again": "يرجى المحاولة مرة أخرى",
            "operation_timeout": "انتهت مهلة العملية",
            "connection_failed": "فشل الاتصال",
        }

    def _get_spanish_translations(self) -> Dict[str, str]:
        """Spanish translations."""
        return {
            "welcome": "Bienvenido a RAG Engine",
            "success": "Éxito",
            "error": "Ocurrió un error",
            "not_found": "No encontrado",
            "document_uploaded": "Documento cargado exitosamente",
        }

    def _get_french_translations(self) -> Dict[str, str]:
        """French translations."""
        return {
            "welcome": "Bienvenue dans RAG Engine",
            "success": "Succès",
            "error": "Une erreur s'est produite",
            "not_found": "Non trouvé",
            "document_uploaded": "Document téléchargé avec succès",
        }

    def _get_german_translations(self) -> Dict[str, str]:
        """German translations."""
        return {
            "welcome": "Willkommen bei RAG Engine",
            "success": "Erfolg",
            "error": "Ein Fehler ist aufgetreten",
            "not_found": "Nicht gefunden",
            "document_uploaded": "Dokument erfolgreich hochgeladen",
        }

    def get_translation(
        self,
        language: Language,
        key: str,
        category: MessageCategory = MessageCategory.COMMON,
        **kwargs,
    ) -> str:
        """
        Get translated message with interpolation support.

        Args:
            language: Target language
            key: Translation key
            category: Message category
            **kwargs: Interpolation variables

        Returns:
            Translated message

        الحصول على رسالة مترجمة مع دعم الاستيفال

        الاستخدام:
            translation = i18n.get_translation(Language.ARABIC, "document_uploaded")
            print(translation)  # "تم تحميل المستند بنجاح"

            translation = i18n.get_translation(Language.ARABIC, "document_too_large", max_mb=20)
            print(translation)  # "المستند كبير جدًا (الحد الأقصى 20 ميجابايت)"
        """
        lang_translations = self._translations.get(language, self._get_english_translations())
        template = lang_translations.get(key, key)

        # Return key if not found
        if not template:
            log.warning(f"Translation key not found: {key}")
            return key

        # Interpolate variables
        if kwargs:
            try:
                return template.format(**kwargs)
            except (KeyError, ValueError) as e:
                log.error(f"Translation interpolation failed for key {key}", error=str(e))
                return template

        return template

    def detect_language(
        self,
        headers: Dict[str, str],
        accept_language: str | None = None,
        query_params: Dict[str, str] | None = None,
    ) -> Language:
        """
        Detect language from multiple sources with priority.

        Priority:
        1. Query parameter (?lang=ar)
        2. Accept-Language header
        3. User preference from database

        Args:
            headers: HTTP request headers
            accept_language: Accept-Language header value
            query_params: Query parameters

        Returns:
            Detected language (default: English)

        كشف اللغة من مصادر متعددة بالأولوية
        """
        # 1. Check query parameter (highest priority)
        if query_params and "lang" in query_params:
            lang_code = query_params["lang"].lower()
            try:
                return Language(lang_code)
            except ValueError:
                pass

        # 2. Check Accept-Language header
        if accept_language:
            for lang_part in accept_language.split(","):
                lang_code = lang_part.split(";")[0].strip().lower()
                try:
                    return Language(lang_code)
                except ValueError:
                    pass

        # 3. Check headers (fallback)
        if headers:
            for header_name in ["Accept-Language", "Language"]:
                if header_name in headers:
                    accept_lang = headers[header_name]
                    for lang_part in accept_lang.split(","):
                        lang_code = lang_part.split(";")[0].strip().lower()
                        try:
                            return Language(lang_code)
                        except ValueError:
                            pass

        # Default to English
        return Language.ENGLISH

    def is_rtl(self, language: Language) -> bool:
        """
        Check if language is right-to-left.

        Args:
            language: Language code

        Returns:
            True if RTL, False if LTR

        التحقق مما إذا كانت اللغة من اليمين إلى اليسار
        """
        return language in self.RTL_LANGUAGES

    def get_supported_languages(self) -> List[Dict[str, str]]:
        """
        Get list of supported languages with their native names.

        Returns:
            List of language dictionaries

        الحصول على قائمة اللغات المدعومة بأسمائها الأصلية
        """
        return [
            {"code": Language.ENGLISH.value, "name": "English", "native_name": "English"},
            {"code": Language.ARABIC.value, "name": "Arabic", "native_name": "العربية"},
            {"code": Language.SPANISH.value, "name": "Spanish", "native_name": "Español"},
            {"code": Language.FRENCH.value, "name": "French", "native_name": "Français"},
            {"code": Language.GERMAN.value, "name": "German", "native_name": "Deutsch"},
        ]

    def get_locale_info(self, language: Language) -> Dict[str, str]:
        """
        Get locale information for a language.

        Returns:
            Dictionary with language code, direction, and display name

        الحصول على معلومات الإعدادات المحلية للغة
        """
        return {
            "language_code": language.value,
            "direction": "rtl" if self.is_rtl(language) else "ltr",
            "display_name": self.get_translation(
                language, f"lang_{language.value}", MessageCategory.COMMON
            ),
            "is_rtl": self.is_rtl(language),
        }
