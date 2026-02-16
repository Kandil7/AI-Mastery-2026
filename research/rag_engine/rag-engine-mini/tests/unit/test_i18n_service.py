"""
Tests for Internationalization (i18n) Service

This module tests the i18n service functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.application.services.i18n_service import (
    i18nService,
    Language
)


@pytest.fixture
def i18n_service():
    """Create an i18n service instance for testing."""
    return i18nService()


@pytest.mark.asyncio
async def test_translate(i18n_service):
    """Test translating text to different languages."""
    # Test English to Spanish translation for a known key
    result = await i18n_service.translate("hello", Language.SPANISH)
    assert result == "Hola"

    # Test English to Spanish for a known phrase
    result = await i18n_service.translate("welcome_message", Language.SPANISH)
    assert result == "Bienvenido al Motor RAG"

    # Test translating to same language (should return original)
    result = await i18n_service.translate("hello", Language.ENGLISH)
    assert result == "Hello"

    # Test unknown text (should return original)
    unknown_text = "unknown_text"
    result = await i18n_service.translate(unknown_text, Language.SPANISH)
    assert result == unknown_text


@pytest.mark.asyncio
async def test_get_translations(i18n_service):
    """Test getting all translations for a language."""
    translations = await i18n_service.get_translations(Language.ENGLISH)
    assert isinstance(translations, dict)
    assert "hello" in translations
    assert translations["hello"] == "Hello"

    # Test getting specific keys
    translations = await i18n_service.get_translations(Language.SPANISH, keys=["hello", "goodbye"])
    assert len(translations) == 2
    assert translations["hello"] == "Hola"
    assert translations["goodbye"] == "Adiós"


@pytest.mark.asyncio
async def test_set_translation(i18n_service):
    """Test setting a new translation."""
    # Add a new translation
    result = await i18n_service.set_translation(Language.ENGLISH, "new_key", "New Value")
    assert result is True

    # Verify the translation was set
    translations = await i18n_service.get_translations(Language.ENGLISH, keys=["new_key"])
    assert translations["new_key"] == "New Value"


@pytest.mark.asyncio
async def test_detect_language(i18n_service):
    """Test language detection."""
    # Test English detection
    english_text = "Hello, how are you?"
    detected_lang = await i18n_service.detect_language(english_text)
    assert detected_lang == Language.ENGLISH

    # Test Spanish detection with Spanish keywords
    spanish_text = "Hola, cómo estás?"
    detected_lang = await i18n_service.detect_language(spanish_text)
    assert detected_lang == Language.SPANISH

    # Test Spanish detection with Spanish keywords
    spanish_text2 = "Buscar documento"
    detected_lang = await i18n_service.detect_language(spanish_text2)
    assert detected_lang == Language.SPANISH


@pytest.mark.asyncio
async def test_get_available_languages(i18n_service):
    """Test getting available languages."""
    languages = await i18n_service.get_available_languages()
    assert isinstance(languages, list)
    assert Language.ENGLISH in languages
    assert Language.SPANISH in languages
    assert len(languages) > 0


@pytest.mark.asyncio
async def test_multiple_language_translations(i18n_service):
    """Test that multiple languages have appropriate translations."""
    # Test English
    en_hello = await i18n_service.translate("hello", Language.ENGLISH)
    assert en_hello == "Hello"

    # Test Spanish
    es_hello = await i18n_service.translate("hello", Language.SPANISH)
    assert es_hello == "Hola"

    # Test other phrases
    en_welcome = await i18n_service.translate("welcome_message", Language.ENGLISH)
    es_welcome = await i18n_service.translate("welcome_message", Language.SPANISH)
    
    assert en_welcome == "Welcome to the RAG Engine"
    assert es_welcome == "Bienvenido al Motor RAG"


@pytest.mark.asyncio
async def test_fallback_behavior(i18n_service):
    """Test behavior when a translation is not available."""
    # Set a translation in English but not in Spanish
    await i18n_service.set_translation(Language.ENGLISH, "only_english", "Only English")
    
    # Try to get it in Spanish (should return original text if not found)
    result = await i18n_service.translate("only_english", Language.SPANISH)
    # Since we don't have a Spanish translation, it should return the key as-is
    # (in our implementation, it would return the key if not found in target language)
    assert result == "only_english"


@pytest.mark.asyncio
async def test_detect_language_with_more_examples(i18n_service):
    """Test language detection with more examples."""
    # Test with different English phrases
    assert await i18n_service.detect_language("Goodbye for now") == Language.ENGLISH
    assert await i18n_service.detect_language("Thank you very much") == Language.ENGLISH
    
    # Test with different Spanish phrases
    assert await i18n_service.detect_language("Adiós por ahora") == Language.SPANISH
    assert await i18n_service.detect_language("Gracias por todo") == Language.SPANISH
    assert await i18n_service.detect_language("Configuración del usuario") == Language.SPANISH