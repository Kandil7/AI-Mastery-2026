"""
Tests for Arabic Text Utilities

Test Arabic character counting, normalization, etc.
"""

import pytest
from arabic_llm.utils.arabic import (
    count_arabic_chars,
    count_diacritics,
    get_arabic_ratio,
    get_diacritics_ratio,
    normalize_arabic_text,
    remove_tashkeel,
    is_arabic_text,
)


class TestArabicCharCounting:
    """Test Arabic character counting"""

    def test_count_arabic_chars(self):
        """Test counting Arabic characters"""
        text = "العلمُ نورٌ"
        count = count_arabic_chars(text)
        assert count > 0

    def test_count_arabic_chars_mixed(self):
        """Test counting in mixed text"""
        text = "Hello العلم World"
        count = count_arabic_chars(text)
        assert count == 5  # Only العلم

    def test_count_arabic_chars_empty(self):
        """Test counting in empty text"""
        assert count_arabic_chars("") == 0


class TestDiacritics:
    """Test diacritics counting"""

    def test_count_diacritics(self):
        """Test counting diacritics"""
        text = "العلمُ نورٌ"
        count = count_diacritics(text)
        assert count >= 2  # At least ُ and ٌ

    def test_count_diacritics_no_tashkeel(self):
        """Test counting without diacritics"""
        text = "العلم نور"
        count = count_diacritics(text)
        assert count == 0


class TestArabicRatio:
    """Test Arabic ratio calculation"""

    def test_full_arabic_ratio(self):
        """Test ratio for fully Arabic text"""
        text = "العلم نور"
        ratio = get_arabic_ratio(text)
        assert ratio == 1.0

    def test_mixed_ratio(self):
        """Test ratio for mixed text"""
        text = "Hello العلم"
        ratio = get_arabic_ratio(text)
        assert 0.0 < ratio < 1.0

    def test_empty_ratio(self):
        """Test ratio for empty text"""
        ratio = get_arabic_ratio("")
        assert ratio == 0.0


class TestNormalization:
    """Test Arabic text normalization"""

    def test_normalize_alif_forms(self):
        """Test normalizing Alif forms"""
        text = "أحمد وإيمان وآمنة"
        normalized = normalize_arabic_text(text)
        assert "أ" not in normalized
        assert "إ" not in normalized
        assert "آ" not in normalized
        assert "ا" in normalized

    def test_normalize_alif_maqsura(self):
        """Test normalizing Alif Maqsura"""
        text = "فتى ومشى"
        normalized = normalize_arabic_text(text)
        assert "ى" not in normalized
        assert "ي" in normalized

    def test_normalize_ta_marbuta(self):
        """Test normalizing Ta Marbuta"""
        text = "فاطمة وخديجة"
        normalized = normalize_arabic_text(text)
        assert "ة" not in normalized
        assert "ه" in normalized


class TestTashkeelRemoval:
    """Test Tashkeel removal"""

    def test_remove_tashkeel(self):
        """Test removing diacritics"""
        text = "العلمُ نورٌ"
        without_tashkeel = remove_tashkeel(text)
        assert "ُ" not in without_tashkeel
        assert "ٌ" not in without_tashkeel
        assert len(without_tashkeel) < len(text)


class TestArabicDetection:
    """Test Arabic text detection"""

    def test_is_arabic_true(self):
        """Test Arabic detection returns True"""
        text = "العلم نور"
        assert is_arabic_text(text) is True

    def test_is_arabic_false(self):
        """Test Arabic detection returns False"""
        text = "Hello World"
        assert is_arabic_text(text) is False

    def test_is_arabic_threshold(self):
        """Test Arabic detection with custom threshold"""
        text = "Hello العلم"
        assert is_arabic_text(text, threshold=0.3) is True
        assert is_arabic_text(text, threshold=0.7) is False
