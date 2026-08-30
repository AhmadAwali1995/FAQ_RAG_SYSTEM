"""Tests for language detection and Arabic-aware tokenisation."""

from __future__ import annotations

import pytest

from faqrag.lang import arabic_char_ratio, detect_language, normalise, tokenize


class TestDetectLanguage:
    """Query language detection."""

    @pytest.mark.parametrize(
        "text",
        [
            "ما هي منصة موفق؟",
            "كيف أحجز فحصي الطبي؟",
            "أين يقع مقر موفق؟",
        ],
    )
    def test_detects_arabic(self, text: str) -> None:
        assert detect_language(text) == "ar"

    @pytest.mark.parametrize(
        "text",
        [
            "What is the Mwfaq platform?",
            "How do I book a medical exam?",
            "Tabby",
        ],
    )
    def test_detects_english(self, text: str) -> None:
        assert detect_language(text) == "en"

    def test_arabic_query_with_latin_brand_stays_arabic(self) -> None:
        """Embedded product names must not flip an Arabic query to English."""
        assert detect_language("كيف أدفع عبر Tabby؟") == "ar"
        assert detect_language("ما هي أكاديمية موفق Mwfaq Academy؟") == "ar"

    def test_english_query_with_one_arabic_word_stays_english(self) -> None:
        assert detect_language("What does موفق mean in English exactly?") == "en"

    @pytest.mark.parametrize("text", ["", "   ", "2030", "!!!"])
    def test_no_script_signal_falls_back_to_default(self, text: str) -> None:
        assert detect_language(text) == "en"
        assert detect_language(text, default="ar") == "ar"

    def test_threshold_is_respected(self) -> None:
        text = "موفق Mwfaq platform booking system for exams"
        assert detect_language(text, threshold=0.05) == "ar"
        assert detect_language(text, threshold=0.9) == "en"


class TestArabicCharRatio:
    """Script-share measurement."""

    def test_pure_scripts(self) -> None:
        assert arabic_char_ratio("مرحبا") == 1.0
        assert arabic_char_ratio("hello") == 0.0

    def test_digits_and_punctuation_are_ignored(self) -> None:
        """Only script-bearing characters count, so numbers cannot skew it."""
        assert arabic_char_ratio("موفق 2030 ...") == 1.0
        assert arabic_char_ratio("Mwfaq 2030 ...") == 0.0

    def test_no_script_characters(self) -> None:
        assert arabic_char_ratio("123 !!!") == 0.0
        assert arabic_char_ratio("") == 0.0


class TestNormalise:
    """Text normalisation for lexical matching."""

    def test_folds_alef_variants(self) -> None:
        assert normalise("أحمد") == normalise("احمد") == normalise("إحمد")

    def test_folds_taa_marbuta_and_alef_maqsura(self) -> None:
        assert normalise("منصة") == "منصه"
        assert normalise("على") == "علي"

    def test_strips_tashkeel(self) -> None:
        assert normalise("مُوَفَّق") == normalise("موفق")

    def test_converts_arabic_indic_digits(self) -> None:
        assert normalise("٢٠٣٠") == "2030"

    def test_lowercases_english(self) -> None:
        assert normalise("Mwfaq ACADEMY") == "mwfaq academy"

    def test_strips_bidi_marks(self) -> None:
        """RLM marks in the source data must not survive into index terms."""
        assert normalise("‏موفق") == normalise("موفق")


class TestTokenize:
    """BM25 tokenisation."""

    def test_strips_arabic_definite_article(self) -> None:
        """'الفحوصات' and 'فحوصات' must share a term or BM25 misses matches."""
        assert tokenize("الفحوصات") == tokenize("فحوصات")

    def test_strips_proclitic_plus_article(self) -> None:
        assert tokenize("والفحوصات") == tokenize("فحوصات")
        assert tokenize("بالفحوصات") == tokenize("فحوصات")

    def test_keeps_short_words_intact(self) -> None:
        """The lookahead protects words that merely start with alef-lam."""
        assert tokenize("الله") == ["الله"]

    def test_excludes_arabic_punctuation(self) -> None:
        """Arabic '?' and ',' live in the Arabic block and must not be tokens."""
        tokens = tokenize("كيف أدفع؟ نعم، شكرا؛")
        assert all("؟" not in t and "،" not in t and "؛" not in t for t in tokens)
        assert tokens == ["كيف", "ادفع", "نعم", "شكرا"]

    def test_mixed_script_query(self) -> None:
        assert tokenize("كيف أدفع عبر Tabby؟") == ["كيف", "ادفع", "عبر", "tabby"]

    def test_empty_input(self) -> None:
        assert tokenize("") == []
        assert tokenize("!!! ???") == []
