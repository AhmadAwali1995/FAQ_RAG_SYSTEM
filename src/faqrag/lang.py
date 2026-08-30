"""Query language detection and Arabic-aware text normalisation.

Detection is a deliberate heuristic rather than a learned model: the corpus is
strictly Arabic/English, and counting script membership is both faster and more
predictable than a statistical detector on short FAQ-style queries.
"""

from __future__ import annotations

import re
import unicodedata

from .models import Language

# Arabic script blocks: Arabic (0600-06FF), Supplement (0750-077F),
# Extended-A (08A0-08FF), Presentation Forms A (FB50-FDFF) and B (FE70-FEFF).
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# Arabic diacritics (tashkeel) and the tatweel elongation character. Both are
# presentational; stripping them makes lexical matching robust to typing style.
_TASHKEEL_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")

# Orthographic variants Saudi users type interchangeably.
_ARABIC_NORMALISATION = str.maketrans(
    {
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي", "ئ": "ي",
        "ة": "ه",
        "ؤ": "و",
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    }
)

_BIDI_MARKS = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩"))


def arabic_char_ratio(text: str) -> float:
    """Return the share of script-bearing characters that are Arabic.

    Digits, punctuation, and whitespace are ignored so that a query like
    "Tabby 2030" does not dilute the signal.
    """
    arabic = len(_ARABIC_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total = arabic + latin
    return arabic / total if total else 0.0


def detect_language(text: str, default: Language = "en", threshold: float = 0.2) -> Language:
    """Detect whether ``text`` is Arabic or English.

    Args:
        text: The query or document text.
        default: Language returned when the text carries no script signal at
            all (for example a bare number or an emoji).
        threshold: Minimum Arabic character ratio to classify as Arabic. The
            low default is intentional -- Arabic queries frequently embed Latin
            product names ("كيف أدفع عبر Tabby؟") and should still route to the
            Arabic corpus.

    Returns:
        ``"ar"`` or ``"en"``.
    """
    if not text or not text.strip():
        return default
    ratio = arabic_char_ratio(text)
    if ratio == 0.0 and not _LATIN_RE.search(text):
        return default
    return "ar" if ratio >= threshold else "en"


def normalise(text: str, lang: Language | None = None) -> str:
    """Normalise text for lexical (BM25) matching.

    Applies NFKC normalisation, strips bidi controls, and -- for Arabic --
    removes tashkeel and folds orthographic variants (أ/إ/آ to ا, ة to ه,
    ى to ي) plus Arabic-Indic digits.  English text is simply lowercased.
    """
    text = unicodedata.normalize("NFKC", text).translate(_BIDI_MARKS)
    if lang is None:
        lang = detect_language(text)
    if lang == "ar" or _ARABIC_RE.search(text):
        text = _TASHKEEL_RE.sub("", text)
        text = text.translate(_ARABIC_NORMALISATION)
    return text.lower().strip()


# Arabic definite article and common single-letter proclitics (و/ف/ب/ك/ل + ال).
_AL_PREFIX_RE = re.compile(r"^(?:و|ف)?(?:ب|ك|ل)?ال(?=.{3,})")
# ``\w`` is Unicode-aware in Python 3 and already covers Arabic letters, while
# correctly excluding Arabic punctuation such as ؟ ، and ؛.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str, lang: Language | None = None) -> list[str]:
    """Tokenise text for BM25.

    Beyond splitting on non-word characters, this strips the Arabic definite
    article ``ال`` (and the common و/ف/ب/ك/ل proclitics that precede it) so that
    "الفحوصات" and "فحوصات" share a term.  Full stemming is deliberately avoided:
    aggressive Arabic stemmers conflate distinct FAQ topics at this corpus size.
    """
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(normalise(text, lang)):
        stripped = _AL_PREFIX_RE.sub("", token)
        tokens.append(stripped or token)
    return tokens
