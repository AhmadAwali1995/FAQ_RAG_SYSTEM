"""Greetings must short-circuit the pipeline; real questions must not."""

import pytest

from faqrag.greeting import greeting_reply, is_greeting, is_salam

GREETINGS = [
    "السلام عليكم",
    "السلام عليكم ورحمة الله وبركاته",
    "وعليكم السلام",
    "هلا والله",
    "يا هلا",
    "مرحبا",
    "أهلاً وسهلاً",
    "صباح الخير",
    "مساء النور",
    "كيفك؟",
    "كيف حالك",
    "شلونك",
    "شخبارك",
    "وش أخبارك",
    "قواك الله",
    "قواك الله يا أبو سهل",
    "حياك الله",
    "يعطيك العافية",
    "السلام عليكم!! 😄",
    "مرحبا، كيف حالك؟",
    "السلام عليكم سهل، كيفك؟",
    "هلا سهل",
    "السلام عليكم أبو سهل",
    "صباح الخير يا سهل، كيف الحال",
    "hi sahl, how are you",
    "hi",
    "hellooo",
    "hey there",
    "good morning",
    "how are you?",
    "what's up",
    "salam",
]

QUESTIONS = [
    "السلام عليكم، كيف أسدد الفاتورة؟",
    "كيف أفتح حساب جديد",
    "مرحبا كم رسوم التحويل",
    "السلام عليكم سهل، كم سعر الفحص؟",
    "هلا سهل ابي احجز موعد",
    "يا ريت أعرف كيف أسدد",
    "ما هي سياسة الاسترجاع",
    "hi, how do I reset my password?",
    "how much does shipping cost",
    "",
    "   ",
]

SALAMS = [
    "السلام عليكم",
    "السلام عليكم ورحمة الله وبركاته",
    "السلام عليكم!! 😄",
    "السلام عليكم يا أبو سهل",
    "سلام",
    "سلامات",
]


@pytest.mark.parametrize("text", GREETINGS)
def test_greeting_is_detected(text):
    assert is_greeting(text)


@pytest.mark.parametrize("text", QUESTIONS)
def test_question_is_not_a_greeting(text):
    assert not is_greeting(text)


def test_reply_language_follows_the_greeting():
    assert any("؀" <= ch <= "ۿ" for ch in greeting_reply("ar"))
    assert not any("؀" <= ch <= "ۿ" for ch in greeting_reply("en"))


def test_replies_vary_so_repeat_greetings_do_not_repeat_verbatim():
    seen = {greeting_reply("ar") for _ in range(60)}
    assert len(seen) > 1


@pytest.mark.parametrize("text", SALAMS)
def test_salam_gets_the_prescribed_answer_every_time(text):
    """A salam is a formula: the reply is fixed, never rotated."""
    replies = {greeting_reply("ar", text) for _ in range(20)}
    assert len(replies) == 1
    reply = replies.pop()
    assert reply.startswith("وعليكم السلام ورحمة الله وبركاته")
    assert "يا أهلاً" in reply


def test_non_salam_greeting_still_rotates():
    assert not is_salam("كيفك")
    assert len({greeting_reply("ar", "كيفك") for _ in range(60)}) > 1
