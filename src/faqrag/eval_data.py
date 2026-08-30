"""The evaluation question set.

Deliberately mixes four kinds of query, because a set of only clean questions
reports a hit-rate that says nothing about real behaviour:

* **clean** -- unambiguous, maps to exactly one FAQ.
* **paraphrase** -- the user's wording shares few words with the FAQ question,
  so it tests dense retrieval rather than keyword overlap.
* **ambiguous** -- reasonably satisfied by more than one FAQ; any of
  ``expected_faq_ids`` counts as a hit.
* **out_of_scope** -- not answerable from the FAQ at all. These have no expected
  ids and instead assert that the system *declines*. They are the cases where a
  RAG system is most tempted to hallucinate.
* **no_fabrication** -- a relevant FAQ exists, but it explicitly leaves the
  requested detail unspecified. The right behaviour is neither refusing nor
  inventing: retrieve the FAQ, answer from it, and say the detail is not
  specified. ``forbidden_pattern`` asserts no invented value appears.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Pattern

QueryKind = Literal["clean", "paraphrase", "ambiguous", "out_of_scope", "no_fabrication"]

#: Anything resembling a phone number or an email address. The source FAQ marks
#: both as unfinalised, so any such string in an answer is fabricated.
CONTACT_PATTERN = re.compile(
    r"(?:\+?\d[\d\s()\-]{7,}\d)|(?:[\w.+-]+@[\w-]+\.[\w.]+)|(?:\b800[\s-]?\d{3,})"
)


@dataclass(frozen=True)
class EvalCase:
    """One evaluation question and its expectation.

    Attributes:
        question: The query to send.
        lang: The language the answer must come back in.
        kind: Which behaviour this case probes.
        expected_faq_ids: FAQ ids that count as a correct retrieval. Empty for
            out-of-scope cases, which instead expect a refusal.
        forbidden_pattern: If set, the answer must not match it. Used to assert
            that unspecified details are never invented.
        note: Why this case is in the set.
    """

    question: str
    lang: Literal["ar", "en"]
    kind: QueryKind
    expected_faq_ids: tuple[str, ...] = field(default_factory=tuple)
    forbidden_pattern: Pattern[str] | None = None
    note: str = ""

    @property
    def expects_refusal(self) -> bool:
        """Whether the system should decline to answer this question."""
        return self.kind == "out_of_scope"

    def fabricated(self, answer: str) -> bool:
        """Whether ``answer`` contains a value this case forbids."""
        return bool(self.forbidden_pattern and self.forbidden_pattern.search(answer))


EVAL_CASES: tuple[EvalCase, ...] = (
    # --- English, clean ---------------------------------------------------
    EvalCase("What is the Mwfaq platform?", "en", "clean", ("001",)),
    EvalCase("What are Mwfaq's core values?", "en", "clean", ("003",)),
    EvalCase("What payment methods does Mwfaq accept?", "en", "clean", ("011",)),
    EvalCase("What is Mwfaq Academy?", "en", "clean", ("012",)),
    EvalCase("What is Mwfaq Business?", "en", "clean", ("015",)),
    # --- Arabic, clean ----------------------------------------------------
    EvalCase("ما هي منصة موفق؟", "ar", "clean", ("001",)),
    EvalCase("ما هي قيم موفق الأساسية؟", "ar", "clean", ("003",)),
    EvalCase("كيف أحجز فحصي الطبي عبر موفق؟", "ar", "clean", ("009",)),
    EvalCase("ما هي أكاديمية موفق؟", "ar", "clean", ("012",)),
    EvalCase("أين يقع مقر موفق؟", "ar", "clean", ("024",)),
    # --- Paraphrases: low lexical overlap with the FAQ wording ------------
    EvalCase(
        "Can I split my payment into instalments with Tabby?",
        "en",
        "paraphrase",
        ("011",),
        note=(
            "Brand name appears only in the answer text, not in any FAQ question, "
            "so this tests retrieval past the question wording. The corpus lists "
            "Tabby as accepted but says nothing about instalment terms, so either "
            "a hedged answer or a refusal is correct here -- only the retrieval "
            "of FAQ 011 is asserted."
        ),
    ),
    EvalCase(
        "I run a company. How do I automate my staff's medical checks?",
        "en",
        "paraphrase",
        ("015", "017", "026"),
        note="Business intent phrased as a personal situation.",
    ),
    EvalCase(
        "هل التدريب في موفق معتمد؟ وكم نسبة النجاح؟",
        "ar",
        "paraphrase",
        ("013", "014"),
        note="Two-part question spanning adjacent Academy FAQs.",
    ),
    EvalCase(
        "أبغى أسوي فحص قبل الزواج",
        "ar",
        "paraphrase",
        ("007", "009"),
        note="Saudi colloquial phrasing rather than formal Arabic.",
    ),
    # --- Ambiguous: several FAQs are legitimately acceptable --------------
    EvalCase(
        "How do I get started?",
        "en",
        "ambiguous",
        ("025", "026", "009"),
        note="Individual or company onboarding are both valid readings.",
    ),
    EvalCase(
        "ما الفائدة من استخدام موفق؟",
        "ar",
        "ambiguous",
        ("002", "008", "016", "021"),
        note="Benefits are spread across individual and business FAQs.",
    ),
    # --- Out of scope: the system must decline ----------------------------
    EvalCase(
        "What is the capital of France?",
        "en",
        "out_of_scope",
        note="Wholly unrelated to the knowledge base.",
    ),
    # --- Detail exists in scope but is explicitly unspecified -------------
    EvalCase(
        "What is Mwfaq's phone number and email address?",
        "en",
        "no_fabrication",
        ("024",),
        forbidden_pattern=CONTACT_PATTERN,
        note=(
            "FAQ 024 covers contact and states phone/email were never finalised. "
            "Answering from it is correct; inventing a value is the failure."
        ),
    ),
    EvalCase(
        "كم تكلفة الفحص الطبي بالريال؟",
        "ar",
        "out_of_scope",
        note="No pricing anywhere in the corpus; a plausible number must not be fabricated.",
    ),
    EvalCase(
        "هل يمكنني إلغاء موعدي واسترداد المبلغ؟",
        "ar",
        "out_of_scope",
        note="No cancellation or refund policy in the corpus.",
    ),
)
