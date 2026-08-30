"""Tests for the BM25 lexical index."""

from __future__ import annotations

import pytest

from faqrag.bm25 import BM25Index
from faqrag.models import Chunk


def chunk(faq_id: str, lang: str, question: str, answer: str, keywords: list[str]) -> Chunk:
    return Chunk(
        chunk_id=f"{faq_id}::{lang}",
        faq_id=faq_id,
        category="Test",
        lang=lang,  # type: ignore[arg-type]
        question=question,
        answer=answer,
        keywords=keywords,
    )


@pytest.fixture
def index() -> BM25Index:
    """A small bilingual index mirroring the shape of the real corpus."""
    return BM25Index().fit(
        [
            chunk(
                "011",
                "en",
                "What payment methods does Mwfaq accept?",
                "Mwfaq accepts Mada, Visa, Apple Pay, Tabby and Tamara.",
                ["payment", "Tabby", "Tamara"],
            ),
            chunk(
                "011",
                "ar",
                "ما هي طرق الدفع المتاحة؟",
                "تقبل موفق مدى وفيزا وآبل باي وتابي وتمارا.",
                ["الدفع", "تابي"],
            ),
            chunk(
                "012",
                "en",
                "What is Mwfaq Academy?",
                "Mwfaq Academy offers accredited medical training courses.",
                ["academy", "training"],
            ),
            chunk(
                "009",
                "ar",
                "كيف أحجز فحصي الطبي؟",
                "اختر نوع الفحص والمدينة والتاريخ ثم ادفع لتأكيد الحجز.",
                ["الحجز", "الفحوصات الطبية"],
            ),
        ]
    )


class TestSearch:
    """Lexical retrieval behaviour."""

    def test_exact_brand_term_wins(self, index: BM25Index) -> None:
        """The case BM25 exists to cover: rare product names vector search blurs."""
        hits = index.search("Tabby", k=3)
        assert hits[0].chunk.faq_id == "011"

    def test_multi_word_term(self, index: BM25Index) -> None:
        hits = index.search("Mwfaq Academy", k=3)
        assert hits[0].chunk.faq_id == "012"

    def test_matches_content_in_the_answer_not_only_the_question(
        self, index: BM25Index
    ) -> None:
        """Answers are indexed too; 'Apple Pay' appears in no question."""
        hits = index.search("Apple Pay", k=3)
        assert hits and hits[0].chunk.faq_id == "011"

    def test_matches_keywords(self, index: BM25Index) -> None:
        hits = index.search("training", k=3)
        assert hits[0].chunk.faq_id == "012"

    def test_language_filter(self, index: BM25Index) -> None:
        hits = index.search("الدفع", k=5, lang="ar")
        assert hits and all(hit.chunk.lang == "ar" for hit in hits)

    def test_arabic_definite_article_is_normalised(self, index: BM25Index) -> None:
        """'فحص' must match documents that only contain 'الفحص'."""
        assert index.search("الفحوصات", k=3, lang="ar")
        assert index.search("فحوصات", k=3, lang="ar")

    def test_arabic_orthographic_variants_match(self, index: BM25Index) -> None:
        """A user typing 'احجز' must match text written 'أحجز'."""
        assert index.search("احجز", k=3, lang="ar")

    def test_zero_score_documents_are_excluded(self, index: BM25Index) -> None:
        hits = index.search("Tabby", k=10)
        assert all(hit.score > 0 for hit in hits)

    def test_no_match_returns_empty(self, index: BM25Index) -> None:
        assert index.search("zzzzz quantum chromodynamics", k=5) == []

    def test_empty_query_returns_empty(self, index: BM25Index) -> None:
        assert index.search("", k=5) == []
        assert index.search("!!! ???", k=5) == []

    def test_k_limits_results(self, index: BM25Index) -> None:
        assert len(index.search("موفق الدفع الفحص", k=1, lang="ar")) <= 1

    def test_results_are_sorted_by_descending_score(self, index: BM25Index) -> None:
        hits = index.search("Mwfaq medical training payment", k=5, lang="en")
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)


class TestFit:
    """Index construction."""

    def test_len_reflects_corpus_size(self, index: BM25Index) -> None:
        assert len(index) == 4

    def test_searching_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="empty"):
            BM25Index().search("anything", k=3)

    def test_refit_replaces_corpus(self, index: BM25Index) -> None:
        index.fit([chunk("099", "en", "Only doc", "Only answer", [])])
        assert len(index) == 1
        assert index.search("Tabby", k=3) == []

    def test_idf_downweights_ubiquitous_terms(self, index: BM25Index) -> None:
        """'Mwfaq' is in most English docs, so it must discriminate less than
        a rare term like 'Tabby'."""
        common = index.search("Mwfaq", k=5, lang="en")
        rare = index.search("Tabby", k=5, lang="en")
        assert rare[0].score > common[0].score
