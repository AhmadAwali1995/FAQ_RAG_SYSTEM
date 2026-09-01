"""Tests for citation parsing, reranker score parsing, and refusal behaviour.

These cover the boundary where free-form model output becomes structured data --
the point at which a malformed reply must degrade gracefully rather than crash a
query or, worse, produce a confident answer with a wrong citation attached.
"""

from __future__ import annotations

import pytest

from faqrag.config import Settings
from faqrag.generate import AnswerGenerator, clean_answer_text, parse_sources, to_citations
from faqrag.llm import LLMClient, LLMError, strip_reasoning
from faqrag.models import Chunk, RetrievalResult, ScoredChunk
from faqrag.prompts import (
    INSUFFICIENT_CONTEXT_MARKER,
    casual_reply,
    format_context,
    no_match_message,
)
from faqrag.rerank import LLMReranker, NoOpReranker, parse_rerank_scores


def make_scored(faq_id: str, lang: str = "en", score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"{faq_id}::{lang}",
            faq_id=faq_id,
            category="Test",
            lang=lang,  # type: ignore[arg-type]
            question=f"Question {faq_id}?",
            answer=f"Answer for {faq_id}.",
            keywords=["kw"],
        ),
        score=score,
        relevance=score,
    )


def make_result(*faq_ids: str, confident: bool = True, lang: str = "en") -> RetrievalResult:
    return RetrievalResult(
        query="test query",
        query_lang=lang,  # type: ignore[arg-type]
        chunks=[make_scored(fid, lang) for fid in faq_ids],
        confident=confident,
        threshold=0.45,
    )


class StubLLM(LLMClient):
    """An LLM stub returning a canned reply, or raising a configured error."""

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, str]] = []
        self.stream_chunks: list[str] = []

    def complete(self, system, user, temperature=None, max_tokens=None) -> str:
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply

    def stream_complete(self, system, user, temperature=None, max_tokens=None):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        if self.stream_chunks:
            yield from self.stream_chunks
            return
        yield self.reply


class TestParseSources:
    """Extraction of the trailing SOURCES line."""

    def test_extracts_ids_and_strips_the_line(self) -> None:
        body, cited = parse_sources("Mwfaq accepts Tabby.\nSOURCES: 011", {"011"})
        assert body == "Mwfaq accepts Tabby."
        assert cited == ["011"]

    def test_multiple_ids(self) -> None:
        _, cited = parse_sources("Text.\nSOURCES: 013, 014", {"013", "014"})
        assert cited == ["013", "014"]

    def test_ignores_ids_not_in_context(self) -> None:
        """A model must not be able to cite a FAQ it was never shown."""
        _, cited = parse_sources("Text.\nSOURCES: 011, 999", {"011"})
        assert cited == ["011"]

    def test_deduplicates_preserving_order(self) -> None:
        _, cited = parse_sources("Text.\nSOURCES: 014, 013, 014", {"013", "014"})
        assert cited == ["014", "013"]

    def test_is_case_insensitive_and_tolerates_spacing(self) -> None:
        _, cited = parse_sources("Text.\nsources :  011 ", {"011"})
        assert cited == ["011"]

    def test_missing_sources_line(self) -> None:
        body, cited = parse_sources("Just an answer.", {"011"})
        assert body == "Just an answer."
        assert cited == []

    def test_normalises_spacing_and_invisible_characters(self) -> None:
        body, cited = parse_sources("Hello ,world. \u200bNext!\nSOURCES: 011", {"011"})
        assert body == "Hello,world. Next!"
        assert cited == ["011"]


class TestCleanAnswerText:
    def test_repairs_safe_punctuation_spacing(self) -> None:
        assert clean_answer_text(" First sentence.Second sentence?Third ") == (
            "First sentence. Second sentence? Third"
        )

    def test_arabic_answer_body_is_preserved(self) -> None:
        body, cited = parse_sources("موفق تقبل تابي.\nSOURCES: 011", {"011"})
        assert body == "موفق تقبل تابي."
        assert cited == ["011"]


class TestFormatContext:
    """The context block handed to the generator."""

    def test_labels_entries_by_faq_id(self) -> None:
        text = format_context([make_scored("015"), make_scored("016")])
        assert "FAQ 015" in text and "FAQ 016" in text

    def test_does_not_number_candidates(self) -> None:
        """Regression: numbering candidates [1], [2] next to zero-padded FAQ ids
        made the model cite position 1 as FAQ '001'. Since '001' is a real id,
        the bogus citation passed validation and pointed at the wrong FAQ."""
        text = format_context([make_scored("015"), make_scored("016")])
        assert "[1]" not in text
        assert "[2]" not in text


class TestParseRerankScores:
    """Parsing of the reranker's JSON verdict."""

    def test_plain_json(self) -> None:
        assert parse_rerank_scores('{"1": 9, "2": 3}', 2) == {1: 9.0, 2: 3.0}

    def test_tolerates_markdown_fences_and_prose(self) -> None:
        raw = 'Here are the scores:\n```json\n{"1": 8, "2": 2}\n```\nDone.'
        assert parse_rerank_scores(raw, 2) == {1: 8.0, 2: 2.0}

    def test_clamps_out_of_range_scores(self) -> None:
        assert parse_rerank_scores('{"1": 42, "2": -5}', 2) == {1: 10.0, 2: 0.0}

    def test_drops_indices_outside_the_candidate_set(self) -> None:
        assert parse_rerank_scores('{"1": 7, "9": 10}', 2) == {1: 7.0}

    def test_skips_malformed_entries_without_failing(self) -> None:
        """One bad entry must not discard the whole (expensive) rerank call."""
        assert parse_rerank_scores('{"1": 7, "2": "high"}', 2) == {1: 7.0}

    def test_accepts_float_scores(self) -> None:
        assert parse_rerank_scores('{"1": 7.5}', 1) == {1: 7.5}

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON object"):
            parse_rerank_scores("I cannot score these.", 2)


class TestReranker:
    """Reranking behaviour and its failure modes."""

    def test_reorders_by_model_score(self) -> None:
        chunks = [make_scored("A", score=0.9), make_scored("B", score=0.5)]
        reranked = LLMReranker(StubLLM('{"1": 2, "2": 9}')).rerank("q", chunks, top_k=2)
        assert [c.chunk.faq_id for c in reranked] == ["B", "A"]
        assert reranked[0].rerank_score == 9.0

    def test_falls_back_to_fused_order_when_the_model_fails(self) -> None:
        """A reranker outage must degrade quality, never break the query."""
        chunks = [make_scored("A", score=0.9), make_scored("B", score=0.5)]
        reranked = LLMReranker(StubLLM(error=LLMError("down"))).rerank("q", chunks, top_k=2)
        assert [c.chunk.faq_id for c in reranked] == ["A", "B"]
        assert all(c.rerank_score is None for c in reranked)

    def test_falls_back_on_unparseable_output(self) -> None:
        chunks = [make_scored("A"), make_scored("B")]
        reranked = LLMReranker(StubLLM("no json here")).rerank("q", chunks, top_k=2)
        assert [c.chunk.faq_id for c in reranked] == ["A", "B"]

    def test_respects_top_k(self) -> None:
        chunks = [make_scored(c) for c in "ABC"]
        assert len(LLMReranker(StubLLM('{"1": 9, "2": 8, "3": 7}')).rerank("q", chunks, 2)) == 2

    def test_single_candidate_skips_the_model_call(self) -> None:
        stub = StubLLM('{"1": 9}')
        LLMReranker(stub).rerank("q", [make_scored("A")], top_k=5)
        assert stub.calls == []

    def test_noop_reranker_passes_through(self) -> None:
        chunks = [make_scored(c) for c in "ABC"]
        result = NoOpReranker().rerank("q", chunks, top_k=2)
        assert [c.chunk.faq_id for c in result] == ["A", "B"]


class TestAnswerGenerator:
    """End-to-end generation behaviour, with the LLM stubbed."""

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings()

    def test_declines_when_retrieval_is_not_confident(self, settings: Settings) -> None:
        """The refusal happens before the model sees a weak context, so a poor
        match can never be dressed up as a confident answer."""
        stub = StubLLM("Some answer.\nSOURCES: 011")
        answer, cited = AnswerGenerator(settings, stub).generate(
            make_result("011", confident=False)
        )
        assert cited == []
        assert answer == no_match_message("en", settings.answer_style)
        assert stub.calls == [], "the LLM must not be called on a weak match"

    @pytest.mark.parametrize("style", ["saudi", "msa"])
    def test_declines_in_the_query_language_and_configured_voice(self, style: str) -> None:
        """A refusal must speak the user's language *and* the configured voice --
        otherwise every "I don't know" sounds like a different assistant."""
        settings = Settings(answer_style=style)
        answer, cited = AnswerGenerator(settings, StubLLM()).generate(
            make_result("011", confident=False, lang="ar")
        )
        assert cited == []
        assert answer == no_match_message("ar", style)

    def test_saudi_refusal_uses_dialect_not_msa(self) -> None:
        """The fallback should explain that the assistant is Mwfaq-specific."""
        answer = no_match_message("ar", "saudi")
        assert "\u0627\u0644\u0623\u0633\u0626\u0644\u0629 \u0627\u0644\u0645\u062a\u0639\u0644\u0642\u0629 \u0628\u0646\u0638\u0627\u0645 \u0645\u0648\u0641\u0642" in answer

    def test_answers_a_standalone_arabic_greeting_naturally(self, settings: Settings) -> None:
        result = make_result("011", confident=False, lang="ar")
        result.query = "\u0647\u0644\u0627 \u0623\u0628\u0648 \u0633\u0647\u0644"
        answer, cited = AnswerGenerator(settings, StubLLM()).generate(result)
        assert answer == "\u0647\u0644\u0627 \u0648\u063a\u0644\u0627\u060c \u0623\u0646\u0627 \u0623\u0628\u0648 \u0633\u0647\u0644. \u0648\u0634 \u062d\u0627\u0628 \u062a\u0639\u0631\u0641 \u0639\u0646 \u0646\u0638\u0627\u0645 \u0645\u0648\u0641\u0642\u061f"
        assert cited == []

    def test_answers_an_arabic_check_in_with_a_question_mark(self, settings: Settings) -> None:
        result = make_result("011", confident=False, lang="ar")
        result.query = "\u0643\u064a\u0641\u0643\u061f"
        answer, cited = AnswerGenerator(settings, StubLLM()).generate(result)
        assert answer == "\u064a\u0627 \u0647\u0644\u0627\u060c \u0623\u0646\u0627 \u0628\u062e\u064a\u0631 \u0637\u0627\u0644\u0645\u0627 \u0623\u0646\u062a \u0628\u062e\u064a\u0631. \u0648\u0634 \u062d\u0627\u0628 \u062a\u0639\u0631\u0641 \u0639\u0646 \u0646\u0638\u0627\u0645 \u0645\u0648\u0641\u0642\u061f"
        assert cited == []

    @pytest.mark.parametrize(
        "query",
        [
            "\u0643\u064a\u0641\u0643 \u0623\u0628\u0648 \u0633\u0647\u0644",
            "\u0623\u062e\u0628\u0627\u0631\u0643 \u0623\u0628\u0648 \u0633\u0647\u0644",
            "\u0639\u0644\u0648\u0645\u0643",
            "\u0627\u062d\u0648\u0627\u0644\u0643",
            "\u0634\u0644\u0648\u0646\u0643",
        ],
    )
    def test_understands_common_arabic_check_ins(self, settings: Settings, query: str) -> None:
        result = make_result("011", confident=False, lang="ar")
        result.query = query
        answer, cited = AnswerGenerator(settings, StubLLM()).generate(result)
        assert answer.startswith("\u064a\u0627 \u0647\u0644\u0627")
        assert cited == []

    def test_out_of_scope_reply_explains_the_supported_topic(self) -> None:
        answer = no_match_message("ar", "saudi")
        assert "\u0627\u0644\u0623\u0633\u0626\u0644\u0629 \u0627\u0644\u0645\u062a\u0639\u0644\u0642\u0629 \u0628\u0646\u0638\u0627\u0645 \u0645\u0648\u0641\u0642" in answer


class TestAnswerGeneratorContinuation:
    @pytest.fixture
    def settings(self) -> Settings:
        return Settings()

    def test_does_not_match_a_greeting_with_a_real_question(self) -> None:
        assert casual_reply("\u0647\u0644\u0627 \u0623\u0628\u0648 \u0633\u0647\u0644 \u0648\u0634 \u0637\u0631\u0642 \u0627\u0644\u062f\u0641\u0639\u061f", "ar") is None

    def test_answers_english_identity_question(self) -> None:
        assert casual_reply("Who are you?", "en") == (
            "I'm Abu Sahl, the Mwfaq system assistant. What would you like to know?"
        )

    def test_honours_the_insufficient_context_marker(self, settings: Settings) -> None:
        stub = StubLLM(f"I don't know.\n{INSUFFICIENT_CONTEXT_MARKER}")
        answer, cited = AnswerGenerator(settings, stub).generate(make_result("011"))
        assert cited == []
        assert INSUFFICIENT_CONTEXT_MARKER not in answer

    def test_returns_answer_and_citations(self, settings: Settings) -> None:
        stub = StubLLM("Mwfaq accepts Tabby.\nSOURCES: 011")
        answer, cited = AnswerGenerator(settings, stub).generate(make_result("011", "012"))
        assert answer == "Mwfaq accepts Tabby."
        assert cited == ["011"]

    def test_missing_citation_falls_back_to_the_top_chunk(self, settings: Settings) -> None:
        stub = StubLLM("An answer with no sources line.")
        answer, cited = AnswerGenerator(settings, stub).generate(make_result("011", "012"))
        assert answer == "An answer with no sources line."
        assert cited == ["011"]

    def test_llm_failure_falls_back_to_extractive(self, settings: Settings) -> None:
        """During an outage the retrieved text is returned verbatim -- degraded,
        but still grounded, and never invented."""
        stub = StubLLM(error=LLMError("model down"))
        answer, cited = AnswerGenerator(settings, stub).generate(make_result("011"))
        assert answer == "Answer for 011."
        assert cited == ["011"]

    def test_stream_generate_yields_arabic_chunks(self, settings: Settings) -> None:
        stub = StubLLM()
        stub.stream_chunks = ["أهلًا ", "بك ", "في موفق."]
        parts = list(AnswerGenerator(settings, stub).stream_generate(make_result("011", lang="ar")))
        assert parts == ["أهلًا ", "بك ", "في موفق."]

    def test_stream_generate_falls_back_to_extractive_on_error(self, settings: Settings) -> None:
        stub = StubLLM(error=LLMError("stream down"))
        parts = list(AnswerGenerator(settings, stub).stream_generate(make_result("011")))
        assert parts == ["Answer for 011."]

    def test_extractive_mode_returns_source_text_verbatim(self, settings: Settings) -> None:
        answer, cited = AnswerGenerator(settings, None).generate(make_result("011", "012"))
        assert answer == "Answer for 011."
        assert cited == ["011"]

    def test_empty_retrieval_declines(self, settings: Settings) -> None:
        result = RetrievalResult(query="q", query_lang="en", chunks=[], confident=False)
        answer, cited = AnswerGenerator(settings, StubLLM()).generate(result)
        assert cited == []
        assert answer


class TestToCitations:
    """Citation objects handed to the API."""

    def test_orders_by_model_citation_order(self) -> None:
        chunks = [make_scored("011"), make_scored("012")]
        citations = to_citations(chunks, ["012", "011"])
        assert [c.faq_id for c in citations] == ["012", "011"]

    def test_falls_back_to_all_chunks_when_nothing_was_cited(self) -> None:
        chunks = [make_scored("011"), make_scored("012")]
        assert len(to_citations(chunks, [])) == 2

    def test_ignores_unknown_ids(self) -> None:
        citations = to_citations([make_scored("011")], ["999"])
        assert [c.faq_id for c in citations] == ["011"]


class TestStripReasoning:
    """Reasoning-block removal."""

    @pytest.mark.parametrize("tag", ["think", "thinking", "reasoning"])
    def test_removes_reasoning_blocks(self, tag: str) -> None:
        assert strip_reasoning(f"<{tag}>hidden</{tag}>The answer.") == "The answer."

    def test_leaves_normal_text_untouched(self) -> None:
        assert strip_reasoning("  Just an answer.  ") == "Just an answer."

    def test_handles_multiline_blocks(self) -> None:
        assert strip_reasoning("<think>\na\nb\n</think>\nAnswer.") == "Answer."
