from __future__ import annotations

from faqrag.config import Settings
from faqrag.generate import AnswerGenerator
from faqrag.models import Chunk, RetrievalResult, ScoredChunk
from faqrag.pipeline import RagPipeline


class StubRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self._result = result

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        return self._result


class FailingRetriever:
    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        raise AssertionError("a social message must not reach retrieval")


class StubStreamingLLM:
    def complete(self, system, user, temperature=None, max_tokens=None) -> str:
        return "أهلًا بك.\nSOURCES: 011"

    def stream_complete(self, system, user, temperature=None, max_tokens=None):
        yield "أهلًا "
        yield "بك."
        yield "\nSOURCES: 011"


def make_result() -> RetrievalResult:
    chunk = ScoredChunk(
        chunk=Chunk(
            chunk_id="011::ar",
            faq_id="011",
            category="Payments",
            lang="ar",
            question="وش طرق الدفع؟",
            answer="نقبل عدة طرق دفع.",
            keywords=["tabby"],
        ),
        score=0.9,
        relevance=0.9,
    )
    return RetrievalResult(query="وش طرق الدفع؟", query_lang="ar", chunks=[chunk], confident=True)


def test_stream_answer_emits_metadata_delta_and_final() -> None:
    pipeline = RagPipeline(
        Settings(log_retrieval_traces=False),
        StubRetriever(make_result()),
        AnswerGenerator(Settings(log_retrieval_traces=False), StubStreamingLLM()),
    )

    events = list(pipeline.stream_answer("وش طرق الدفع؟"))
    assert [event.event for event in events] == ["metadata", "delta", "final"]
    assert events[-1].data["answer"] == "أهلًا بك."
    assert events[-1].data["cited_faq_ids"] == ["011"]


def test_stream_hides_a_sources_marker_split_across_chunks() -> None:
    class SplitSourceLLM(StubStreamingLLM):
        def stream_complete(self, system, user, temperature=None, max_tokens=None):
            yield "First sentence."
            yield "\nSOU"
            yield "RCES: 011"

    pipeline = RagPipeline(
        Settings(log_retrieval_traces=False),
        StubRetriever(make_result()),
        AnswerGenerator(Settings(log_retrieval_traces=False), SplitSourceLLM()),
    )
    events = list(pipeline.stream_answer("test"))
    streamed = " ".join(event.data["text"] for event in events if event.event == "delta")
    assert streamed == "First sentence."


def test_social_message_bypasses_retrieval_and_streams_immediately() -> None:
    settings = Settings(log_retrieval_traces=False)
    pipeline = RagPipeline(settings, FailingRetriever(), AnswerGenerator(settings, None))
    events = list(pipeline.stream_answer("\u0647\u0644\u0627 \u0623\u0628\u0648 \u0633\u0647\u0644"))
    assert [event.event for event in events] == ["metadata", "delta", "final"]
    assert events[0].data["social"] is True
    assert events[-1].data["answer"] == "\u0647\u0644\u0627 \u0648\u063a\u0644\u0627\u060c \u0623\u0646\u0627 \u0623\u0628\u0648 \u0633\u0647\u0644. \u0648\u0634 \u062d\u0627\u0628 \u062a\u0639\u0631\u0641 \u0639\u0646 \u0646\u0638\u0627\u0645 \u0645\u0648\u0641\u0642\u061f"
