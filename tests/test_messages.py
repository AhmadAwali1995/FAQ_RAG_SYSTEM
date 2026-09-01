"""Tests for the asynchronous message API.

The pipeline is stubbed throughout: what matters here is the hand-off contract --
that a message is accepted instantly, answered exactly once, collectable
afterwards, and that a failing pipeline surfaces as a failed message rather than
a hung one.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from faqrag.messages import (
    MessageService,
    MessageStore,
    build_messages_router,
    to_result,
    to_speech_text,
)
from faqrag.models import QueryResponse, SourceCitation


def make_response(answer: str = "Answer.", confident: bool = True) -> QueryResponse:
    return QueryResponse(
        answer=answer,
        cited_faq_ids=["011"],
        sources=[
            SourceCitation(
                faq_id="011", category="Payments & Support", lang="ar",
                question="ما هي طرق الدفع؟", answer="تابي، تمارا، مدى.", score=0.9,
            )
        ],
        confidence=0.9,
        confident=confident,
        language="ar",
        query="q",
        latency_ms=1234.5,
    )


class StubPipeline:
    """A pipeline stand-in with controllable latency and failure."""

    def __init__(self, delay: float = 0.0, error: Exception | None = None) -> None:
        self.delay = delay
        self.error = error
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def answer(self, question: str, top_k=None) -> QueryResponse:
        with self._lock:
            self.calls.append(question)
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return make_response(f"answered: {question}")


@pytest.fixture
def store() -> MessageStore:
    return MessageStore(ttl_seconds=60.0, max_messages=100)


class TestMessageStore:
    """Storage, lifecycle, and eviction."""

    def test_create_starts_queued_with_unique_id(self, store: MessageStore) -> None:
        a = store.create("first")
        b = store.create("second")
        assert a.status == "queued"
        assert a.message_id != b.message_id
        assert a.message_id.startswith("msg_")

    def test_lifecycle_transitions(self, store: MessageStore) -> None:
        message = store.create("q")
        store.mark_processing(message.message_id)
        assert store.get(message.message_id).status == "processing"

        store.complete(message.message_id, make_response())
        stored = store.get(message.message_id)
        assert stored.status == "done"
        assert stored.terminal
        assert stored.response.answer == "Answer."

    def test_failure_is_recorded(self, store: MessageStore) -> None:
        message = store.create("q")
        store.fail(message.message_id, "model down")
        stored = store.get(message.message_id)
        assert stored.status == "failed"
        assert stored.error == "model down"
        assert stored.terminal

    def test_completion_releases_waiters(self, store: MessageStore) -> None:
        """A GET with wait= must unblock the instant the answer lands."""
        message = store.create("q")
        released = threading.Event()

        def waiter() -> None:
            message._finished.wait(timeout=5)
            released.set()

        threading.Thread(target=waiter, daemon=True).start()
        time.sleep(0.05)
        assert not released.is_set()

        store.complete(message.message_id, make_response())
        assert released.wait(timeout=2), "waiter was not released on completion"

    def test_failure_also_releases_waiters(self, store: MessageStore) -> None:
        """Otherwise a failed message would hang every long-poll against it."""
        message = store.create("q")
        store.fail(message.message_id, "boom")
        assert message._finished.is_set()

    def test_unknown_id_returns_none(self, store: MessageStore) -> None:
        assert store.get("msg_nope") is None

    def test_list_recent_is_ordered_oldest_first(self, store: MessageStore) -> None:
        for text in ("a", "b", "c"):
            store.create(text)
        assert [m.text for m in store.list_recent()] == ["a", "b", "c"]

    def test_list_recent_respects_limit_and_keeps_newest(
        self, store: MessageStore
    ) -> None:
        """Backfill wants the tail of the transcript, not its head."""
        for i in range(10):
            store.create(f"m{i}")
        assert [m.text for m in store.list_recent(limit=3)] == ["m7", "m8", "m9"]

    def test_expired_completed_messages_are_evicted(self) -> None:
        store = MessageStore(ttl_seconds=0.0, max_messages=100)
        old = store.create("old")
        store.complete(old.message_id, make_response())
        # Backdate so it falls outside the (zero-length) TTL window.
        old.updated_at = datetime.now(timezone.utc) - timedelta(seconds=10)

        store.create("new")  # writes trigger the sweep
        assert store.get(old.message_id) is None

    def test_in_flight_messages_are_never_evicted(self) -> None:
        """Evicting a running job would lose an answer the caller is waiting for."""
        store = MessageStore(ttl_seconds=0.0, max_messages=1)
        running = store.create("running")
        store.mark_processing(running.message_id)

        for i in range(5):
            done = store.create(f"done-{i}")
            store.complete(done.message_id, make_response())

        assert store.get(running.message_id) is not None

    def test_capacity_evicts_oldest_completed(self) -> None:
        store = MessageStore(ttl_seconds=9999.0, max_messages=3)
        ids = []
        for i in range(6):
            message = store.create(f"m{i}")
            store.complete(message.message_id, make_response())
            ids.append(message.message_id)
        assert store.get(ids[-1]) is not None, "newest must survive"
        assert store.stats()["total"] <= 3

    def test_stats_counts_by_status(self, store: MessageStore) -> None:
        a = store.create("a")
        store.complete(a.message_id, make_response())
        store.create("b")
        stats = store.stats()
        assert stats["total"] == 2
        assert stats["done"] == 1
        assert stats["queued"] == 1


class TestMessageService:
    """Submission and background processing."""

    def test_answers_a_message(self, store: MessageStore) -> None:
        pipeline = StubPipeline()
        service = MessageService(store, lambda: pipeline, max_workers=2)
        message = service.submit("وش طرق الدفع؟")

        assert message._finished.wait(timeout=5)
        stored = store.get(message.message_id)
        assert stored.status == "done"
        assert stored.response.answer == "answered: وش طرق الدفع؟"
        service.shutdown()

    def test_pipeline_failure_marks_message_failed_not_hung(
        self, store: MessageStore
    ) -> None:
        """A crash must reach the caller as a failed status, never as silence."""
        service = MessageService(
            store, lambda: StubPipeline(error=RuntimeError("model down")), max_workers=1
        )
        message = service.submit("q")

        assert message._finished.wait(timeout=5), "failure did not release waiters"
        stored = store.get(message.message_id)
        assert stored.status == "failed"
        assert "model down" in stored.error
        service.shutdown()

    def test_each_message_is_processed_exactly_once(self, store: MessageStore) -> None:
        pipeline = StubPipeline(delay=0.02)
        service = MessageService(store, lambda: pipeline, max_workers=4)
        messages = [service.submit(f"q{i}") for i in range(12)]

        for message in messages:
            assert message._finished.wait(timeout=10)
        assert sorted(pipeline.calls) == sorted(f"q{i}" for i in range(12))
        service.shutdown()

    def test_submit_returns_before_processing_finishes(self, store: MessageStore) -> None:
        """The whole point of the split: accepting must not wait on generation."""
        service = MessageService(store, lambda: StubPipeline(delay=1.0), max_workers=1)
        started = time.perf_counter()
        message = service.submit("q")
        elapsed = time.perf_counter() - started

        assert elapsed < 0.5, f"submit blocked for {elapsed:.2f}s"
        assert message.status in ("queued", "processing")
        service.shutdown()


@pytest.fixture
def client_and_pipeline():
    """A TestClient wired to the message router over a stub pipeline."""
    pipeline = StubPipeline()
    service = MessageService(MessageStore(), lambda: pipeline, max_workers=2)
    app = FastAPI()
    app.include_router(build_messages_router(service))
    with TestClient(app) as client:
        yield client, pipeline, service
    service.shutdown()


class TestMessageEndpoints:
    """The HTTP contract the other side integrates against."""

    def test_wait_zero_accepts_immediately_with_202(self, client_and_pipeline) -> None:
        """wait=0 opts out of waiting and hands back an id to collect with."""
        client, _, _ = client_and_pipeline
        response = client.post(
            "/v1/messages", params={"wait": 0}, json={"text": "وش طرق الدفع؟"}
        )

        assert response.status_code == 202
        body = response.json()
        # Any status is legitimate: with a fast pipeline the worker can finish
        # before the response is serialised. That POST does not *wait* is proved
        # by test_submit_returns_before_processing_finishes, which uses a slow one.
        assert body["status"] in ("queued", "processing", "done")
        assert body["message_id"].startswith("msg_")
        assert body["poll_url"] == f"/v1/messages/{body['message_id']}"

    def test_get_returns_the_answer_once_ready(self, client_and_pipeline) -> None:
        client, _, _ = client_and_pipeline
        message_id = client.post("/v1/messages", json={"text": "hi"}).json()["message_id"]

        body = client.get(f"/v1/messages/{message_id}", params={"wait": 5}).json()
        assert body["status"] == "done"
        assert body["answer"] == "answered: hi"
        assert body["cited_faq_ids"] == ["011"]
        assert body["confident"] is True
        assert body["language"] == "ar"
        assert len(body["sources"]) == 1

    def test_answer_fields_are_null_until_done(self) -> None:
        """Callers must branch on status, so pending answers must not look empty."""
        store = MessageStore()
        message = store.create("q")
        result = to_result(message)
        assert result.status == "queued"
        assert result.answer is None
        assert result.confident is None
        assert result.cited_faq_ids == []

    def test_pending_get_sets_retry_after(self) -> None:
        service = MessageService(
            MessageStore(), lambda: StubPipeline(delay=5.0), max_workers=1
        )
        app = FastAPI()
        app.include_router(build_messages_router(service))
        with TestClient(app) as client:
            message_id = client.post(
                "/v1/messages", params={"wait": 0}, json={"text": "q"}
            ).json()["message_id"]
            response = client.get(f"/v1/messages/{message_id}")
            assert response.json()["status"] in ("queued", "processing")
            assert response.headers["Retry-After"] == "2"
        service.shutdown()

    def test_unknown_message_id_is_404(self, client_and_pipeline) -> None:
        client, _, _ = client_and_pipeline
        response = client.get("/v1/messages/msg_doesnotexist")
        assert response.status_code == 404
        assert "unknown message_id" in response.json()["detail"]

    def test_empty_text_is_rejected(self, client_and_pipeline) -> None:
        client, _, _ = client_and_pipeline
        assert client.post("/v1/messages", json={"text": ""}).status_code == 422
        assert client.post("/v1/messages", json={"text": "   "}).status_code == 422

    def test_oversized_text_is_rejected(self, client_and_pipeline) -> None:
        client, _, _ = client_and_pipeline
        response = client.post("/v1/messages", json={"text": "x" * 2001})
        assert response.status_code == 422

    def test_wait_is_capped(self, client_and_pipeline) -> None:
        client, _, _ = client_and_pipeline
        message_id = client.post("/v1/messages", json={"text": "q"}).json()["message_id"]
        assert client.get(f"/v1/messages/{message_id}", params={"wait": 9999}).status_code == 422

    def test_failed_message_reports_its_error(self) -> None:
        service = MessageService(
            MessageStore(), lambda: StubPipeline(error=RuntimeError("boom")), max_workers=1
        )
        app = FastAPI()
        app.include_router(build_messages_router(service))
        with TestClient(app) as client:
            message_id = client.post("/v1/messages", json={"text": "q"}).json()["message_id"]
            body = client.get(f"/v1/messages/{message_id}", params={"wait": 5}).json()
            assert body["status"] == "failed"
            assert "boom" in body["error"]
            assert body["answer"] is None
        service.shutdown()

    def test_history_backfill(self, client_and_pipeline) -> None:
        """A reconnecting client uses this to fill the gap the socket left."""
        client, _, _ = client_and_pipeline
        for text in ("first", "second", "third"):
            client.post("/v1/messages", json={"text": text})

        recent = client.get("/v1/history").json()
        assert [m["text"] for m in recent] == ["first", "second", "third"]

    def test_latest_returns_only_the_newest_message(self, client_and_pipeline) -> None:
        """The whole point: one object for the last question, not a transcript."""
        client, _, _ = client_and_pipeline
        for text in ("first", "second", "third"):
            client.post("/v1/messages", json={"text": text})

        latest = client.get("/v1/latest", params={"wait": 5}).json()
        assert isinstance(latest, dict), "must be a single message, not a list"
        assert latest["text"] == "third"
        assert latest["answer"] == "answered: third"

    def test_latest_as_plain_text(self, client_and_pipeline) -> None:
        client, _, _ = client_and_pipeline
        client.post("/v1/messages", json={"text": "hello"})

        response = client.get("/v1/latest", params={"wait": 5, "format": "text"})
        assert response.headers["content-type"].startswith("text/plain")
        # Speech-ready: a sentence terminator is added so a synthesiser pauses.
        assert response.text == "answered: hello."
        assert response.headers["X-Message-Status"] == "done"

    def test_latest_with_no_messages_is_404(self, client_and_pipeline) -> None:
        client, _, _ = client_and_pipeline
        response = client.get("/v1/latest")
        assert response.status_code == 404
        assert "no messages" in response.json()["detail"]

    def test_no_session_id_in_the_contract(self, client_and_pipeline) -> None:
        """session_id was removed; an unknown field must not resurrect it."""
        client, _, _ = client_and_pipeline
        body = client.post("/v1/messages", json={"text": "hi"}).json()
        assert "session_id" not in body

        result = client.get(f"/v1/messages/{body['message_id']}", params={"wait": 5}).json()
        assert "session_id" not in result

    def test_arabic_round_trips_intact(self, client_and_pipeline) -> None:
        client, _, _ = client_and_pipeline
        question = "وش طرق الدفع عندكم؟"
        message_id = client.post("/v1/messages", json={"text": question}).json()["message_id"]
        body = client.get(f"/v1/messages/{message_id}", params={"wait": 5}).json()
        assert body["text"] == question
        assert question in body["answer"]


class TestSpeechText:
    """The text/plain representation is spoken aloud by the caller's TTS.

    Markdown that reads fine on screen is noise out loud: a synthesiser either
    pronounces the markers or treats them as unnatural pauses.
    """

    def test_strips_emphasis_markers(self) -> None:
        assert to_speech_text("**مرحبا** و *أهلاً* و `كود`") == "مرحبا و أهلاً و كود."

    def test_strips_bullets_and_numbering(self) -> None:
        spoken = to_speech_text("- أول\n- ثاني\n1. ثالث\n2) رابع")
        assert spoken == "أول. ثاني. ثالث. رابع."

    def test_strips_headings(self) -> None:
        assert to_speech_text("## عنوان\nنص") == "عنوان. نص."

    def test_terminates_lines_so_speech_pauses(self) -> None:
        """Without a terminator a synthesiser runs the lines together."""
        assert to_speech_text("سطر أول\nسطر ثاني") == "سطر أول. سطر ثاني."

    def test_keeps_existing_punctuation(self) -> None:
        assert to_speech_text("سؤال؟\nجواب!") == "سؤال؟ جواب!"
        assert to_speech_text("جملة.") == "جملة."

    def test_drops_blank_lines(self) -> None:
        assert to_speech_text("أول\n\n\nثاني") == "أول. ثاني."

    def test_plain_sentence_survives_unchanged(self) -> None:
        text = "طرق الدفع: تابي، تمارا، مدى."
        assert to_speech_text(text) == text

    def test_english_is_handled_too(self) -> None:
        assert to_speech_text("- **First** item\n- Second item") == "First item. Second item."

    def test_pronounces_the_english_brand_as_arabic(self) -> None:
        assert to_speech_text("Mwfaq Business is ready.") == "موفق Business is ready."

    def test_empty_input(self) -> None:
        assert to_speech_text("") == ""


class TestOneCallFlow:
    """POST with wait=: the answer comes back in the same call.

    This is the flow to use when the caller can hold a request open for the
    ~10s a generation takes; it removes the message_id round-trip entirely.
    """

    def test_explicit_wait_returns_the_answer_inline_with_200(
        self, client_and_pipeline
    ) -> None:
        client, _, _ = client_and_pipeline
        response = client.post("/v1/messages", params={"wait": 10}, json={"text": "hi"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "done"
        assert body["answer"] == "answered: hi"
        assert "poll_url" not in body, "no need to poll when the answer is inline"

    def test_wait_with_text_format_returns_speech_ready_plain_text(
        self, client_and_pipeline
    ) -> None:
        client, _, _ = client_and_pipeline
        response = client.post(
            "/v1/messages", params={"wait": 10, "format": "text"}, json={"text": "hi"}
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == "answered: hi."
        assert response.headers["X-Answer-Confident"] == "true"

    def test_answer_comes_back_by_default_without_any_parameters(
        self, client_and_pipeline
    ) -> None:
        """The plain call is the one-call flow: no query string needed."""
        client, _, _ = client_and_pipeline
        response = client.post("/v1/messages", json={"text": "hi"})

        assert response.status_code == 200
        assert response.json()["answer"] == "answered: hi"

    def test_wait_zero_opts_out_and_returns_an_id(self, client_and_pipeline) -> None:
        client, _, _ = client_and_pipeline
        response = client.post("/v1/messages", params={"wait": 0}, json={"text": "hi"})

        assert response.status_code == 202
        assert response.json()["message_id"].startswith("msg_")

    def test_wait_that_elapses_degrades_to_the_async_flow(self) -> None:
        """A slow answer must fall back to 202, never fail the request."""
        service = MessageService(
            MessageStore(), lambda: StubPipeline(delay=5.0), max_workers=1
        )
        app = FastAPI()
        app.include_router(build_messages_router(service))
        with TestClient(app) as client:
            response = client.post("/v1/messages", params={"wait": 1}, json={"text": "hi"})
            assert response.status_code == 202
            body = response.json()
            assert body["message_id"].startswith("msg_")
            assert body["poll_url"] == f"/v1/messages/{body['message_id']}"
        service.shutdown()

    def test_wait_is_capped(self, client_and_pipeline) -> None:
        client, _, _ = client_and_pipeline
        response = client.post("/v1/messages", params={"wait": 9999}, json={"text": "hi"})
        assert response.status_code == 422
