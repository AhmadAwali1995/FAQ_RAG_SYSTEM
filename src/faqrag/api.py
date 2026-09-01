"""FastAPI service exposing the FAQ RAG pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .ingest import IngestionError
from .logging_utils import configure_logging
from .messages import MessageService, MessageStore, build_messages_router, to_speech_text
from .models import QueryResponse, RetrieveResponse
from .pipeline import RagPipeline

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {"pipeline": None, "error": None, "messages": None}


API_DESCRIPTION = """\
Bilingual (Arabic / English) question answering over the **Mwfaq** FAQ knowledge base.

`POST /query` supports two modes:

* Default JSON mode for a single final answer.
* Low-latency SSE mode with `stream=true`, which emits metadata, text deltas,
  and a final structured answer so voice playback can begin early.
"""

TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Liveness, and which index and models are actually loaded.",
    },
    {
        "name": "Ask",
        "description": (
            "Answer a question in one call. Use JSON mode for a final answer, or "
            "`stream=true` for `text/event-stream` output that flushes deltas as "
            "soon as the model produces them."
        ),
    },
    {
        "name": "Integration (async)",
        "description": (
            "The pair to integrate an external system against. `POST /v1/messages` "
            "accepts the text in ~40ms and returns a `message_id`; "
            "`GET /v1/messages/{id}` collects the answer once it is ready."
        ),
    },
    {
        "name": "Debug",
        "description": "Retrieval internals, for diagnosing ranking and confidence.",
    },
]

QUESTION_EXAMPLES = {
    "arabic_dialect": {
        "summary": "Arabic (Saudi dialect)",
        "description": "A colloquial voice transcript.",
        "value": {"text": "وش طرق الدفع عندكم؟"},
    },
    "arabic_msa": {
        "summary": "Arabic (Modern Standard)",
        "value": {"text": "ما هي أكاديمية موفق؟"},
    },
    "english": {
        "summary": "English",
        "value": {"text": "What is Mwfaq Business?"},
    },
    "streaming_voice": {
        "summary": "Streaming voice mode",
        "value": {
            "text": "وش طرق الدفع عندكم؟",
            "stream": True,
            "session": {"session_id": "sess_123"},
            "user": {"user_id": "user_456"},
        },
    },
}


class QueryRequest(BaseModel):
    """Body of a ``POST /query`` or ``POST /retrieve`` request."""

    question: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description="Legacy alias. Prefer `text` for voice integrations.",
    )
    text: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description="The user transcript, in Arabic or English.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="How many FAQ chunks to retrieve. Defaults to the configured value.",
    )
    session: dict[str, Any] | None = Field(
        default=None,
        description="Optional session metadata for caller-side tracking.",
    )
    user: dict[str, Any] | None = Field(
        default=None,
        description="Optional user metadata for caller-side tracking.",
    )
    stream: bool = Field(
        default=False,
        description="When true, return `text/event-stream` instead of JSON.",
    )

    def prompt_text(self) -> str:
        """Return whichever field carried the user's utterance."""
        return (self.text or self.question or "").strip()


class HealthResponse(BaseModel):
    """Body of a ``GET /health`` response."""

    status: str = Field(description='"ok" when the index is loaded, otherwise "unhealthy".')
    indexed_chunks: int = Field(description="Number of FAQ chunks in the loaded index.")
    embedding_model: str
    llm_model: str
    vector_store: str
    rerank_enabled: bool
    answer_style: str = Field(description='Answer voice: "saudi" or "msa".')
    messages: dict[str, int] = Field(default_factory=dict)
    detail: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        _state["pipeline"] = RagPipeline.from_settings(settings)
        logger.info("pipeline ready: %d chunks indexed", _state["pipeline"].retriever.store.count())
    except (IngestionError, RuntimeError) as exc:
        _state["error"] = str(exc)
        logger.error("failed to load pipeline: %s", exc)

    _state["messages"] = MessageService(
        store=MessageStore(
            ttl_seconds=settings.message_ttl_seconds,
            max_messages=settings.message_max_stored,
        ),
        pipeline_getter=_get_pipeline,
        max_workers=settings.message_workers,
    )
    _state["messages"].broadcaster.bind_loop(asyncio.get_running_loop())
    app.include_router(build_messages_router(_state["messages"]))

    yield

    service = _state.get("messages")
    if service is not None:
        service.shutdown()
    _state.clear()


app = FastAPI(
    title="Mwfaq FAQ RAG API",
    description=API_DESCRIPTION,
    version="0.1.0",
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)

ALLOWED_ORIGINS = [
    "https://abusahel.ahmadawali1995.workers.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_pipeline() -> RagPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=_state.get("error") or "pipeline is not loaded; run python -m faqrag.index",
        )
    return pipeline


def _sse_frame(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    settings = get_settings()
    pipeline = _state.get("pipeline")
    service = _state.get("messages")
    message_stats = service.store.stats() if service is not None else {}

    if pipeline is None:
        return HealthResponse(
            status="unhealthy",
            indexed_chunks=0,
            embedding_model=settings.embedding_model,
            llm_model=settings.llm_model,
            vector_store=settings.vector_store,
            rerank_enabled=settings.rerank_enabled,
            answer_style=settings.answer_style,
            messages=message_stats,
            detail=_state.get("error") or "index not loaded",
        )
    return HealthResponse(
        status="ok",
        indexed_chunks=pipeline.retriever.store.count(),
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        vector_store=settings.vector_store,
        rerank_enabled=settings.rerank_enabled,
        answer_style=settings.answer_style,
        messages=message_stats,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Ask"],
    summary="Ask a question and get JSON or streamed SSE output",
)
async def query(
    request_obj: Request,
    request: QueryRequest = Body(..., openapi_examples=QUESTION_EXAMPLES),
) -> QueryResponse | StreamingResponse:
    pipeline = _get_pipeline()
    question = request.prompt_text()
    if not question:
        raise HTTPException(status_code=422, detail="text must not be empty")

    wants_stream = request.stream or request_obj.query_params.get("stream") in {"1", "true", "yes"}

    try:
        if not wants_stream:
            return pipeline.answer(question, request.top_k)

        async def event_stream():
            try:
                for event in pipeline.stream_answer(question, request.top_k):
                    data = event.data.copy()
                    if event.event == "delta" and isinstance(data.get("text"), str):
                        data["speech_text"] = to_speech_text(data["text"])
                    if event.event == "final" and isinstance(data.get("answer"), str):
                        data["speech_answer"] = to_speech_text(data["answer"])
                    yield _sse_frame(event.event, data)
                    await asyncio.sleep(0)
            except Exception as exc:  # noqa: BLE001
                logger.exception("streaming query failed: %s", question)
                yield _sse_frame("error", {"message": str(exc)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("query failed: %s", question)
        raise HTTPException(status_code=500, detail=f"query failed: {exc}") from exc


@app.post("/retrieve", response_model=RetrieveResponse, tags=["Debug"])
def retrieve(
    request: QueryRequest = Body(..., openapi_examples=QUESTION_EXAMPLES),
) -> RetrieveResponse:
    pipeline = _get_pipeline()
    result = pipeline.retrieve(request.prompt_text(), request.top_k)
    return RetrieveResponse(
        query=result.query,
        language=result.query_lang,
        confident=result.confident,
        threshold=result.threshold,
        reranked=result.reranked,
        cross_lingual_fallback=result.cross_lingual_fallback,
        chunks=[
            {
                "faq_id": item.chunk.faq_id,
                "chunk_id": item.chunk.chunk_id,
                "category": item.chunk.category,
                "lang": item.chunk.lang,
                "question": item.chunk.question,
                "answer": item.chunk.answer,
                "rank_score": round(item.score, 4),
                "relevance": round(item.relevance or 0.0, 4),
                "vector_score": item.vector_score,
                "lexical_score": item.lexical_score,
                "rerank_score": item.rerank_score,
            }
            for item in result.chunks
        ],
    )


try:
    from .web import mount_chat_ui

    _state["chat_ui"] = mount_chat_ui(app, get_settings())
except ImportError:
    _state["chat_ui"] = False


def main() -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "faqrag.api:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
