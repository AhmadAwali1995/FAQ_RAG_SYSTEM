"""FastAPI service exposing the FAQ RAG pipeline.

Start it with::

    python -m faqrag.api
    uvicorn faqrag.api:app --reload

Interactive documentation:
    ``/docs``          Swagger UI -- every endpoint, with runnable examples
    ``/redoc``         ReDoc, a reference-style rendering of the same schema
    ``/openapi.json``  the raw OpenAPI 3.1 document, for client generation

Endpoints:
    ``GET  /health``                        liveness plus index metadata
    ``POST /query``                         question in, answer out (synchronous)
    ``POST /retrieve``                      retrieval only, for debugging ranking
    ``POST /v1/messages``                   receive text from an external system
    ``GET  /v1/messages/{id}``              collect that message's answer
    ``WS   /v1/ws``                         live push of every message event
    ``GET  /v1/events``                     the same events as SSE (see caveat)
    ``GET  /v1/sessions/{id}/messages``     a session's message history

``/query`` answers in one call and blocks for the ~10s a full generation takes.
``/v1/messages`` is the pair to use from another system: it accepts instantly
and lets the caller collect the answer separately, so their request timeout does
not have to accommodate ours.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .ingest import IngestionError
from .logging_utils import configure_logging
from .messages import MessageService, MessageStore, build_messages_router
from .models import QueryResponse, RetrieveResponse
from .pipeline import RagPipeline

logger = logging.getLogger(__name__)

# Populated on startup so the index and embedding client are loaded once for the
# process rather than rebuilt per request.
_state: dict[str, Any] = {"pipeline": None, "error": None, "messages": None}


API_DESCRIPTION = """\
Bilingual (Arabic / English) question answering over the **Mwfaq** FAQ knowledge base.

Answers are generated **only** from retrieved FAQ entries and always cite the
`faq_id`s they came from. When nothing in the knowledge base covers the question,
the service says so instead of guessing.

### Choosing an endpoint

| You are | Use | Why |
|---|---|---|
| A server-side script that can wait ~10s | `POST /query` | One call, answer returned inline |
| A widget, app, or messaging bridge | `POST /v1/messages` then `GET /v1/messages/{id}` | Accepts in ~40ms; your timeout is independent of ours |
| A live screen that must react instantly | `POST /v1/messages` + **`WS /v1/ws`** | Pushed on arrival (~90ms), no polling |
| Debugging why an answer was wrong | `POST /retrieve` | Every ranking score, no LLM call |

### Two fields every client must handle

* **`confident`** — `false` means the FAQ did not cover the question. The `answer`
  is a polite "I don't have that" and `cited_faq_ids` is empty. Render it
  differently; do not present it as fact.
* **`status`** (async only) — answer fields stay `null` until it is `"done"`, so
  branch on the status rather than on whether `answer` is present.

### Notes

Answers are written in Saudi dialect for Arabic questions (`FAQRAG_ANSWER_STYLE`).
There is no authentication; the server binds to `127.0.0.1` by default.
"""

TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Liveness, and which index and models are actually loaded.",
    },
    {
        "name": "Ask (synchronous)",
        "description": (
            "Answer a question in a single call. **Blocks for roughly 10 seconds** "
            "while retrieval, reranking, and generation run. Good for server-side "
            "scripts; use the async endpoints for anything user-facing."
        ),
    },
    {
        "name": "Integration (async)",
        "description": (
            "The pair to integrate an external system against. `POST /v1/messages` "
            "accepts the text in ~40ms and returns a `message_id`; "
            "`GET /v1/messages/{id}` collects the answer once it is ready. "
            "Splitting them keeps the caller's request timeout independent of "
            "our generation time."
        ),
    },
    {
        "name": "Debug",
        "description": (
            "Retrieval internals, for working out whether a bad answer came from "
            "retrieval or from generation. Makes no LLM call."
        ),
    },
]

# Reusable example payloads, shown as a dropdown in Swagger's "Try it out".
QUESTION_EXAMPLES = {
    "arabic_dialect": {
        "summary": "Arabic (Saudi dialect)",
        "description": "A colloquial question. The answer comes back in dialect too.",
        "value": {"question": "وش طرق الدفع عندكم؟"},
    },
    "arabic_msa": {
        "summary": "Arabic (Modern Standard)",
        "value": {"question": "ما هي أكاديمية موفق؟"},
    },
    "english": {
        "summary": "English",
        "value": {"question": "What is Mwfaq Business?"},
    },
    "out_of_scope": {
        "summary": "Out of scope (expect confident=false)",
        "description": "Pricing is not in the knowledge base, so the service declines.",
        "value": {"question": "بكم الفحص الطبي؟"},
    },
}


class QueryRequest(BaseModel):
    """Body of a ``POST /query`` or ``POST /retrieve`` request."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question, in Arabic or English. Language is detected automatically.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="How many FAQ chunks to retrieve. Defaults to the configured value (5).",
    )


class HealthResponse(BaseModel):
    """Body of a ``GET /health`` response."""

    status: str = Field(description='"ok" when the index is loaded, otherwise "unhealthy".')
    indexed_chunks: int = Field(description="Number of FAQ chunks in the loaded index.")
    embedding_model: str
    llm_model: str
    vector_store: str
    rerank_enabled: bool
    answer_style: str = Field(description='Answer voice: "saudi" or "msa".')
    messages: dict[str, int] = Field(
        default_factory=dict, description="Async message counts by status."
    )
    detail: str | None = Field(default=None, description="Why the service is unhealthy, if it is.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline once at startup.

    A failure here is captured rather than raised so the service still starts
    and ``/health`` can report *why* it is unhealthy -- a process that refuses
    to boot tells you far less than one that explains itself.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        _state["pipeline"] = RagPipeline.from_settings(settings)
        logger.info("pipeline ready: %d chunks indexed", _state["pipeline"].retriever.store.count())
    except (IngestionError, RuntimeError) as exc:
        _state["error"] = str(exc)
        logger.error("failed to load pipeline: %s", exc)

    # The async message API shares the one pipeline; workers are bounded so a
    # burst of inbound messages queues instead of spawning unbounded threads.
    _state["messages"] = MessageService(
        store=MessageStore(
            ttl_seconds=settings.message_ttl_seconds,
            max_messages=settings.message_max_stored,
        ),
        pipeline_getter=_get_pipeline,
        max_workers=settings.message_workers,
    )
    # Worker threads publish SSE events into this loop, so bind it before any
    # message can be accepted.
    _state["messages"].broadcaster.bind_loop(asyncio.get_running_loop())
    app.include_router(build_messages_router(_state["messages"]))
    logger.info("message API mounted at /v1/messages (%d workers)", settings.message_workers)
    logger.info("live event stream at /v1/events")

    # The chat UI mounts at import time, before logging is configured, so its
    # outcome is reported here where it is actually visible.
    if _state.get("chat_ui"):
        logger.info("chat UI: http://%s:%d/chat", settings.api_host, settings.api_port)
    elif settings.enable_chat_ui:
        logger.warning("chat UI enabled but not mounted (missing %s)", settings.web_dir)

    logger.info("API docs: http://%s:%d/docs", settings.api_host, settings.api_port)

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
    contact={"name": "Mwfaq FAQ RAG", "url": "http://127.0.0.1:8000/docs"},
    license_info={"name": "Internal use"},
    servers=[{"url": "/", "description": "This server"}],
    swagger_ui_parameters={
        "defaultModelsExpandDepth": 2,
        "displayRequestDuration": True,
        "docExpansion": "list",
        "tryItOutEnabled": True,
    },
)


def _get_pipeline() -> RagPipeline:
    """Return the loaded pipeline, or raise 503 with the startup error."""
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=_state.get("error") or "pipeline is not loaded; run python -m faqrag.index",
        )
    return pipeline


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service health and active configuration",
    response_description="Health status plus the index and models actually in use.",
)
def health() -> HealthResponse:
    """Report service health and the active model/index configuration.

    Always returns `200`. Check the `status` field: `"unhealthy"` with a
    `detail` explaining why usually means the index has not been built
    (`python -m faqrag.index`).
    """
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
    tags=["Ask (synchronous)"],
    summary="Ask a question and get the answer in the same call",
    response_description="The grounded answer, its citations, and every retrieved chunk.",
    responses={
        422: {"description": "The question was empty or too long."},
        500: {"description": "Generation failed. The retrieval trace is in logs/retrieval.jsonl."},
        503: {"description": "No index loaded. Run `python -m faqrag.index`."},
    },
)
def query(
    request: QueryRequest = Body(..., openapi_examples=QUESTION_EXAMPLES),
) -> QueryResponse:
    """Answer a question from the FAQ corpus, blocking until the answer is ready.

    **This takes about 10 seconds** — retrieval, an LLM rerank, then generation.
    Set your client timeout to at least 60s, or use `POST /v1/messages` instead.

    The response carries the answer, the cited FAQ ids, the cited source entries,
    **every** retrieved chunk, a confidence score, and the detected language.

    `sources` holds only what the model cited — render these as citations.
    `retrieved` holds everything retrieval surfaced, cited or not, and is
    populated even on a refusal so a wrong decline is diagnosable.

    **Check `confident` before displaying the answer.** When it is `false`,
    nothing cleared the relevance threshold, `cited_faq_ids` is empty, and the
    answer is an "I don't have that" message in the user's language.
    """
    pipeline = _get_pipeline()
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    try:
        return pipeline.answer(question, request.top_k)
    except Exception as exc:  # noqa: BLE001 - surface a clean 500, log the trace
        logger.exception("query failed: %s", question)
        raise HTTPException(status_code=500, detail=f"query failed: {exc}") from exc


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["Debug"],
    summary="Retrieval only — every score, no answer generated",
    response_description="Ranked candidates with their vector, BM25, and rerank scores.",
    responses={503: {"description": "No index loaded. Run `python -m faqrag.index`."}},
)
def retrieve(
    request: QueryRequest = Body(..., openapi_examples=QUESTION_EXAMPLES),
) -> RetrieveResponse:
    """Run retrieval only, returning every score that produced the ranking.

    Use this to tell a **retrieval** fault from a **generation** fault without
    paying for an LLM call. Fast (~2s) because nothing is generated.

    Two scores are reported per candidate and must not be confused:

    * `rank_score` orders candidates *within this query*, scaled so the best is
      `1.0`. The top candidate of a completely out-of-scope query still scores
      `1.0` — it is the best of a bad set.
    * `relevance` is absolute and comparable across queries. This is the one
      compared against `threshold` to decide whether to answer at all.
    """
    pipeline = _get_pipeline()
    result = pipeline.retrieve(request.question.strip(), request.top_k)
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


# --- Optional chat UI ------------------------------------------------------
# The browser UI is a self-contained add-on: this guarded import is the only
# place the core API references it. Delete `web/` and `src/faqrag/web.py` to
# remove it entirely, or set FAQRAG_ENABLE_CHAT_UI=false to switch it off. The
# API behaves identically either way. Its routes are excluded from the OpenAPI
# schema -- it is a page, not part of the API contract.
try:
    from .web import mount_chat_ui

    _state["chat_ui"] = mount_chat_ui(app, get_settings())
except ImportError:
    _state["chat_ui"] = False


def main() -> int:
    """Run the service with uvicorn (``python -m faqrag.api``)."""
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
