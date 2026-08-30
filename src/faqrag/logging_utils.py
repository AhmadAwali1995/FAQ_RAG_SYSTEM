"""Logging setup and the per-query retrieval trace.

Every query appends one JSON line to ``logs/retrieval.jsonl`` recording the
detected language, each candidate's individual and fused scores, whether it was
reranked, and which chunks reached the generator.  That trace is what makes a
bad answer debuggable after the fact instead of a black box.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import RetrievalResult

TRACE_FILENAME = "retrieval.jsonl"

_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, with UTF-8-safe stderr output.

    Windows consoles default to cp1252, which raises on Arabic text; the stream
    is reconfigured so log records containing Arabic never crash the process.
    """
    global _configured
    if _configured:
        return

    stream = sys.stderr
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - stream dependent
            pass

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=stream,
    )
    # Chroma and httpx are chatty at INFO and drown out retrieval traces.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    _configured = True


def ensure_utf8_stdout() -> None:
    """Make ``sys.stdout`` UTF-8 so Arabic answers print on Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - stream dependent
            pass


def build_trace(result: RetrievalResult, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the JSON-serialisable trace record for one retrieval pass."""
    trace: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": result.query,
        "query_lang": result.query_lang,
        "confident": result.confident,
        "threshold": round(result.threshold, 4),
        "reranked": result.reranked,
        "cross_lingual_fallback": result.cross_lingual_fallback,
        "candidates": [
            {
                "chunk_id": item.chunk.chunk_id,
                "faq_id": item.chunk.faq_id,
                "lang": item.chunk.lang,
                "category": item.chunk.category,
                "question": item.chunk.question,
                "score": round(item.score, 4),
                "vector_score": None if item.vector_score is None else round(item.vector_score, 4),
                "lexical_score": (
                    None if item.lexical_score is None else round(item.lexical_score, 4)
                ),
                "vector_rank": item.vector_rank,
                "lexical_rank": item.lexical_rank,
                "rerank_score": item.rerank_score,
                "language_boosted": item.language_boosted,
            }
            for item in result.chunks
        ],
    }
    if extra:
        trace.update(extra)
    return trace


def write_trace(log_dir: Path, trace: dict[str, Any]) -> None:
    """Append ``trace`` to the retrieval log, never raising on IO failure."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / TRACE_FILENAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    except OSError as exc:  # pragma: no cover - filesystem dependent
        logging.getLogger(__name__).warning("could not write retrieval trace: %s", exc)


def log_retrieval_summary(logger: logging.Logger, result: RetrievalResult) -> None:
    """Emit a human-readable one-line-per-candidate summary at INFO level."""
    logger.info(
        "query=%r lang=%s confident=%s reranked=%s cross_lingual=%s candidates=%d",
        result.query,
        result.query_lang,
        result.confident,
        result.reranked,
        result.cross_lingual_fallback,
        len(result.chunks),
    )
    for rank, item in enumerate(result.chunks, start=1):
        logger.info(
            "  #%d %s score=%.3f vec=%s bm25=%s rerank=%s | %s",
            rank,
            item.chunk.chunk_id,
            item.score,
            "-" if item.vector_score is None else f"{item.vector_score:.3f}",
            "-" if item.lexical_score is None else f"{item.lexical_score:.3f}",
            "-" if item.rerank_score is None else f"{item.rerank_score:.1f}",
            item.chunk.question[:70],
        )
