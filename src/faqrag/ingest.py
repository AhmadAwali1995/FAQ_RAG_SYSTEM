"""Load the FAQ JSON and turn it into retrievable chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import Chunk, FaqRecord


class IngestionError(ValueError):
    """Raised when the source FAQ file is malformed."""


def make_chunk_id(faq_id: str, lang: str) -> str:
    """Build the stable chunk identifier for a ``(faq_id, lang)`` pair."""
    return f"{faq_id}::{lang}"


def records_to_chunks(records: Iterable[FaqRecord]) -> list[Chunk]:
    """Convert FAQ records into chunks, one per ``(faq_id, lang)`` pair.

    Arabic and English entries are never merged: a mixed-script chunk both
    degrades multilingual embedding quality and makes language-aware retrieval
    impossible.

    Raises:
        IngestionError: If two records share a ``(faq_id, lang)`` pair.
    """
    chunks: list[Chunk] = []
    seen: set[str] = set()

    for record in records:
        chunk_id = make_chunk_id(record.faq_id, record.lang)
        if chunk_id in seen:
            raise IngestionError(f"duplicate chunk id {chunk_id!r} in source data")
        seen.add(chunk_id)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                faq_id=record.faq_id,
                category=record.category,
                lang=record.lang,
                question=record.question,
                answer=record.answer,
                keywords=list(record.keywords),
            )
        )

    return chunks


def parse_faq_payload(payload: dict[str, Any]) -> list[Chunk]:
    """Validate a decoded FAQ JSON payload and return its chunks."""
    if "faqs" not in payload:
        raise IngestionError("source JSON has no top-level 'faqs' key")
    if not isinstance(payload["faqs"], list):
        raise IngestionError("'faqs' must be a list")

    records = []
    for i, raw in enumerate(payload["faqs"]):
        try:
            records.append(FaqRecord.model_validate(raw))
        except Exception as exc:  # pragma: no cover - message detail only
            raise IngestionError(f"invalid FAQ record at index {i}: {exc}") from exc

    chunks = records_to_chunks(records)
    if not chunks:
        raise IngestionError("source JSON contains no FAQ records")
    return chunks


def load_chunks(path: Path) -> list[Chunk]:
    """Load and validate the FAQ JSON at ``path``.

    Args:
        path: Path to a JSON file matching the ``{"faqs": [...]}`` schema.

    Returns:
        One :class:`Chunk` per ``(faq_id, lang)`` pair.

    Raises:
        IngestionError: If the file is missing, unparseable, or malformed.
    """
    if not path.exists():
        raise IngestionError(
            f"FAQ source not found at {path}. Generate it with:\n"
            f"  python scripts/md_to_json.py data/mwfaq_faq_rag.md {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestionError(f"{path} is not valid JSON: {exc}") from exc

    return parse_faq_payload(payload)


def summarise(chunks: list[Chunk]) -> dict[str, Any]:
    """Return counts by language and category, for indexing logs."""
    by_lang: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for chunk in chunks:
        by_lang[chunk.lang] = by_lang.get(chunk.lang, 0) + 1
        by_category[chunk.category] = by_category.get(chunk.category, 0) + 1
    return {
        "chunks": len(chunks),
        "faqs": len({c.faq_id for c in chunks}),
        "by_lang": by_lang,
        "by_category": by_category,
    }
