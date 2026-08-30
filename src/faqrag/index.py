"""Build the vector index from the FAQ JSON.

Run after any change to the source FAQ data, the chunking logic, or the
embedding model::

    python -m faqrag.index
    python -m faqrag.index --source data/mwfaq_faq_rag.json --force
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings, get_settings
from .embeddings import build_embedder
from .ingest import load_chunks, summarise
from .logging_utils import configure_logging, ensure_utf8_stdout
from .models import Chunk
from .stores import build_vector_store

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def build_index(settings: Settings, source: Path | None = None) -> dict:
    """Embed every chunk and persist the vector store.

    Args:
        settings: Runtime configuration.
        source: Overrides ``settings.data_path`` when given.

    Returns:
        The manifest describing the index that was built.
    """
    source = source or settings.data_path
    chunks: list[Chunk] = load_chunks(source)
    stats = summarise(chunks)
    logger.info(
        "loaded %d chunks (%d FAQs) from %s: %s",
        stats["chunks"],
        stats["faqs"],
        source,
        stats["by_lang"],
    )

    embedder = build_embedder(settings)
    started = time.perf_counter()
    vectors = embedder.embed_documents([chunk.embedding_text for chunk in chunks])
    elapsed = time.perf_counter() - started
    logger.info(
        "embedded %d chunks in %.1fs with %s (dim=%d)",
        len(vectors),
        elapsed,
        embedder.name,
        len(vectors[0]),
    )

    store = build_vector_store(settings)
    store.add(chunks, vectors)
    store.persist()

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "vector_store": settings.vector_store,
        "collection": settings.collection_name,
        "embedding_model": embedder.name,
        "dimension": len(vectors[0]),
        "embed_seconds": round(elapsed, 2),
        **stats,
    }
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    (settings.index_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("index ready at %s", settings.index_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m faqrag.index``."""
    parser = argparse.ArgumentParser(description="Build the FAQ vector index.")
    parser.add_argument("--source", type=Path, default=None, help="Override the FAQ JSON path.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if an index already exists (rebuilds are the default).",
    )
    args = parser.parse_args(argv)

    ensure_utf8_stdout()
    settings = get_settings()
    configure_logging(settings.log_level)

    manifest = build_index(settings, args.source)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
