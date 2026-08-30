"""Command-line interface for querying the FAQ RAG system.

Examples::

    python -m faqrag.cli "How do I book a medical exam?"
    python -m faqrag.cli "ما هي أكاديمية موفق؟" --json
    python -m faqrag.cli "What is Mwfaq Business?" --retrieve-only --no-rerank
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import get_settings
from .ingest import IngestionError
from .logging_utils import configure_logging, ensure_utf8_stdout
from .pipeline import RagPipeline

# Arabic answers must be printed as one block; terminals handle the bidi
# reordering themselves.
_SEPARATOR = "-" * 72


def _print_human(response, show_sources: bool) -> None:
    print(f"\n{response.answer}\n")
    print(_SEPARATOR)
    print(
        f"language: {response.language}   confidence: {response.confidence:.3f}   "
        f"confident: {response.confident}   reranked: {response.reranked}   "
        f"{response.latency_ms:.0f} ms"
    )
    if response.cited_faq_ids:
        print(f"cited FAQs: {', '.join(response.cited_faq_ids)}")
    if show_sources and response.sources:
        print(_SEPARATOR)
        for source in response.sources:
            print(f"  [{source.faq_id}] ({source.lang}, {source.score:.3f}) {source.question}")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m faqrag.cli``."""
    parser = argparse.ArgumentParser(
        prog="python -m faqrag.cli",
        description="Ask the Mwfaq FAQ knowledge base a question in Arabic or English.",
    )
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="Chunks to retrieve.")
    parser.add_argument("--json", action="store_true", help="Emit the full JSON response.")
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Show retrieved chunks and scores without calling the LLM.",
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="Disable reranking for this query."
    )
    parser.add_argument("--sources", action="store_true", help="Print the cited FAQ entries.")
    parser.add_argument("--verbose", action="store_true", help="Log retrieval scores to stderr.")
    args = parser.parse_args(argv)

    ensure_utf8_stdout()
    settings = get_settings()
    configure_logging("DEBUG" if args.verbose else "WARNING")
    if args.no_rerank:
        settings.rerank_enabled = False

    try:
        pipeline = RagPipeline.from_settings(settings)
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.retrieve_only:
        result = pipeline.retrieve(args.question, args.top_k)
        payload = {
            "query": result.query,
            "language": result.query_lang,
            "confident": result.confident,
            "threshold": result.threshold,
            "cross_lingual_fallback": result.cross_lingual_fallback,
            "chunks": [
                {
                    "faq_id": c.chunk.faq_id,
                    "lang": c.chunk.lang,
                    "question": c.chunk.question,
                    "rank_score": round(c.score, 4),
                    "relevance": round(c.relevance or 0.0, 4),
                    "vector_score": c.vector_score,
                    "lexical_score": c.lexical_score,
                    "rerank_score": c.rerank_score,
                }
                for c in result.chunks
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    response = pipeline.answer(args.question, args.top_k)
    if args.json:
        print(response.model_dump_json(indent=2))
    else:
        _print_human(response, show_sources=args.sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
