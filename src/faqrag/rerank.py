"""Reranking of fused candidates before generation.

Hybrid fusion optimises for recall; reranking restores precision by scoring each
candidate against the query with a model that sees both texts together.

The cross-encoder route (``bge-reranker-v2-m3``) would need PyTorch, which this
deployment deliberately avoids, so the default reranker is LLM-based: one extra
call that scores all candidates at once.  It is a config toggle either way --
set ``FAQRAG_RERANK_ENABLED=false`` to drop the round-trip when latency matters,
and retrieval falls back to the fused ranking.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

from .config import Settings
from .llm import LLMClient, LLMError, build_llm
from .models import ScoredChunk
from .prompts import RERANK_SYSTEM_PROMPT, build_rerank_prompt

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_MAX_RERANK_SCORE = 10.0


class Reranker(ABC):
    """Reorders retrieved candidates by relevance to the query."""

    @abstractmethod
    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Return the ``top_k`` most relevant chunks, best first."""


class NoOpReranker(Reranker):
    """Passes the fused ranking through unchanged."""

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Return the first ``top_k`` chunks as-is."""
        return chunks[:top_k]


def parse_rerank_scores(raw: str, n_candidates: int) -> dict[int, float]:
    """Extract ``{candidate_index: score}`` from a model's JSON reply.

    Tolerates markdown fences and surrounding prose by scanning for the outermost
    JSON object.  Out-of-range indices and non-numeric scores are dropped rather
    than raising, so one malformed entry cannot fail the whole query.

    Args:
        raw: The model's raw response text.
        n_candidates: Number of candidates that were scored (1-indexed).

    Returns:
        Mapping of 1-based candidate index to a score in ``[0, 10]``.

    Raises:
        ValueError: If no JSON object can be found in ``raw``.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        raise ValueError(f"no JSON object in rerank response: {raw[:200]!r}")

    payload = json.loads(match.group(0))
    scores: dict[int, float] = {}
    for key, value in payload.items():
        try:
            index = int(str(key).strip().strip("[]"))
            score = float(value)
        except (TypeError, ValueError):
            logger.warning("skipping malformed rerank entry %r: %r", key, value)
            continue
        if 1 <= index <= n_candidates:
            scores[index] = max(0.0, min(_MAX_RERANK_SCORE, score))
    return scores


class LLMReranker(Reranker):
    """Scores every candidate in a single LLM call.

    On any failure -- unreachable model, unparseable JSON -- the fused ordering
    is returned unchanged.  A reranker outage degrades quality; it must not
    break the query.
    """

    def __init__(self, client: LLMClient, max_tokens: int | None = None) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Rescore ``chunks`` against ``query`` and return the best ``top_k``."""
        if len(chunks) <= 1:
            return chunks[:top_k]

        try:
            raw = self._client.complete(
                RERANK_SYSTEM_PROMPT,
                build_rerank_prompt(query, chunks),
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
            scores = parse_rerank_scores(raw, len(chunks))
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("rerank failed (%s); falling back to fused order", exc)
            return chunks[:top_k]

        if not scores:
            logger.warning("rerank returned no usable scores; keeping fused order")
            return chunks[:top_k]

        reranked = []
        for i, item in enumerate(chunks, start=1):
            # Candidates the model omitted keep their fused score, rescaled to
            # the rerank range, rather than being dropped to zero.
            score = scores.get(i)
            normalised = (
                score / _MAX_RERANK_SCORE if score is not None else item.score * 0.5
            )
            reranked.append(
                item.model_copy(update={"score": normalised, "rerank_score": score})
            )

        reranked.sort(key=lambda sc: (-sc.score, sc.chunk.chunk_id))
        return reranked[:top_k]


def build_reranker(settings: Settings) -> Reranker:
    """Build the reranker named by settings, or a no-op when disabled."""
    if not settings.rerank_enabled or settings.rerank_provider == "none":
        return NoOpReranker()

    try:
        client = build_llm(settings, model=settings.effective_rerank_model)
    except LLMError as exc:
        logger.warning("cannot build reranker (%s); reranking disabled", exc)
        return NoOpReranker()

    if client is None:  # extractive mode has no model to rerank with
        return NoOpReranker()
    return LLMReranker(client, max_tokens=settings.rerank_max_tokens)
