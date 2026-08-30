"""Combine vector and lexical result lists into one ranking.

Two strategies are offered:

* **Reciprocal rank fusion (RRF)** -- the default. It consumes only *ranks*, so
  it needs no calibration between cosine similarity and unbounded BM25 scores.
  This robustness is why it is the default for a corpus this small, where score
  distributions shift noticeably as the FAQ set grows.
* **Weighted fusion** -- min-max normalises each list to ``[0, 1]`` and takes a
  weighted sum. More tunable, but sensitive to outliers in short result lists.

Both emit scores normalised to ``[0, 1]`` so a single relevance threshold in
:mod:`faqrag.config` applies regardless of the method chosen.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .bm25 import BM25Hit
from .models import Chunk, ScoredChunk
from .stores.base import VectorHit


def min_max_normalise(values: Sequence[float]) -> list[float]:
    """Scale ``values`` into ``[0, 1]``.

    A list whose values are all equal maps to all ``1.0`` -- every item is
    equally and maximally relevant within that list, and returning zeros would
    silently discard an otherwise usable result set.
    """
    if not values:
        return []
    lowest, highest = min(values), max(values)
    if highest - lowest < 1e-12:
        return [1.0 for _ in values]
    return [(v - lowest) / (highest - lowest) for v in values]


def _index_by_chunk(
    vector_hits: Sequence[VectorHit], lexical_hits: Sequence[BM25Hit]
) -> dict[str, Chunk]:
    chunks: dict[str, Chunk] = {}
    for hit in vector_hits:
        chunks[hit.chunk.chunk_id] = hit.chunk
    for hit in lexical_hits:
        chunks.setdefault(hit.chunk.chunk_id, hit.chunk)
    return chunks


def reciprocal_rank_fusion(
    vector_hits: Sequence[VectorHit],
    lexical_hits: Sequence[BM25Hit],
    k: int = 60,
) -> list[ScoredChunk]:
    """Fuse two ranked lists by reciprocal rank.

    Each list contributes ``1 / (k + rank)`` for the documents it ranks, with
    ``rank`` starting at 1.  The damping constant ``k`` limits how much a single
    top-1 placement can dominate.

    Args:
        vector_hits: Results from the vector store, best first.
        lexical_hits: Results from BM25, best first.
        k: RRF damping constant.

    Returns:
        Chunks sorted by fused score, normalised so the best scores ``1.0``.
    """
    chunks = _index_by_chunk(vector_hits, lexical_hits)
    raw: dict[str, float] = {cid: 0.0 for cid in chunks}

    vector_ranks = {hit.chunk.chunk_id: i + 1 for i, hit in enumerate(vector_hits)}
    lexical_ranks = {hit.chunk.chunk_id: i + 1 for i, hit in enumerate(lexical_hits)}
    vector_scores = {hit.chunk.chunk_id: hit.score for hit in vector_hits}
    lexical_scores = {hit.chunk.chunk_id: hit.score for hit in lexical_hits}

    for chunk_id, rank in vector_ranks.items():
        raw[chunk_id] += 1.0 / (k + rank)
    for chunk_id, rank in lexical_ranks.items():
        raw[chunk_id] += 1.0 / (k + rank)

    # Raw RRF sums are returned deliberately. Dividing by the theoretical
    # maximum (2/(k+1)) would squeeze every candidate into a narrow band -- with
    # k=60 and 20 candidates the whole range is [0.75, 1.0] -- which destroys
    # both the ranking spread and any absolute meaning. Callers normalise for
    # display via normalise_ranking_scores() and judge confidence separately.
    fused = [
        ScoredChunk(
            chunk=chunks[chunk_id],
            score=raw[chunk_id],
            vector_score=vector_scores.get(chunk_id),
            lexical_score=lexical_scores.get(chunk_id),
            vector_rank=vector_ranks.get(chunk_id),
            lexical_rank=lexical_ranks.get(chunk_id),
        )
        for chunk_id in chunks
    ]
    fused.sort(key=lambda sc: (-sc.score, sc.chunk.chunk_id))
    return fused


def weighted_fusion(
    vector_hits: Sequence[VectorHit],
    lexical_hits: Sequence[BM25Hit],
    vector_weight: float = 0.6,
    lexical_weight: float = 0.4,
) -> list[ScoredChunk]:
    """Fuse two ranked lists by weighted, min-max normalised score.

    Documents missing from a list contribute ``0`` for that list rather than
    being dropped, so a strong hit in either retriever can still surface.

    Args:
        vector_hits: Results from the vector store, best first.
        lexical_hits: Results from BM25, best first.
        vector_weight: Weight applied to the normalised vector score.
        lexical_weight: Weight applied to the normalised lexical score.

    Returns:
        Chunks sorted by fused score in ``[0, 1]``.
    """
    total_weight = vector_weight + lexical_weight
    if total_weight <= 0:
        raise ValueError("fusion weights must sum to a positive number")

    chunks = _index_by_chunk(vector_hits, lexical_hits)

    vector_norm = dict(
        zip(
            (h.chunk.chunk_id for h in vector_hits),
            min_max_normalise([h.score for h in vector_hits]),
        )
    )
    lexical_norm = dict(
        zip(
            (h.chunk.chunk_id for h in lexical_hits),
            min_max_normalise([h.score for h in lexical_hits]),
        )
    )
    vector_ranks = {h.chunk.chunk_id: i + 1 for i, h in enumerate(vector_hits)}
    lexical_ranks = {h.chunk.chunk_id: i + 1 for i, h in enumerate(lexical_hits)}
    vector_raw = {h.chunk.chunk_id: h.score for h in vector_hits}
    lexical_raw = {h.chunk.chunk_id: h.score for h in lexical_hits}

    fused = [
        ScoredChunk(
            chunk=chunk,
            score=(
                vector_weight * vector_norm.get(chunk_id, 0.0)
                + lexical_weight * lexical_norm.get(chunk_id, 0.0)
            )
            / total_weight,
            vector_score=vector_raw.get(chunk_id),
            lexical_score=lexical_raw.get(chunk_id),
            vector_rank=vector_ranks.get(chunk_id),
            lexical_rank=lexical_ranks.get(chunk_id),
        )
        for chunk_id, chunk in chunks.items()
    ]
    fused.sort(key=lambda sc: (-sc.score, sc.chunk.chunk_id))
    return fused


def apply_language_boost(
    scored: Iterable[ScoredChunk], query_lang: str, boost: float
) -> list[ScoredChunk]:
    """Scale up chunks matching the query language, then re-sort.

    This *prefers* the query language rather than hard-filtering it, so a
    strongly relevant chunk in the other language can still win -- which is what
    makes cross-lingual fallback possible.

    The boost is multiplicative (``score * (1 + boost)``) rather than additive:
    an additive bonus against a clamped ceiling drives every candidate to the
    same value and collapses the ranking into an arbitrary tie-break.

    Args:
        scored: Fused results.
        query_lang: Detected query language.
        boost: Fractional bonus for matching chunks, e.g. ``0.15`` for +15%.

    Returns:
        A new list sorted by boosted score.
    """
    boosted = []
    for item in scored:
        matches = item.chunk.lang == query_lang
        new_score = item.score * (1.0 + boost) if matches else item.score
        boosted.append(item.model_copy(update={"score": new_score, "language_boosted": matches}))

    boosted.sort(key=lambda sc: (-sc.score, sc.chunk.chunk_id))
    return boosted


def normalise_ranking_scores(scored: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Scale ranking scores so the best candidate is ``1.0``.

    Division by the maximum is used rather than min-max because min-max always
    drives the weakest candidate to exactly ``0.0``, which reads as "irrelevant"
    in logs and API responses even when that chunk scored respectably.

    Purely cosmetic: it makes scores readable and comparable *within* one query.
    It says nothing about absolute relevance -- the top candidate of a
    completely out-of-scope query still normalises to 1.0 -- so confidence is
    judged from :attr:`ScoredChunk.relevance` instead.
    """
    highest = max((item.score for item in scored), default=0.0)
    if highest <= 0.0:
        return list(scored)
    return [item.model_copy(update={"score": item.score / highest}) for item in scored]
