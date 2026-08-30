"""Tests for rank fusion, language boosting, and score normalisation.

These guard the arithmetic that decides which chunk reaches the generator. Bugs
here are silent: the pipeline still returns an answer, just a worse-sourced one.
"""

from __future__ import annotations

import pytest

from faqrag.bm25 import BM25Hit
from faqrag.fusion import (
    apply_language_boost,
    min_max_normalise,
    normalise_ranking_scores,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from faqrag.models import Chunk, ScoredChunk
from faqrag.stores.base import VectorHit


def make_chunk(faq_id: str, lang: str = "en") -> Chunk:
    """Build a minimal chunk for fusion tests."""
    return Chunk(
        chunk_id=f"{faq_id}::{lang}",
        faq_id=faq_id,
        category="Test",
        lang=lang,  # type: ignore[arg-type]
        question=f"Question {faq_id}",
        answer=f"Answer {faq_id}",
        keywords=[],
    )


def vector_hits(*specs: tuple[str, float], lang: str = "en") -> list[VectorHit]:
    return [VectorHit(chunk=make_chunk(fid, lang), score=score) for fid, score in specs]


def lexical_hits(*specs: tuple[str, float], lang: str = "en") -> list[BM25Hit]:
    return [BM25Hit(chunk=make_chunk(fid, lang), score=score) for fid, score in specs]


class TestMinMaxNormalise:
    """Score scaling helper."""

    def test_scales_to_unit_range(self) -> None:
        assert min_max_normalise([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]

    def test_identical_values_map_to_one(self) -> None:
        """All-equal scores mean all equally relevant, not all irrelevant."""
        assert min_max_normalise([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]

    def test_single_value(self) -> None:
        assert min_max_normalise([7.0]) == [1.0]

    def test_empty(self) -> None:
        assert min_max_normalise([]) == []


class TestReciprocalRankFusion:
    """RRF combination of two ranked lists."""

    def test_agreement_beats_single_list_strength(self) -> None:
        """A doc ranked well by both retrievers must beat a one-list winner."""
        fused = reciprocal_rank_fusion(
            vector_hits(("A", 0.9), ("B", 0.8)),
            lexical_hits(("B", 9.0), ("C", 1.0)),
        )
        assert fused[0].chunk.faq_id == "B"

    def test_union_of_both_lists_is_kept(self) -> None:
        """Documents found by only one retriever must not be dropped."""
        fused = reciprocal_rank_fusion(
            vector_hits(("A", 0.9)), lexical_hits(("B", 5.0))
        )
        assert {item.chunk.faq_id for item in fused} == {"A", "B"}

    def test_preserves_individual_scores_and_ranks(self) -> None:
        """Per-retriever detail must survive for debugging."""
        fused = reciprocal_rank_fusion(
            vector_hits(("A", 0.91), ("B", 0.7)), lexical_hits(("B", 4.2))
        )
        by_id = {item.chunk.faq_id: item for item in fused}

        assert by_id["A"].vector_score == pytest.approx(0.91)
        assert by_id["A"].vector_rank == 1
        assert by_id["A"].lexical_score is None
        assert by_id["A"].lexical_rank is None
        assert by_id["B"].lexical_score == pytest.approx(4.2)
        assert by_id["B"].lexical_rank == 1
        assert by_id["B"].vector_rank == 2

    def test_returns_raw_scores_not_scaled_to_the_theoretical_maximum(self) -> None:
        """Regression: dividing by the theoretical maximum (2/(k+1)) mapped every
        candidate into [0.75, 1.0]. A later additive language boost then clamped
        them all to 1.0, reducing the whole ranking to an arbitrary tie-break.

        Raw RRF sums are small by construction; that small absolute magnitude is
        what keeps a downstream boost from saturating.
        """
        k = 60
        fused = reciprocal_rank_fusion(
            vector_hits(("A", 0.9), ("B", 0.8)), lexical_hits(("A", 9.0), ("B", 8.0)), k=k
        )
        # Top document is rank 1 in both lists: 2/(k+1), not a normalised 1.0.
        assert fused[0].score == pytest.approx(2.0 / (k + 1))
        assert fused[0].score < 0.1

    def test_agreeing_lists_produce_a_strict_ordering(self) -> None:
        """With both retrievers agreeing, fused order must match and stay strict."""
        specs = [(f"F{i}", 1.0 - i / 20) for i in range(20)]
        fused = reciprocal_rank_fusion(
            vector_hits(*specs),
            lexical_hits(*[(fid, 20.0 - i) for i, (fid, _) in enumerate(specs)]),
        )
        assert [item.chunk.faq_id for item in fused] == [fid for fid, _ in specs]
        scores = [item.score for item in fused]
        assert scores == sorted(scores, reverse=True)
        assert len(set(scores)) == len(scores), "ranks must not collapse into ties"

    def test_is_order_dependent_not_score_dependent(self) -> None:
        """RRF must use ranks only, so score magnitude cannot sway it."""
        a = reciprocal_rank_fusion(
            vector_hits(("A", 0.99), ("B", 0.98)), lexical_hits(("A", 100.0))
        )
        b = reciprocal_rank_fusion(
            vector_hits(("A", 0.11), ("B", 0.10)), lexical_hits(("A", 0.5))
        )
        assert [i.chunk.faq_id for i in a] == [i.chunk.faq_id for i in b]
        assert [i.score for i in a] == [i.score for i in b]

    def test_damping_constant_changes_spread(self) -> None:
        small_k = reciprocal_rank_fusion(
            vector_hits(("A", 0.9), ("B", 0.8)), lexical_hits(("A", 1.0)), k=1
        )
        large_k = reciprocal_rank_fusion(
            vector_hits(("A", 0.9), ("B", 0.8)), lexical_hits(("A", 1.0)), k=1000
        )
        # A larger k flattens the gap between the top two candidates.
        assert (small_k[0].score - small_k[1].score) > (large_k[0].score - large_k[1].score)

    def test_empty_inputs(self) -> None:
        assert reciprocal_rank_fusion([], []) == []


class TestWeightedFusion:
    """Weighted min-max fusion."""

    def test_weights_shift_the_winner(self) -> None:
        vectors = vector_hits(("A", 0.9), ("B", 0.1))
        lexicals = lexical_hits(("B", 9.0), ("A", 1.0))

        vector_led = weighted_fusion(vectors, lexicals, vector_weight=0.9, lexical_weight=0.1)
        lexical_led = weighted_fusion(vectors, lexicals, vector_weight=0.1, lexical_weight=0.9)

        assert vector_led[0].chunk.faq_id == "A"
        assert lexical_led[0].chunk.faq_id == "B"

    def test_missing_from_one_list_scores_zero_there_not_dropped(self) -> None:
        fused = weighted_fusion(vector_hits(("A", 0.9)), lexical_hits(("B", 5.0)))
        assert {item.chunk.faq_id for item in fused} == {"A", "B"}

    def test_scores_stay_in_unit_range(self) -> None:
        fused = weighted_fusion(
            vector_hits(("A", 0.9), ("B", 0.3)), lexical_hits(("A", 8.0), ("B", 2.0))
        )
        assert all(0.0 <= item.score <= 1.0 for item in fused)

    def test_zero_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            weighted_fusion(vector_hits(("A", 0.9)), [], vector_weight=0.0, lexical_weight=0.0)


class TestLanguageBoost:
    """Language preference applied after fusion."""

    def test_matching_language_is_promoted(self) -> None:
        scored = [
            ScoredChunk(chunk=make_chunk("A", "en"), score=0.50),
            ScoredChunk(chunk=make_chunk("B", "ar"), score=0.48),
        ]
        boosted = apply_language_boost(scored, "ar", boost=0.15)
        assert boosted[0].chunk.faq_id == "B"
        assert boosted[0].language_boosted is True
        assert boosted[1].language_boosted is False

    def test_is_a_preference_not_a_filter(self) -> None:
        """A clearly better cross-language chunk must still be able to win."""
        scored = [
            ScoredChunk(chunk=make_chunk("A", "en"), score=0.90),
            ScoredChunk(chunk=make_chunk("B", "ar"), score=0.40),
        ]
        boosted = apply_language_boost(scored, "ar", boost=0.15)
        assert boosted[0].chunk.faq_id == "A"
        assert len(boosted) == 2, "other-language chunks must never be removed"

    def test_does_not_saturate_and_destroy_ordering(self) -> None:
        """Regression: an additive boost against a 1.0 ceiling drove every
        candidate to the same score, reducing the ranking to a tie-break."""
        scored = [
            ScoredChunk(chunk=make_chunk(f"F{i}", "ar"), score=0.90 + i * 0.01)
            for i in range(5)
        ]
        boosted = apply_language_boost(scored, "ar", boost=0.15)
        assert len({item.score for item in boosted}) == 5, "boost collapsed distinct scores"
        assert [i.chunk.faq_id for i in boosted] == ["F4", "F3", "F2", "F1", "F0"]

    def test_zero_boost_preserves_order(self) -> None:
        scored = [
            ScoredChunk(chunk=make_chunk("A", "en"), score=0.5),
            ScoredChunk(chunk=make_chunk("B", "ar"), score=0.4),
        ]
        boosted = apply_language_boost(scored, "ar", boost=0.0)
        assert [i.chunk.faq_id for i in boosted] == ["A", "B"]


class TestNormaliseRankingScores:
    """Cosmetic normalisation of the final candidate list."""

    def test_best_becomes_one_and_order_is_preserved(self) -> None:
        scored = [
            ScoredChunk(chunk=make_chunk("A"), score=0.04),
            ScoredChunk(chunk=make_chunk("B"), score=0.02),
            ScoredChunk(chunk=make_chunk("C"), score=0.01),
        ]
        result = normalise_ranking_scores(scored)
        assert result[0].score == pytest.approx(1.0)
        assert [i.chunk.faq_id for i in result] == ["A", "B", "C"]

    def test_weakest_candidate_is_not_forced_to_zero(self) -> None:
        """min-max would report a respectable last-place chunk as 0.0."""
        scored = [
            ScoredChunk(chunk=make_chunk("A"), score=0.04),
            ScoredChunk(chunk=make_chunk("B"), score=0.038),
        ]
        result = normalise_ranking_scores(scored)
        assert result[-1].score > 0.9

    def test_all_zero_scores_are_left_alone(self) -> None:
        scored = [ScoredChunk(chunk=make_chunk("A"), score=0.0)]
        assert normalise_ranking_scores(scored)[0].score == 0.0

    def test_empty(self) -> None:
        assert normalise_ranking_scores([]) == []
