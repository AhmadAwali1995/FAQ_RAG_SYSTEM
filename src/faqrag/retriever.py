"""Hybrid retrieval: vector search + BM25, fused, language-aware, reranked."""

from __future__ import annotations

import logging

from .bm25 import BM25Index
from .config import Settings
from .embeddings import EmbeddingProvider, build_embedder
from .fusion import (
    apply_language_boost,
    min_max_normalise,
    normalise_ranking_scores,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from .ingest import IngestionError
from .lang import detect_language
from .models import Language, RetrievalResult, ScoredChunk
from .rerank import Reranker, build_reranker
from .stores import VectorStore, build_vector_store

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieves FAQ chunks by combining dense and lexical search.

    The flow for each query is:

    1. Detect the query language.
    2. Run vector search and BM25 over the *whole* corpus (both languages).
    3. Fuse the two rankings with RRF or a weighted sum.
    4. Boost same-language chunks -- a preference, not a filter.
    5. If the best same-language result is weak, keep the cross-lingual
       candidates in play and flag the fallback.
    6. Rerank the survivors, then apply the relevance threshold.

    Ranking and confidence use different signals on purpose. Fusion scores only
    order candidates *within* one query, so they cannot say whether the best of
    a bad set is any good. Absolute relevance -- the reranker's 0-10 judgement,
    or raw embedding cosine when reranking is off -- is what decides confidence.
    """

    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        embedder: EmbeddingProvider,
        bm25: BM25Index,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.reranker = reranker if reranker is not None else build_reranker(settings)

    @classmethod
    def from_settings(cls, settings: Settings) -> "HybridRetriever":
        """Load a persisted index and build a ready-to-query retriever.

        Raises:
            IngestionError: If no index has been built yet.
        """
        store = build_vector_store(settings)
        if not store.load():
            raise IngestionError(
                f"no index found in {settings.index_dir}. Build it with:\n"
                f"  python -m faqrag.index"
            )
        chunks = store.all_chunks()
        logger.info("loaded index: %d chunks from %s", len(chunks), settings.index_dir)
        return cls(
            settings=settings,
            store=store,
            embedder=build_embedder(settings),
            bm25=BM25Index().fit(chunks),
        )

    def _fuse(self, vector_hits, lexical_hits) -> list[ScoredChunk]:
        if self.settings.fusion_method == "rrf":
            return reciprocal_rank_fusion(vector_hits, lexical_hits, k=self.settings.rrf_k)
        return weighted_fusion(
            vector_hits,
            lexical_hits,
            vector_weight=self.settings.vector_weight,
            lexical_weight=self.settings.lexical_weight,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve the most relevant FAQ chunks for ``query``.

        Args:
            query: The user's question, in Arabic or English.
            top_k: Number of chunks to return; defaults to ``settings.top_k``.

        Returns:
            A :class:`RetrievalResult` whose ``confident`` flag is ``False`` when
            no candidate clears ``settings.min_relevance_score``. Callers must
            respect that flag rather than answering from a weak match.
        """
        settings = self.settings
        top_k = top_k or settings.top_k
        query_lang: Language = detect_language(query)

        if not query.strip():
            return RetrievalResult(
                query=query, query_lang=query_lang, threshold=settings.min_relevance_score
            )

        # Search the full corpus in both languages; preference is applied during
        # fusion so a strong cross-lingual match can still surface.
        vector_hits = self.store.search(
            self.embedder.embed_query(query), k=settings.candidate_k, lang=None
        )
        lexical_hits = self.bm25.search(query, k=settings.candidate_k, lang=None)
        logger.debug(
            "vector hits=%d lexical hits=%d for %r", len(vector_hits), len(lexical_hits), query
        )

        fused = self._fuse(vector_hits, lexical_hits)
        if not fused:
            return RetrievalResult(
                query=query, query_lang=query_lang, threshold=settings.min_relevance_score
            )

        boosted = apply_language_boost(fused, query_lang, settings.language_boost)

        # Cross-lingual fallback: when nothing in the query's own language looks
        # strong, note it so the trace explains why an other-language chunk won.
        # This must be judged on an absolute signal -- raw embedding cosine --
        # because fused scores are only meaningful relative to the same query.
        same_lang_best = max(
            (
                item.vector_score or 0.0
                for item in boosted
                if item.chunk.lang == query_lang
            ),
            default=0.0,
        )
        cross_lingual = same_lang_best < settings.cross_lingual_fallback_score
        if cross_lingual:
            logger.info(
                "same-language best score %.3f below fallback threshold %.3f; "
                "allowing cross-lingual results",
                same_lang_best,
                settings.cross_lingual_fallback_score,
            )

        candidates = boosted[: settings.rerank_candidates]
        reranked = False
        if settings.rerank_enabled and len(candidates) > 1:
            before = [c.chunk.chunk_id for c in candidates]
            candidates = self.reranker.rerank(query, candidates, top_k)
            reranked = any(c.rerank_score is not None for c in candidates)
            if reranked and [c.chunk.chunk_id for c in candidates] != before[: len(candidates)]:
                logger.debug("reranker reordered candidates for %r", query)

        final = normalise_ranking_scores(self._attach_relevance(candidates[:top_k]))
        best_relevance = max((c.relevance or 0.0 for c in final), default=0.0)

        return RetrievalResult(
            query=query,
            query_lang=query_lang,
            chunks=final,
            confident=best_relevance >= settings.min_relevance_score,
            threshold=settings.min_relevance_score,
            cross_lingual_fallback=cross_lingual,
            reranked=reranked,
        )

    #: Weight on the reranker's judgement when blending it with cosine.
    RERANK_RELEVANCE_WEIGHT = 0.6

    @classmethod
    def _attach_relevance(cls, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        """Annotate each candidate with an absolute relevance estimate in [0, 1].

        Two independent signals are blended rather than ranked by preference:

        * The reranker's 0-10 score, rescaled. It is the only signal that judges
          query and chunk together, so it carries the larger weight.
        * Raw embedding cosine similarity. The absolute cosine scale is a
          property of the embedding model, not of relevance: under bge-m3 this
          corpus separated at 0.6-0.8 for genuine matches vs ~0.3 for
          out-of-scope, while the OpenAI text-embedding-3 models run lower
          across the board. After changing the embedding model, re-run
          ``python -m faqrag.eval`` and re-tune ``FAQRAG_MIN_RELEVANCE_SCORE``
          -- left alone, a lower cosine scale shows up as spurious "no
          confident match" refusals.

        Blending matters because a strict reranker will score a correct-but-terse
        FAQ entry mid-range, which alone would fall below the threshold and
        wrongly decline. Requiring *both* signals to be weak before declining
        makes the confidence decision far less sensitive to rubric wording, and
        an out-of-scope query still fails both.

        Chunks that only BM25 found have no cosine, so their normalised lexical
        score stands in, damped to 0.6 -- a within-query normalisation overstates
        how good a lexical-only match is in absolute terms.
        """
        lexical_norm = dict(
            zip(
                (c.chunk.chunk_id for c in candidates),
                min_max_normalise([c.lexical_score or 0.0 for c in candidates]),
            )
        )

        annotated = []
        for item in candidates:
            if item.vector_score is not None:
                dense = max(0.0, min(1.0, item.vector_score))
            else:
                dense = 0.6 * lexical_norm.get(item.chunk.chunk_id, 0.0)

            if item.rerank_score is None:
                relevance = dense
            else:
                weight = cls.RERANK_RELEVANCE_WEIGHT
                relevance = weight * (item.rerank_score / 10.0) + (1.0 - weight) * dense

            annotated.append(item.model_copy(update={"relevance": relevance}))
        return annotated
