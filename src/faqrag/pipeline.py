"""The end-to-end RAG pipeline shared by the CLI, the API, and the eval harness."""

from __future__ import annotations

import logging
import time

from .config import Settings, get_settings
from .generate import AnswerGenerator, to_citations
from .llm import build_llm
from .logging_utils import build_trace, log_retrieval_summary, write_trace
from .models import QueryResponse, RetrievalResult
from .retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RagPipeline:
    """Retrieval plus grounded generation, with per-query tracing.

    Construct once and reuse: building it loads the index and the embedding
    client, which should not happen per request.
    """

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RagPipeline":
        """Build a pipeline from configuration, loading the persisted index."""
        settings = settings or get_settings()
        retriever = HybridRetriever.from_settings(settings)
        generator = AnswerGenerator(settings, build_llm(settings))
        return cls(settings, retriever, generator)

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        """Run retrieval only, without generating an answer."""
        return self.retriever.retrieve(question, top_k)

    def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        """Answer ``question`` from the FAQ corpus.

        Args:
            question: The user's question in Arabic or English.
            top_k: Chunks to retrieve; defaults to ``settings.top_k``.

        Returns:
            A :class:`QueryResponse` with the answer, cited FAQ ids, the source
            chunks, a confidence score, and the detected language. When nothing
            clears the relevance threshold, ``confident`` is ``False`` and the
            answer says so rather than guessing.
        """
        started = time.perf_counter()
        result = self.retriever.retrieve(question, top_k)
        log_retrieval_summary(logger, result)

        answer_text, cited_ids = self.generator.generate(result)
        latency_ms = (time.perf_counter() - started) * 1000.0

        confidence = max((c.relevance or 0.0 for c in result.chunks), default=0.0)
        response = QueryResponse(
            answer=answer_text,
            cited_faq_ids=cited_ids,
            sources=to_citations(result.chunks, cited_ids) if result.confident else [],
            # Always reported, including on a refusal: seeing what retrieval
            # surfaced is exactly what you need to debug a wrong refusal.
            retrieved=to_citations(result.chunks, []),
            confidence=round(confidence, 4),
            confident=result.confident,
            language=result.query_lang,
            query=question,
            reranked=result.reranked,
            cross_lingual_fallback=result.cross_lingual_fallback,
            latency_ms=round(latency_ms, 1),
        )

        if self.settings.log_retrieval_traces:
            write_trace(
                self.settings.log_dir,
                build_trace(
                    result,
                    extra={
                        "answer": answer_text,
                        "cited_faq_ids": cited_ids,
                        "confidence": response.confidence,
                        "latency_ms": response.latency_ms,
                    },
                ),
            )
        return response
