"""Core data structures shared across ingestion, retrieval, and generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["ar", "en"]

# Sentinel used when the retriever finds nothing above the relevance threshold.
NO_MATCH = "NO_CONFIDENT_MATCH"


class FaqRecord(BaseModel):
    """One raw FAQ entry as it appears in the source JSON."""

    faq_id: str
    category: str
    lang: Language
    question: str
    answer: str
    keywords: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """One retrievable unit: a single ``(faq_id, lang)`` FAQ entry.

    Arabic and English versions of the same FAQ are deliberately kept as
    separate chunks -- mixing scripts inside one chunk degrades multilingual
    embedding quality and pollutes BM25 term statistics.
    """

    chunk_id: str
    faq_id: str
    category: str
    lang: Language
    question: str
    answer: str
    keywords: list[str] = Field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model.

        Question *and* answer are embedded together: FAQ answers carry most of
        the retrievable signal, and real user phrasings often match answer
        content more closely than the canonical question wording.  Keywords are
        appended because they encode product names and synonyms that neither
        field always spells out.
        """
        parts = [self.question, self.answer]
        if self.keywords:
            parts.append(" | ".join(self.keywords))
        return "\n".join(parts)

    @property
    def lexical_text(self) -> str:
        """Text indexed by BM25 (question, answer, and keywords)."""
        return "\n".join([self.question, self.answer, " ".join(self.keywords)])


class ScoredChunk(BaseModel):
    """A chunk with the scores that led to its retrieval.

    Two scores serve distinct purposes and must not be conflated:

    * ``score`` is a *ranking* score. It orders candidates against each other
      and is min-max normalised across the candidate set, so the best candidate
      of any query -- relevant or not -- scores near 1.0.
    * ``relevance`` is an *absolute* estimate of how well this chunk answers the
      query, comparable across queries. Only this is compared against the
      confidence threshold.

    Every underlying signal is retained rather than collapsed into one number so
    that bad answers can be debugged after the fact from the query log.
    """

    chunk: Chunk
    score: float
    relevance: float | None = None
    vector_score: float | None = None
    lexical_score: float | None = None
    vector_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None
    language_boosted: bool = False


class RetrievalResult(BaseModel):
    """Outcome of a hybrid retrieval pass."""

    query: str
    query_lang: Language
    chunks: list[ScoredChunk] = Field(default_factory=list)
    confident: bool = False
    threshold: float = 0.0
    cross_lingual_fallback: bool = False
    reranked: bool = False


class SourceCitation(BaseModel):
    """A single FAQ the answer drew on, for rendering citations in a UI."""

    faq_id: str
    category: str
    lang: Language
    question: str
    answer: str
    score: float


class QueryResponse(BaseModel):
    """Structured response returned by the CLI and the ``/query`` endpoint.

    ``sources`` and ``retrieved`` are deliberately distinct. ``sources`` holds
    only the FAQs the model actually cited, for rendering citations under an
    answer. ``retrieved`` holds everything retrieval surfaced, whether cited or
    not -- which is what retrieval quality must be measured against, and what
    makes a wrong citation visible instead of invisible.
    """

    answer: str
    cited_faq_ids: list[str] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)
    retrieved: list[SourceCitation] = Field(default_factory=list)
    confidence: float
    confident: bool
    language: Language
    query: str
    reranked: bool = False
    cross_lingual_fallback: bool = False
    latency_ms: float | None = None


class StreamEvent(BaseModel):
    """One streaming event emitted by the RAG API."""

    event: Literal["metadata", "delta", "final", "error"]
    data: dict[str, object] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """One retrieval candidate with every score that produced its rank.

    Returned by ``POST /retrieve`` so a bad answer can be diagnosed as a
    retrieval fault or a generation fault without re-running the LLM.
    """

    faq_id: str
    chunk_id: str
    category: str
    lang: Language
    question: str
    answer: str
    rank_score: float = Field(
        description="Fused rank within THIS query, scaled so the best is 1.0. "
        "Not comparable across queries."
    )
    relevance: float = Field(
        description="Absolute relevance in [0,1], comparable across queries. "
        "This is what the confidence threshold is applied to."
    )
    vector_score: float | None = Field(
        default=None, description="Raw embedding cosine similarity, if the vector search found it."
    )
    lexical_score: float | None = Field(
        default=None, description="Raw BM25 score, if the keyword search found it."
    )
    rerank_score: float | None = Field(
        default=None, description="Reranker's 0-10 judgement, when reranking ran."
    )


class RetrieveResponse(BaseModel):
    """Full retrieval trace for one query, with no answer generated."""

    query: str
    language: Language = Field(description="Detected query language.")
    confident: bool = Field(description="Whether any candidate cleared the threshold.")
    threshold: float
    reranked: bool
    cross_lingual_fallback: bool = Field(
        description="True when the query's own language had no strong match, "
        "so other-language chunks were allowed to compete."
    )
    chunks: list[RetrievedChunk] = Field(default_factory=list)
