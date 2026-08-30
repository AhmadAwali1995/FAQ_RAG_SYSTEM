"""A small Okapi BM25 implementation with Arabic-aware tokenisation.

Written in-package rather than pulled from a library so the tokeniser -- which
does the Arabic normalisation and definite-article stripping in
:mod:`faqrag.lang` -- is the same one used everywhere else.  BM25 matters here
because pure vector search misses exact product and brand terms ("Tabby",
"Mwfaq Academy") that FAQ users type verbatim.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .lang import tokenize
from .models import Chunk

# Okapi BM25 defaults: k1 controls term-frequency saturation, b the strength of
# document-length normalisation.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass
class BM25Hit:
    """A lexical search result."""

    chunk: Chunk
    score: float


@dataclass
class BM25Index:
    """An in-memory BM25 index over chunk lexical text.

    Attributes:
        k1: Term-frequency saturation parameter.
        b: Document-length normalisation parameter.
    """

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    _chunks: list[Chunk] = field(default_factory=list, repr=False)
    _term_freqs: list[Counter[str]] = field(default_factory=list, repr=False)
    _doc_lengths: list[int] = field(default_factory=list, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _avg_length: float = 0.0

    def fit(self, chunks: list[Chunk]) -> "BM25Index":
        """Build the index over ``chunks``.

        Each chunk's question, answer, and keywords are concatenated and
        tokenised with the chunk's own language, so Arabic normalisation is
        applied only to Arabic documents.
        """
        self._chunks = list(chunks)
        self._term_freqs = []
        self._doc_lengths = []

        doc_freq: Counter[str] = Counter()
        for chunk in self._chunks:
            tokens = tokenize(chunk.lexical_text, chunk.lang)
            counts = Counter(tokens)
            self._term_freqs.append(counts)
            self._doc_lengths.append(len(tokens))
            doc_freq.update(counts.keys())

        n_docs = len(self._chunks)
        self._avg_length = (sum(self._doc_lengths) / n_docs) if n_docs else 0.0

        # Probabilistic IDF with the +1 smoothing that keeps very common terms
        # at a small positive weight instead of going negative.
        self._idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }
        return self

    def _score_document(self, index: int, query_terms: list[str]) -> float:
        counts = self._term_freqs[index]
        length = self._doc_lengths[index]
        norm = self.k1 * (1.0 - self.b + self.b * length / (self._avg_length or 1.0))

        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if freq:
                score += self._idf.get(term, 0.0) * (freq * (self.k1 + 1.0)) / (freq + norm)
        return score

    def search(self, query: str, k: int, lang: str | None = None) -> list[BM25Hit]:
        """Return the ``k`` highest-scoring chunks for ``query``.

        Args:
            query: Raw user query text.
            k: Maximum number of hits.
            lang: Restrict to chunks of this language when given. The query is
                also tokenised as this language, so an Arabic query is
                normalised even while probing the English corpus.

        Returns:
            Hits sorted by descending score, excluding zero-score documents.
        """
        if not self._chunks:
            raise RuntimeError("BM25 index is empty; call fit() first")

        query_terms = tokenize(query, lang)
        if not query_terms:
            return []

        scored: list[BM25Hit] = []
        for i, chunk in enumerate(self._chunks):
            if lang is not None and chunk.lang != lang:
                continue
            score = self._score_document(i, query_terms)
            if score > 0.0:
                scored.append(BM25Hit(chunk=chunk, score=score))

        scored.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return scored[:k]

    def __len__(self) -> int:
        return len(self._chunks)
