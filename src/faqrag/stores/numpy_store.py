"""A file-backed vector store built on NumPy.

At FAQ scale (tens to low thousands of chunks) an exhaustive cosine scan over a
single in-memory matrix is faster than any ANN index and adds no dependencies.
The store persists as one ``.npz`` (vectors) plus one ``.json`` (chunks), so the
index is inspectable and diffable.  Swap to :mod:`.chroma_store` -- or a hosted
backend -- via ``FAQRAG_VECTOR_STORE`` when the corpus outgrows this.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..models import Chunk
from .base import VectorHit, VectorStore

logger = logging.getLogger(__name__)


class NumpyVectorStore(VectorStore):
    """Exhaustive cosine-similarity search over an in-memory matrix."""

    def __init__(self, index_dir: Path, collection_name: str = "faq") -> None:
        self._dir = Path(index_dir)
        self._name = collection_name
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    @property
    def _vectors_path(self) -> Path:
        return self._dir / f"{self._name}.vectors.npz"

    @property
    def _chunks_path(self) -> Path:
        return self._dir / f"{self._name}.chunks.json"

    @staticmethod
    def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
        """L2-normalise each row so cosine similarity reduces to a dot product."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Guard against zero vectors, which would otherwise produce NaNs.
        return matrix / np.maximum(norms, 1e-12)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace the store's contents with ``chunks`` and their ``vectors``."""
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            raise ValueError("refusing to build an empty index")

        self._chunks = list(chunks)
        self._matrix = self._normalise_rows(np.asarray(vectors, dtype=np.float32))

    def search(self, vector: list[float], k: int, lang: str | None = None) -> list[VectorHit]:
        """Return the ``k`` most similar chunks, optionally filtered by language."""
        if self._matrix is None or not self._chunks:
            raise RuntimeError("vector store is empty; build the index first")

        query = np.asarray(vector, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = self._matrix @ query

        candidate_idx = np.arange(len(self._chunks))
        if lang is not None:
            mask = np.array([c.lang == lang for c in self._chunks], dtype=bool)
            if not mask.any():
                return []
            candidate_idx = candidate_idx[mask]

        k = min(k, len(candidate_idx))
        if k <= 0:
            return []

        candidate_scores = scores[candidate_idx]
        # argpartition finds the top-k without fully sorting the array.
        top_local = np.argpartition(-candidate_scores, k - 1)[:k]
        top_local = top_local[np.argsort(-candidate_scores[top_local])]

        return [
            VectorHit(chunk=self._chunks[int(candidate_idx[i])], score=float(candidate_scores[i]))
            for i in top_local
        ]

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk."""
        return list(self._chunks)

    def count(self) -> int:
        """Number of stored chunks."""
        return len(self._chunks)

    def persist(self) -> None:
        """Write vectors and chunk metadata to ``index_dir``."""
        if self._matrix is None:
            raise RuntimeError("nothing to persist; call add() first")
        self._dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._vectors_path, vectors=self._matrix)
        self._chunks_path.write_text(
            json.dumps([c.model_dump() for c in self._chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("persisted %d vectors to %s", len(self._chunks), self._dir)

    def load(self) -> bool:
        """Load a persisted index. Returns ``False`` when none exists."""
        if not (self._vectors_path.exists() and self._chunks_path.exists()):
            return False
        self._matrix = np.load(self._vectors_path)["vectors"]
        self._chunks = [
            Chunk.model_validate(raw)
            for raw in json.loads(self._chunks_path.read_text(encoding="utf-8"))
        ]
        if len(self._chunks) != self._matrix.shape[0]:
            raise RuntimeError(
                f"index is corrupt: {len(self._chunks)} chunks vs "
                f"{self._matrix.shape[0]} vectors. Re-run: python -m faqrag.index"
            )
        return True
