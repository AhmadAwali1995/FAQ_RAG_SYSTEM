"""A Chroma-backed :class:`VectorStore`.

Selected with ``FAQRAG_VECTOR_STORE=chroma``.  Chroma is imported lazily so the
package stays importable when it is not installed.  Embeddings are always
supplied by our own :mod:`faqrag.embeddings` provider -- Chroma's built-in
embedding functions are bypassed so the same multilingual model backs every
store backend.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..models import Chunk
from .base import VectorHit, VectorStore

logger = logging.getLogger(__name__)

# Chroma metadata values must be scalars, so keyword lists are joined on this
# separator when stored and split back out on read.
_KEYWORD_SEP = "␟"


class ChromaVectorStore(VectorStore):
    """Persistent Chroma collection using externally computed embeddings."""

    def __init__(self, index_dir: Path, collection_name: str = "faq") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "FAQRAG_VECTOR_STORE=chroma requires chromadb. Install it with:\n"
                "  pip install chromadb"
            ) from exc

        self._dir = Path(index_dir) / "chroma"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._name = collection_name
        self._client = chromadb.PersistentClient(path=str(self._dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # Cosine keeps scores comparable with the NumPy backend.
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _to_metadata(chunk: Chunk) -> dict[str, str]:
        return {
            "faq_id": chunk.faq_id,
            "category": chunk.category,
            "lang": chunk.lang,
            "question": chunk.question,
            "answer": chunk.answer,
            "keywords": _KEYWORD_SEP.join(chunk.keywords),
        }

    @staticmethod
    def _from_metadata(chunk_id: str, meta: dict[str, str]) -> Chunk:
        raw_keywords = meta.get("keywords", "")
        return Chunk(
            chunk_id=chunk_id,
            faq_id=meta["faq_id"],
            category=meta["category"],
            lang=meta["lang"],  # type: ignore[arg-type]
            question=meta["question"],
            answer=meta["answer"],
            keywords=raw_keywords.split(_KEYWORD_SEP) if raw_keywords else [],
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace the collection's contents with ``chunks`` and their vectors."""
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            raise ValueError("refusing to build an empty index")

        # Recreate the collection so a rebuild never leaves stale chunks behind.
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.embedding_text for c in chunks],
            metadatas=[self._to_metadata(c) for c in chunks],
        )

    def search(self, vector: list[float], k: int, lang: str | None = None) -> list[VectorHit]:
        """Return the ``k`` most similar chunks, optionally filtered by language."""
        total = self.count()
        if total == 0:
            raise RuntimeError("vector store is empty; build the index first")

        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(k, total),
            where={"lang": lang} if lang else None,
            include=["metadatas", "distances"],
        )

        ids = result["ids"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        # Chroma reports cosine *distance* (1 - similarity); convert back.
        return [
            VectorHit(chunk=self._from_metadata(cid, meta), score=1.0 - float(dist))
            for cid, meta, dist in zip(ids, metadatas, distances)
        ]

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk."""
        result = self._collection.get(include=["metadatas"])
        return [
            self._from_metadata(cid, meta)
            for cid, meta in zip(result["ids"], result["metadatas"])
        ]

    def count(self) -> int:
        """Number of stored chunks."""
        return self._collection.count()

    def persist(self) -> None:
        """No-op: ``PersistentClient`` writes on every mutation."""

    def load(self) -> bool:
        """Return ``True`` when the persisted collection holds data."""
        return self.count() > 0
