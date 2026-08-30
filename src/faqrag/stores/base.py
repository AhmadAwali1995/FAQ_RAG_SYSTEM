"""The :class:`VectorStore` interface every backend implements.

Migrating to Qdrant, Pinecone, or pgvector means adding one subclass and a
registry entry -- callers only ever see this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..models import Chunk


@dataclass(frozen=True)
class VectorHit:
    """A single nearest-neighbour result.

    Attributes:
        chunk: The stored chunk, with all of its original metadata.
        score: Cosine similarity in ``[-1, 1]``; higher is more similar.
    """

    chunk: Chunk
    score: float


class VectorStore(ABC):
    """Persistent store mapping chunks to embedding vectors."""

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Add chunks and their vectors, replacing any existing content."""

    @abstractmethod
    def search(
        self,
        vector: list[float],
        k: int,
        lang: str | None = None,
    ) -> list[VectorHit]:
        """Return the ``k`` nearest chunks to ``vector``.

        Args:
            vector: The query embedding.
            k: Maximum number of hits to return.
            lang: If given, restrict results to chunks in this language.
        """

    @abstractmethod
    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk, for building the lexical index."""

    @abstractmethod
    def persist(self) -> None:
        """Flush the store to disk."""

    @abstractmethod
    def load(self) -> bool:
        """Load a previously persisted store. Returns ``False`` if absent."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored chunks."""
