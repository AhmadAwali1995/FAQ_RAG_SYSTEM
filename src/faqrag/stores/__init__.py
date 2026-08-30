"""Vector store backends selectable via ``FAQRAG_VECTOR_STORE``."""

from __future__ import annotations

from ..config import Settings
from .base import VectorHit, VectorStore
from .numpy_store import NumpyVectorStore

__all__ = ["VectorStore", "VectorHit", "NumpyVectorStore", "build_vector_store"]


def build_vector_store(settings: Settings) -> VectorStore:
    """Instantiate the vector store named by ``settings.vector_store``.

    Chroma is imported only when selected, so it stays an optional dependency.
    """
    backend = settings.vector_store
    if backend == "numpy":
        return NumpyVectorStore(settings.index_dir, settings.collection_name)
    if backend == "chroma":
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore(settings.index_dir, settings.collection_name)
    raise ValueError(f"unknown vector store {backend!r}; expected 'numpy' or 'chroma'")
