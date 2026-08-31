"""Embedding providers behind a common interface.

Adding a provider means implementing :class:`EmbeddingProvider` and registering
it in :func:`build_embedder` -- no changes anywhere else in the pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import httpx

from .config import Settings
from .http_utils import describe_http_error

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend fails or returns malformed data."""


class EmbeddingProvider(ABC):
    """Turns text into dense vectors."""

    #: Human-readable identifier recorded in the index manifest.
    name: str = "abstract"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus documents."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query.

        Kept separate from :meth:`embed_documents` because several models
        (E5, BGE) expect asymmetric query/document prefixes.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""


class OllamaEmbeddings(EmbeddingProvider):
    """Embeddings from a local Ollama server via ``/api/embed``.

    Defaults to ``bge-m3``, a multilingual model with solid Arabic coverage.
    bge-m3 is symmetric and needs no query/document prefix, so queries and
    documents are embedded identically.
    """

    def __init__(self, settings: Settings) -> None:
        self._model = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._url = settings.ollama_base_url.rstrip("/") + "/api/embed"
        self._timeout = settings.ollama_timeout
        self._dimension: int | None = None
        self.name = f"ollama:{self._model}"

    def _post(self, inputs: list[str]) -> list[list[float]]:
        try:
            response = httpx.post(
                self._url,
                json={"model": self._model, "input": inputs},
                timeout=self._timeout,
            )
            response.raise_for_status()
            vectors = response.json().get("embeddings")
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"Ollama embedding request failed ({exc}). Is the server running at "
                f"{self._url}? Try: ollama pull {self._model}"
            ) from exc

        if not vectors or len(vectors) != len(inputs):
            raise EmbeddingError(
                f"expected {len(inputs)} embeddings from {self._model}, got {len(vectors or [])}"
            )
        self._dimension = len(vectors[0])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in batches, preserving input order."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            logger.debug("embedding batch %d-%d of %d", start, start + len(batch), len(texts))
            vectors.extend(self._post(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._post([text])[0]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension


class OpenAIEmbeddings(EmbeddingProvider):
    """Embeddings from an OpenAI-compatible ``/embeddings`` endpoint.

    Defaults to ``text-embedding-3-large`` (3072-dim), the strongest Arabic
    coverage in the OpenAI family. These models are symmetric, so queries and
    documents are embedded identically. Requires ``FAQRAG_OPENAI_API_KEY``.

    The vector dimension differs from every other model, so switching here
    invalidates an existing index -- rebuild it with ``python -m faqrag.index``.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise EmbeddingError(
                "embedding_provider='openai' requires FAQRAG_OPENAI_API_KEY to be set"
            )
        self._model = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._url = settings.openai_base_url.rstrip("/") + "/embeddings"
        self._headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        self._dimension: int | None = None
        self.name = f"openai:{self._model}"

    def _post(self, inputs: list[str]) -> list[list[float]]:
        try:
            response = httpx.post(
                self._url,
                json={"model": self._model, "input": inputs},
                headers=self._headers,
                timeout=60.0,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"OpenAI embedding request to {self._model} failed: "
                f"{describe_http_error(exc)}"
            ) from exc

        # The API may return items out of order; sort by the echoed index.
        items = sorted(payload.get("data") or [], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in items]
        if len(vectors) != len(inputs):
            raise EmbeddingError(
                f"expected {len(inputs)} embeddings from {self._model}, got {len(vectors)}"
            )
        self._dimension = len(vectors[0])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in batches, preserving input order."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            logger.debug("embedding batch %d-%d of %d", start, start + len(batch), len(texts))
            vectors.extend(self._post(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._post([text])[0]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension


_PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "ollama": OllamaEmbeddings,
    "openai": OpenAIEmbeddings,
}


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """Instantiate the embedding provider named by ``settings.embedding_provider``."""
    try:
        provider_cls = _PROVIDERS[settings.embedding_provider]
    except KeyError:
        raise EmbeddingError(
            f"unknown embedding provider {settings.embedding_provider!r}; "
            f"expected one of {sorted(_PROVIDERS)}"
        ) from None
    return provider_cls(settings)
