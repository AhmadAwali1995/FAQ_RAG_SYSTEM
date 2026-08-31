"""Tests for the vector store backends behind the VectorStore interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from faqrag.config import Settings
from faqrag.models import Chunk
from faqrag.stores import NumpyVectorStore, build_vector_store


def chunk(faq_id: str, lang: str = "en") -> Chunk:
    return Chunk(
        chunk_id=f"{faq_id}::{lang}",
        faq_id=faq_id,
        category="Test",
        lang=lang,  # type: ignore[arg-type]
        question=f"Question {faq_id}?",
        answer=f"Answer {faq_id}.",
        keywords=["kw", faq_id],
    )


CHUNKS = [chunk("001", "en"), chunk("001", "ar"), chunk("002", "en")]
# Deliberately unnormalised, to prove the store normalises internally.
VECTORS = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 5.0]]


@pytest.fixture
def store(tmp_path: Path) -> NumpyVectorStore:
    store = NumpyVectorStore(tmp_path, "test")
    store.add(CHUNKS, VECTORS)
    return store


class TestSearch:
    """Nearest-neighbour behaviour."""

    def test_finds_the_nearest_vector(self, store: NumpyVectorStore) -> None:
        hits = store.search([1.0, 0.0, 0.0], k=1)
        assert hits[0].chunk.chunk_id == "001::en"
        assert hits[0].score == pytest.approx(1.0)

    def test_scores_are_cosine_not_dot_product(self, store: NumpyVectorStore) -> None:
        """Vector magnitude must not influence ranking."""
        hits = store.search([0.0, 0.0, 1.0], k=1)
        assert hits[0].chunk.chunk_id == "002::en"
        assert hits[0].score == pytest.approx(1.0)

    def test_results_are_sorted_by_descending_score(self, store: NumpyVectorStore) -> None:
        hits = store.search([1.0, 1.0, 1.0], k=3)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_language_filter(self, store: NumpyVectorStore) -> None:
        hits = store.search([1.0, 1.0, 1.0], k=5, lang="ar")
        assert hits and all(hit.chunk.lang == "ar" for hit in hits)

    def test_filter_with_no_matches_returns_empty(self, store: NumpyVectorStore) -> None:
        assert store.search([1.0, 0.0, 0.0], k=5, lang="fr") == []

    def test_k_larger_than_corpus_is_clamped(self, store: NumpyVectorStore) -> None:
        assert len(store.search([1.0, 0.0, 0.0], k=99)) == 3

    def test_zero_query_vector_does_not_produce_nan(self, store: NumpyVectorStore) -> None:
        hits = store.search([0.0, 0.0, 0.0], k=3)
        assert all(hit.score == hit.score for hit in hits)  # NaN != NaN

    def test_searching_an_empty_store_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="empty"):
            NumpyVectorStore(tmp_path, "empty").search([1.0], k=1)

    def test_query_of_the_wrong_dimension_names_the_fix(
        self, store: NumpyVectorStore
    ) -> None:
        """A changed embedding model must not surface as a NumPy shape error."""
        with pytest.raises(RuntimeError, match="faqrag.index"):
            store.search([1.0, 0.0, 0.0, 0.0], k=1)


class TestAdd:
    """Index construction guards."""

    def test_mismatched_lengths_are_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="chunks but"):
            NumpyVectorStore(tmp_path, "t").add(CHUNKS, VECTORS[:2])

    def test_empty_index_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty index"):
            NumpyVectorStore(tmp_path, "t").add([], [])

    def test_add_replaces_previous_contents(self, store: NumpyVectorStore) -> None:
        store.add([chunk("099")], [[1.0, 0.0, 0.0]])
        assert store.count() == 1


class TestPersistence:
    """Round-tripping the index through disk."""

    def test_persist_and_load(self, tmp_path: Path, store: NumpyVectorStore) -> None:
        store.persist()

        reloaded = NumpyVectorStore(tmp_path, "test")
        assert reloaded.load() is True
        assert reloaded.count() == 3
        assert reloaded.search([1.0, 0.0, 0.0], k=1)[0].chunk.chunk_id == "001::en"

    def test_metadata_survives_the_round_trip(self, tmp_path: Path, store: NumpyVectorStore) -> None:
        store.persist()
        reloaded = NumpyVectorStore(tmp_path, "test")
        reloaded.load()

        original = {c.chunk_id: c for c in CHUNKS}
        for loaded in reloaded.all_chunks():
            assert loaded == original[loaded.chunk_id]

    def test_load_returns_false_when_absent(self, tmp_path: Path) -> None:
        assert NumpyVectorStore(tmp_path, "missing").load() is False

    def test_persist_before_add_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="nothing to persist"):
            NumpyVectorStore(tmp_path, "t").persist()

    def test_chunk_vector_count_mismatch_is_detected(
        self, tmp_path: Path, store: NumpyVectorStore
    ) -> None:
        """A chunk/vector count mismatch must fail loudly. Silently loading a
        misaligned index would attribute every answer to the wrong FAQ."""
        store.persist()
        chunks_path = tmp_path / "test.chunks.json"
        # Drop one chunk, leaving 2 chunks against 3 persisted vectors.
        remaining = json.loads(chunks_path.read_text(encoding="utf-8"))[:-1]
        chunks_path.write_text(json.dumps(remaining, ensure_ascii=False), encoding="utf-8")

        reloaded = NumpyVectorStore(tmp_path, "test")
        with pytest.raises(RuntimeError, match="corrupt"):
            reloaded.load()


class TestBuildVectorStore:
    """Backend selection via configuration."""

    def test_numpy_backend(self, tmp_path: Path) -> None:
        settings = Settings(vector_store="numpy", index_dir=tmp_path)
        assert isinstance(build_vector_store(settings), NumpyVectorStore)

    def test_unknown_backend_is_rejected(self, tmp_path: Path) -> None:
        settings = Settings(index_dir=tmp_path)
        # Bypass validation to simulate a bad value reaching the factory.
        object.__setattr__(settings, "vector_store", "qdrant")
        with pytest.raises(ValueError, match="unknown vector store"):
            build_vector_store(settings)
