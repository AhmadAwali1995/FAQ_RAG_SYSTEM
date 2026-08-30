"""Tests for FAQ ingestion and chunking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from faqrag.ingest import (
    IngestionError,
    load_chunks,
    make_chunk_id,
    parse_faq_payload,
    records_to_chunks,
    summarise,
)
from faqrag.models import FaqRecord


def make_record(faq_id: str = "001", lang: str = "en", **overrides) -> dict:
    """Build a valid raw FAQ record for tests."""
    record = {
        "faq_id": faq_id,
        "category": "About Mwfaq",
        "lang": lang,
        "question": "What is the Mwfaq platform?",
        "answer": "Mwfaq automates mandatory medical examinations in Saudi Arabia.",
        "keywords": ["Mwfaq", "platform"],
    }
    record.update(overrides)
    return record


class TestChunking:
    """Chunk construction from FAQ records."""

    def test_language_pair_becomes_two_chunks(self) -> None:
        """Arabic and English of one FAQ must stay separate retrievable units."""
        records = [
            FaqRecord.model_validate(make_record("001", "en")),
            FaqRecord.model_validate(
                make_record(
                    "001",
                    "ar",
                    question="ما هي منصة موفق؟",
                    answer="موفق منصة رقمية.",
                    keywords=["موفق", "منصة"],
                )
            ),
        ]
        chunks = records_to_chunks(records)

        assert len(chunks) == 2
        assert {c.lang for c in chunks} == {"ar", "en"}
        assert {c.faq_id for c in chunks} == {"001"}
        # Neither chunk may contain the other language's text.
        english, arabic = chunks[0], chunks[1]
        assert "موفق" not in english.embedding_text
        assert "Mwfaq" not in arabic.embedding_text

    def test_chunk_ids_are_stable_and_unique(self) -> None:
        chunks = records_to_chunks(
            [
                FaqRecord.model_validate(make_record("001", "en")),
                FaqRecord.model_validate(make_record("001", "ar")),
                FaqRecord.model_validate(make_record("002", "en")),
            ]
        )
        ids = [c.chunk_id for c in chunks]
        assert ids == ["001::en", "001::ar", "002::en"]
        assert len(set(ids)) == len(ids)
        assert make_chunk_id("007", "ar") == "007::ar"

    def test_duplicate_pair_is_rejected(self) -> None:
        """A repeated (faq_id, lang) would silently shadow a chunk in the index."""
        records = [
            FaqRecord.model_validate(make_record("001", "en")),
            FaqRecord.model_validate(make_record("001", "en", question="Duplicate?")),
        ]
        with pytest.raises(IngestionError, match="duplicate chunk id"):
            records_to_chunks(records)

    def test_metadata_is_preserved(self) -> None:
        chunk = records_to_chunks([FaqRecord.model_validate(make_record())])[0]
        assert chunk.faq_id == "001"
        assert chunk.category == "About Mwfaq"
        assert chunk.keywords == ["Mwfaq", "platform"]
        # Original text is kept verbatim so retrieval returns source content.
        assert chunk.question == "What is the Mwfaq platform?"


class TestEmbeddingText:
    """The text handed to the embedding model."""

    def test_includes_question_answer_and_keywords(self) -> None:
        """Answers carry most of the retrievable signal, so both are embedded."""
        chunk = records_to_chunks([FaqRecord.model_validate(make_record())])[0]
        text = chunk.embedding_text
        assert "What is the Mwfaq platform?" in text
        assert "automates mandatory medical examinations" in text
        assert "platform" in text

    def test_handles_missing_keywords(self) -> None:
        chunk = records_to_chunks(
            [FaqRecord.model_validate(make_record(keywords=[]))]
        )[0]
        assert chunk.embedding_text.count("\n") == 1

    def test_lexical_text_covers_all_searchable_fields(self) -> None:
        chunk = records_to_chunks([FaqRecord.model_validate(make_record())])[0]
        assert "Mwfaq platform" in chunk.lexical_text
        assert "medical examinations" in chunk.lexical_text
        assert "platform" in chunk.lexical_text


class TestParsePayload:
    """Validation of the source JSON payload."""

    def test_valid_payload(self) -> None:
        chunks = parse_faq_payload({"faqs": [make_record("001", "en"), make_record("001", "ar")]})
        assert len(chunks) == 2

    def test_missing_faqs_key(self) -> None:
        with pytest.raises(IngestionError, match="no top-level 'faqs' key"):
            parse_faq_payload({"items": []})

    def test_faqs_not_a_list(self) -> None:
        with pytest.raises(IngestionError, match="must be a list"):
            parse_faq_payload({"faqs": {"001": "..."}})

    def test_empty_corpus_is_rejected(self) -> None:
        with pytest.raises(IngestionError, match="no FAQ records"):
            parse_faq_payload({"faqs": []})

    def test_invalid_record_reports_its_index(self) -> None:
        payload = {"faqs": [make_record(), {"faq_id": "002"}]}
        with pytest.raises(IngestionError, match="index 1"):
            parse_faq_payload(payload)

    def test_unknown_language_is_rejected(self) -> None:
        with pytest.raises(IngestionError):
            parse_faq_payload({"faqs": [make_record(lang="fr")]})


class TestLoadChunks:
    """Loading from disk."""

    def test_missing_file_explains_how_to_generate_it(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="md_to_json"):
            load_chunks(tmp_path / "absent.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(IngestionError, match="not valid JSON"):
            load_chunks(path)

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"
        path.write_text(
            json.dumps({"faqs": [make_record("001", "en"), make_record("001", "ar")]}),
            encoding="utf-8",
        )
        assert len(load_chunks(path)) == 2


class TestSummarise:
    """Corpus statistics used in indexing logs."""

    def test_counts_by_language_and_category(self) -> None:
        chunks = records_to_chunks(
            [
                FaqRecord.model_validate(make_record("001", "en")),
                FaqRecord.model_validate(make_record("001", "ar")),
                FaqRecord.model_validate(make_record("002", "en", category="Payments")),
            ]
        )
        stats = summarise(chunks)
        assert stats["chunks"] == 3
        assert stats["faqs"] == 2
        assert stats["by_lang"] == {"en": 2, "ar": 1}
        assert stats["by_category"] == {"About Mwfaq": 2, "Payments": 1}


class TestRealCorpus:
    """Guards on the actual shipped dataset."""

    @pytest.fixture
    def corpus_path(self) -> Path:
        path = Path(__file__).resolve().parents[1] / "data" / "mwfaq_faq_rag.json"
        if not path.exists():
            pytest.skip("corpus not generated; run scripts/md_to_json.py")
        return path

    def test_every_faq_has_both_languages(self, corpus_path: Path) -> None:
        chunks = load_chunks(corpus_path)
        by_faq: dict[str, set[str]] = {}
        for chunk in chunks:
            by_faq.setdefault(chunk.faq_id, set()).add(chunk.lang)
        unpaired = {fid: langs for fid, langs in by_faq.items() if langs != {"ar", "en"}}
        assert not unpaired, f"FAQs missing a language pair: {unpaired}"

    def test_no_markdown_artefacts_leaked_into_content(self, corpus_path: Path) -> None:
        """The converter must not carry '---' rules or '##' headings into text."""
        for chunk in load_chunks(corpus_path):
            fields = [chunk.question, chunk.answer, *chunk.keywords]
            for field in fields:
                assert "---" not in field, f"{chunk.chunk_id}: {field!r}"
                assert "##" not in field, f"{chunk.chunk_id}: {field!r}"
                assert "\n" not in field, f"{chunk.chunk_id}: {field!r}"

    def test_no_bidi_control_marks_remain(self, corpus_path: Path) -> None:
        for chunk in load_chunks(corpus_path):
            assert "‏" not in chunk.answer
            assert "‎" not in chunk.answer
