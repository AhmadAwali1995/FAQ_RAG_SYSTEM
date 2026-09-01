"""Grounded answer generation over retrieved FAQ chunks."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterator

from .config import Settings
from .llm import LLMClient, LLMError
from .models import Language, RetrievalResult, ScoredChunk, SourceCitation
from .prompts import (
    INSUFFICIENT_CONTEXT_MARKER,
    build_answer_prompt,
    build_answer_system_prompt,
    casual_reply,
    no_match_message,
)

logger = logging.getLogger(__name__)

# Matches the trailing "SOURCES: 001, 007" line the system prompt mandates.
_SOURCES_RE = re.compile(r"^\s*SOURCES\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_FAQ_ID_RE = re.compile(r"[0-9A-Za-z_-]+")
_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?\u060c\u061b\u061f])")
_MISSING_SPACE_AFTER_SENTENCE_RE = re.compile(
    r"([.!?\u061f])(?=[A-Za-z\u0600-\u06ff])"
)


def clean_answer_text(text: str) -> str:
    """Normalise model output without changing its meaning."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)
    text = _MISSING_SPACE_AFTER_SENTENCE_RE.sub(r"\1 ", text)
    return text.strip()


def parse_sources(text: str, allowed_ids: set[str]) -> tuple[str, list[str]]:
    """Split a model reply into its answer body and cited FAQ ids.

    Citations are intersected with ``allowed_ids`` -- the FAQs actually placed in
    the context -- so a model that invents an id cannot produce a citation
    pointing at content it never saw.

    Args:
        text: The raw model reply.
        allowed_ids: FAQ ids that were supplied as context.

    Returns:
        ``(answer_without_sources_line, cited_ids)``, preserving citation order
        and dropping duplicates.
    """
    cited: list[str] = []
    for match in _SOURCES_RE.finditer(text):
        for token in _FAQ_ID_RE.findall(match.group(1)):
            if token in allowed_ids and token not in cited:
                cited.append(token)

    body = clean_answer_text(_SOURCES_RE.sub("", text))
    return body, cited


def to_citations(chunks: list[ScoredChunk], faq_ids: list[str]) -> list[SourceCitation]:
    """Build citation objects for ``faq_ids``, in the order the model cited them.

    Falls back to every retrieved chunk when the model cited nothing, so a UI
    always has something to render alongside the answer.
    """
    by_faq_id: dict[str, ScoredChunk] = {}
    for item in chunks:
        by_faq_id.setdefault(item.chunk.faq_id, item)

    selected = [by_faq_id[fid] for fid in faq_ids if fid in by_faq_id] or list(chunks)
    return [
        SourceCitation(
            faq_id=item.chunk.faq_id,
            category=item.chunk.category,
            lang=item.chunk.lang,
            question=item.chunk.question,
            answer=item.chunk.answer,
            score=round(item.relevance if item.relevance is not None else item.score, 4),
        )
        for item in selected
    ]


class AnswerGenerator:
    """Turns a :class:`RetrievalResult` into a grounded, cited answer.

    With ``llm_provider='extractive'`` (``client=None``) the top-ranked FAQ
    answer is returned verbatim. That mode cannot hallucinate by construction,
    at the cost of not synthesising across FAQs or matching the user's phrasing.
    """

    def __init__(self, settings: Settings, client: LLMClient | None) -> None:
        self.settings = settings
        self.client = client
        # Built once: the system prompt is fixed for the life of the generator.
        self.system_prompt = build_answer_system_prompt(settings.answer_style)

    def _no_match(self, lang: Language) -> str:
        """Return the refusal message in the configured voice."""
        return no_match_message(lang, self.settings.answer_style)

    def _extractive(self, result: RetrievalResult) -> tuple[str, list[str]]:
        top = result.chunks[0]
        return top.chunk.answer, [top.chunk.faq_id]

    def generate(self, result: RetrievalResult) -> tuple[str, list[str]]:
        """Generate an answer for ``result``.

        Returns:
            ``(answer_text, cited_faq_ids)``. When retrieval was not confident,
            or the model reports insufficient context, the answer is a fixed
            "I don't know" message in the query language and no ids are cited --
            the system declines rather than answering from a weak match.
        """
        # Refusing here, before the model ever sees a weak context, is what keeps
        # a low-relevance match from being dressed up as a confident answer.
        if not result.confident or not result.chunks:
            casual = casual_reply(result.query, result.query_lang)
            if casual:
                return casual, []
            logger.info("no confident match for %r; declining to answer", result.query)
            return self._no_match(result.query_lang), []

        if self.client is None:
            return self._extractive(result)

        prompt = build_answer_prompt(result.query, result.chunks, result.query_lang)
        try:
            raw = self.client.complete(self.system_prompt, prompt)
        except LLMError as exc:
            # Falling back to the retrieved text keeps the system useful during
            # an LLM outage without ever inventing content.
            logger.error("generation failed (%s); falling back to extractive answer", exc)
            return self._extractive(result)

        allowed = {item.chunk.faq_id for item in result.chunks}
        answer, cited = parse_sources(raw, allowed)

        if INSUFFICIENT_CONTEXT_MARKER in answer:
            logger.info("model reported insufficient context for %r", result.query)
            return self._no_match(result.query_lang), []

        if not answer:
            logger.warning("model returned only a sources line; using extractive fallback")
            return self._extractive(result)

        if not cited:
            # The model answered but skipped the citation line. Attribute to the
            # top chunk rather than returning an uncited answer.
            logger.warning("model omitted the SOURCES line; citing the top chunk")
            cited = [result.chunks[0].chunk.faq_id]

        return answer, cited

    def stream_generate(self, result: RetrievalResult) -> Iterator[str]:
        """Yield answer text fragments as soon as they are available."""
        if not result.confident or not result.chunks:
            casual = casual_reply(result.query, result.query_lang)
            if casual:
                yield casual
                return
            logger.info("no confident match for %r; declining to answer", result.query)
            yield self._no_match(result.query_lang)
            return

        if self.client is None:
            answer, _ = self._extractive(result)
            yield answer
            return

        prompt = build_answer_prompt(result.query, result.chunks, result.query_lang)
        try:
            yield from self.client.stream_complete(self.system_prompt, prompt)
        except LLMError as exc:
            logger.error("streaming generation failed (%s); falling back to extractive answer", exc)
            answer, _ = self._extractive(result)
            yield answer
