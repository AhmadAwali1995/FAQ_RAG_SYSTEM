"""Convert the Mwfaq FAQ markdown knowledge base into the RAG JSON schema.

The upstream knowledge base is authored as markdown (``data/mwfaq_faq_rag.md``)
with one ``### FAQ NNN`` block per FAQ, each containing an Arabic and an English
``**Language:**`` sub-block.  This script flattens that into the JSON schema the
indexer consumes::

    {"faqs": [{"faq_id", "category", "lang", "question", "answer", "keywords"}]}

Run it whenever the markdown source changes::

    python scripts/md_to_json.py data/mwfaq_faq_rag.md data/mwfaq_faq_rag.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

# Directional marks the source uses to force RTL rendering; they carry no
# semantic content and would otherwise leak into embeddings and BM25 tokens.
_BIDI_MARKS = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩"))

_LANG_NAMES = {"arabic": "ar", "english": "en"}

_CATEGORY_RE = re.compile(r"^##\s+(?!#)(.+?)\s*$", re.MULTILINE)
_FAQ_RE = re.compile(r"^###\s+FAQ\s+([0-9A-Za-z_-]+)\s*$", re.MULTILINE)
_FIELD_RE = re.compile(r"^\*\*(Language|Question|Answer|Keywords):\*\*\s*(.*)$", re.MULTILINE)

# A field's text ends at the next markdown structure: a horizontal rule (which
# separates FAQ blocks) or any heading (the next category).  Without this the
# trailing keyword line of each FAQ absorbs the "---" and "## Category" that
# follow it.
_STRUCTURAL_RE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,}|#{1,6}[ \t])", re.MULTILINE)


def clean_text(text: str) -> str:
    """Strip bidi control marks, normalise unicode, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_BIDI_MARKS)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def truncate_at_structure(text: str) -> str:
    """Cut ``text`` at the first markdown rule or heading line."""
    match = _STRUCTURAL_RE.search(text)
    return text[: match.start()] if match else text


def split_keywords(raw: str) -> list[str]:
    """Split a keyword line on ASCII or Arabic commas, dropping empties."""
    parts = re.split(r"[,،؛;]", raw)
    return [kw for kw in (clean_text(p) for p in parts) if kw]


def _parse_fields(block: str) -> list[dict[str, str]]:
    """Parse a FAQ block into one dict per ``**Language:**`` sub-block."""
    matches = list(_FIELD_RE.finditer(block))
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for i, match in enumerate(matches):
        field = match.group(1).lower()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        # The value may sit on the same line as the label or on following lines.
        raw = match.group(2) + "\n" + block[match.end() : end]
        value = clean_text(truncate_at_structure(raw))

        if field == "language":
            if current:
                entries.append(current)
            current = {"language": value}
        elif current:
            current[field] = value

    if current:
        entries.append(current)
    return entries


def parse_markdown(markdown: str) -> list[dict[str, Any]]:
    """Parse the FAQ markdown into a flat list of ``{faq_id, lang, ...}`` records.

    Each FAQ contributes one record per language, so an FAQ with an Arabic and
    an English sub-block yields two records sharing the same ``faq_id``.
    """
    # Map each character offset to the category heading in effect at that point.
    categories = [(m.start(), clean_text(m.group(1))) for m in _CATEGORY_RE.finditer(markdown)]

    def category_at(pos: int) -> str:
        name = "Uncategorized"
        for start, title in categories:
            if start < pos:
                name = title
            else:
                break
        return name

    faq_matches = list(_FAQ_RE.finditer(markdown))
    records: list[dict[str, Any]] = []

    for i, match in enumerate(faq_matches):
        faq_id = match.group(1)
        end = faq_matches[i + 1].start() if i + 1 < len(faq_matches) else len(markdown)
        block = markdown[match.end() : end]
        category = category_at(match.start())

        for entry in _parse_fields(block):
            lang = _LANG_NAMES.get(entry.get("language", "").lower())
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            if not (lang and question and answer):
                print(
                    f"warning: skipping incomplete sub-block in FAQ {faq_id} "
                    f"(lang={entry.get('language')!r})",
                    file=sys.stderr,
                )
                continue
            records.append(
                {
                    "faq_id": faq_id,
                    "category": category,
                    "lang": lang,
                    "question": question,
                    "answer": answer,
                    "keywords": split_keywords(entry.get("keywords", "")),
                }
            )

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert FAQ markdown to RAG JSON.")
    parser.add_argument("source", type=Path, help="Path to the FAQ markdown file")
    parser.add_argument("destination", type=Path, help="Path to write the JSON to")
    args = parser.parse_args()

    records = parse_markdown(args.source.read_text(encoding="utf-8"))
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(
        json.dumps({"faqs": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    ids = {r["faq_id"] for r in records}
    langs = sorted({r["lang"] for r in records})
    unpaired = sorted(i for i in ids if sum(r["faq_id"] == i for r in records) != len(langs))
    print(f"wrote {len(records)} records ({len(ids)} FAQs, langs={langs}) -> {args.destination}")
    if unpaired:
        print(f"warning: FAQs without a full language pair: {unpaired}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
