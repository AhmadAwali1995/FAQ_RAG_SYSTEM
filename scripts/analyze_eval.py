"""Analyse an eval report and compare it against the source FAQ document.

Turns ``python -m faqrag.eval --json report.json`` output into a per-FAQ and
per-category breakdown, so a headline hit-rate can be traced to the specific
document sections and question phrasings that fail.

Usage::

    python scripts/analyze_eval.py saudi_full.json
    python scripts/analyze_eval.py saudi_full.json --compare eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    """Load an eval report written by ``faqrag.eval --json``."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(path: Path) -> dict[str, dict[str, str]]:
    """Map ``faq_id`` to its Arabic question and category."""
    faqs = json.loads(path.read_text(encoding="utf-8"))["faqs"]
    return {
        f["faq_id"]: {"question": f["question"], "category": f["category"]}
        for f in faqs
        if f["lang"] == "ar"
    }


def per_faq_breakdown(cases: list[dict], corpus: dict) -> list[tuple]:
    """Count hits and misses per expected FAQ id."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"hit": 0, "miss": 0})
    for case in cases:
        if case["hit"] is None:
            continue
        for faq_id in case["expected_faq_ids"]:
            stats[faq_id]["hit" if case["hit"] else "miss"] += 1

    rows = []
    for faq_id in sorted(stats):
        hit, miss = stats[faq_id]["hit"], stats[faq_id]["miss"]
        info = corpus.get(faq_id, {})
        rows.append((faq_id, hit, miss, hit / (hit + miss), info.get("category", "?"),
                     info.get("question", "?")))
    return rows


def per_category_breakdown(cases: list[dict], corpus: dict) -> list[tuple]:
    """Aggregate hits and misses by the FAQ's document category."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"hit": 0, "miss": 0})
    for case in cases:
        if case["hit"] is None:
            continue
        # Attribute the case to the category of its first expected FAQ.
        first = case["expected_faq_ids"][0] if case["expected_faq_ids"] else None
        category = corpus.get(first, {}).get("category", "Unknown")
        stats[category]["hit" if case["hit"] else "miss"] += 1

    return sorted(
        (
            (cat, s["hit"], s["hit"] + s["miss"], s["hit"] / (s["hit"] + s["miss"]))
            for cat, s in stats.items()
        ),
        key=lambda row: (row[3], -row[2]),
    )


def bar(fraction: float, width: int = 18) -> str:
    """Render a fraction as an ASCII meter."""
    filled = round(fraction * width)
    return "#" * filled + "." * (width - filled)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def report(data: dict, corpus: dict, label: str) -> None:
    """Print the full breakdown for one report."""
    cases = data["cases"]
    metrics = data["metrics"]

    section(f"HEADLINE METRICS - {label}")
    for key in (
        "total_cases", "retrieval_hit_rate@5", "retrieval_scored_cases",
        "out_of_scope_refusal_rate", "in_scope_answer_rate", "groundedness_rate",
        "citation_accuracy", "fabrication_free_rate", "language_match_rate",
        "median_latency_ms",
    ):
        if key in metrics and metrics[key] is not None:
            print(f"  {key:<28} {metrics[key]}")

    section("BY DOCUMENT CATEGORY (retrieval)")
    print(f"  {'category':<38} {'hit/total':>10}  rate")
    for category, hits, total, rate in per_category_breakdown(cases, corpus):
        print(f"  {category:<38} {f'{hits}/{total}':>10}  {rate:5.0%}  {bar(rate)}")

    section("BY FAQ (only entries with a miss)")
    rows = [r for r in per_faq_breakdown(cases, corpus) if r[2] > 0]
    if not rows:
        print("  every FAQ was retrieved correctly for every question that expects it")
    for faq_id, hit, miss, rate, category, question in rows:
        print(f"  FAQ {faq_id}  {hit}/{hit+miss} ({rate:.0%})  [{category}]")
        print(f"           doc: {question}")

    section("FAILED CASES")
    failures = [
        c for c in cases
        if c["hit"] is False
        or (c["kind"] == "out_of_scope" and c["answered"])
        or c.get("fabricated")
        or c.get("grounded") is False
    ]
    if not failures:
        print("  none")
    for case in failures:
        reasons = []
        if case["hit"] is False:
            reasons.append("RETRIEVAL MISS")
        if case["kind"] == "out_of_scope" and case["answered"]:
            reasons.append("SHOULD HAVE DECLINED")
        if case.get("fabricated"):
            reasons.append("FABRICATED")
        if case.get("grounded") is False:
            reasons.append("UNGROUNDED")
        print(f"\n  [{', '.join(reasons)}]  ({case['kind']})")
        print(f"    Q        : {case['question']}")
        if case["expected_faq_ids"]:
            print(f"    expected : {case['expected_faq_ids']}")
        print(f"    retrieved: {case['retrieved_faq_ids']}")
        print(f"    cited    : {case['cited_faq_ids']}  conf={case['confidence']}")
        if case.get("unsupported_claims"):
            for claim in case["unsupported_claims"]:
                print(f"    ! {claim}")
        answer = (case.get("answer") or "").replace("\n", " ")
        print(f"    answer   : {answer[:180]}")

    section("CONFIDENCE SEPARATION (does the threshold still hold?)")
    in_scope = [c["confidence"] for c in cases if c["kind"] != "out_of_scope" and c["answered"]]
    oos = [c["confidence"] for c in cases if c["kind"] == "out_of_scope"]
    if in_scope and oos:
        print(f"  answered in-scope : min={min(in_scope):.3f}  median={sorted(in_scope)[len(in_scope)//2]:.3f}  max={max(in_scope):.3f}  (n={len(in_scope)})")
        print(f"  out-of-scope      : min={min(oos):.3f}  median={sorted(oos)[len(oos)//2]:.3f}  max={max(oos):.3f}  (n={len(oos)})")
        gap = min(in_scope) - max(oos)
        verdict = "clean separation" if gap > 0 else f"OVERLAP of {abs(gap):.3f}"
        print(f"  gap               : {gap:+.3f}  -> {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Eval report JSON to analyse.")
    parser.add_argument("--compare", type=Path, default=None, help="A second report to contrast.")
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/mwfaq_faq_rag.json"), help="Source FAQ JSON."
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    corpus = load_corpus(args.corpus)
    main_data = load(args.report)
    report(main_data, corpus, args.report.name)

    if args.compare:
        other = load(args.compare)
        report(other, corpus, args.compare.name)

        section("SIDE BY SIDE")
        print(f"  {'metric':<28} {args.report.stem:>18} {args.compare.stem:>18}")
        keys = sorted(set(main_data["metrics"]) | set(other["metrics"]))
        for key in keys:
            a, b = main_data["metrics"].get(key), other["metrics"].get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                print(f"  {key:<28} {a:>18} {b:>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
