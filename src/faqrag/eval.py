"""Evaluation harness for retrieval quality and answer groundedness.

Run after any change to chunking, the embedding model, fusion, or the prompts::

    python -m faqrag.eval                    # retrieval + generation + judge
    python -m faqrag.eval --retrieval-only   # fast, no LLM calls
    python -m faqrag.eval --k 3 --json report.json

Two things are measured:

* **Retrieval hit-rate@k** -- did any expected FAQ id appear in the top k?
  Reported overall and split by query kind, because a single averaged number
  hides that clean questions are easy and paraphrases are not.
* **Answer groundedness** -- an LLM judge checks whether every claim in the
  generated answer traces back to the cited chunks. This is the check that
  catches a fluent, confident, invented answer, which retrieval metrics cannot.

Out-of-scope cases are scored on *refusal*, not retrieval: the correct behaviour
is declining to answer, so counting them as retrieval misses would be wrong.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Settings, get_settings
from .eval_data import EVAL_CASES, EvalCase
from .eval_saudi import SAUDI_CASES
from .llm import LLMClient, LLMError, build_llm
from .logging_utils import configure_logging, ensure_utf8_stdout
from .models import QueryResponse
from .pipeline import RagPipeline

logger = logging.getLogger(__name__)

#: Selectable question sets. "core" is MSA/English; "saudi" is 100 questions in
#: Saudi colloquial Arabic, which measures the dialect gap the core set cannot.
SUITES: dict[str, tuple[EvalCase, ...]] = {
    "core": EVAL_CASES,
    "saudi": SAUDI_CASES,
    "all": EVAL_CASES + SAUDI_CASES,
}

GROUNDEDNESS_SYSTEM_PROMPT = """\
You audit whether an answer is fully supported by the source FAQ entries it cites.

You are given SOURCES (FAQ entries) and an ANSWER. Decide whether every factual \
claim in the ANSWER is supported by the SOURCES.

Treat as UNSUPPORTED any phone number, email address, price, fee, percentage, \
date, or policy detail that does not appear in the SOURCES, even if it sounds \
plausible. Do not use outside knowledge: your own belief that a claim is true is \
irrelevant if the SOURCES do not state it.

Ignore language: an answer may translate its sources. An answer that declines to \
give a detail is grounded, not unsupported.

Respond with ONLY this JSON object, no prose and no markdown fences:
{"grounded": true or false, "unsupported_claims": ["..."], "reason": "one sentence"}"""


@dataclass
class CaseResult:
    """Per-case evaluation outcome."""

    question: str
    lang: str
    kind: str
    expected_faq_ids: list[str]
    retrieved_faq_ids: list[str]
    hit: bool | None
    confident: bool
    confidence: float
    answered: bool
    refusal_correct: bool | None
    language_correct: bool | None
    cited_faq_ids: list[str] = field(default_factory=list)
    grounded: bool | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    judge_reason: str = ""
    citation_correct: bool | None = None
    fabricated: bool | None = None
    answer: str = ""
    latency_ms: float = 0.0


def looks_like_refusal(response: QueryResponse) -> bool:
    """Whether the pipeline declined to answer.

    Keyed off structured signals -- no confident match, or no citations -- rather
    than matching refusal wording, which would break the moment the phrasing or
    the language changes.
    """
    return not response.confident or not response.cited_faq_ids


def evaluate_retrieval(
    case: EvalCase, response: QueryResponse, k: int
) -> tuple[bool | None, bool]:
    """Score one case's retrieval and refusal behaviour.

    Returns:
        ``(hit, refusal_correct)``. ``hit`` is ``None`` for out-of-scope cases,
        which have no expected ids to hit.
    """
    refused = looks_like_refusal(response)

    if case.expects_refusal:
        return None, refused

    # Scored against what retrieval surfaced, not what the model cited. Citing
    # the wrong FAQ is a generation fault; folding it in here would report it as
    # a retrieval miss and send you tuning the wrong stage.
    retrieved = [source.faq_id for source in response.retrieved[:k]]
    hit = any(faq_id in case.expected_faq_ids for faq_id in retrieved)
    # An in-scope question should have been answered, so refusing is an error.
    return hit, not refused


def judge_groundedness(
    client: LLMClient, response: QueryResponse
) -> tuple[bool | None, list[str], str]:
    """Ask an LLM judge whether the answer is supported by its cited sources.

    Returns:
        ``(grounded, unsupported_claims, reason)``. ``grounded`` is ``None`` when
        the judge could not be run or its reply could not be parsed -- an
        unavailable judge must not be recorded as a groundedness failure.
    """
    if not response.sources:
        return None, [], "no sources to judge"

    sources = "\n\n".join(
        f"[FAQ {s.faq_id}]\nQ: {s.question}\nA: {s.answer}" for s in response.sources
    )
    user = (
        f"SOURCES:\n{sources}\n\n---\nANSWER:\n{response.answer}\n\nReturn the JSON verdict."
    )

    try:
        raw = client.complete(GROUNDEDNESS_SYSTEM_PROMPT, user, temperature=0.0)
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"no JSON object in judge reply: {raw[:200]!r}")
        verdict = json.loads(raw[start : end + 1])
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("groundedness judge failed: %s", exc)
        return None, [], f"judge unavailable: {exc}"

    claims = verdict.get("unsupported_claims") or []
    return (
        bool(verdict.get("grounded")),
        [str(c) for c in claims],
        str(verdict.get("reason", "")),
    )


def run_case(
    pipeline: RagPipeline,
    case: EvalCase,
    k: int,
    judge: LLMClient | None,
) -> CaseResult:
    """Run a single evaluation case end to end."""
    started = time.perf_counter()
    response = pipeline.answer(case.question, top_k=k)
    hit, refusal_correct = evaluate_retrieval(case, response, k)
    answered = not looks_like_refusal(response)

    result = CaseResult(
        question=case.question,
        lang=case.lang,
        kind=case.kind,
        expected_faq_ids=list(case.expected_faq_ids),
        retrieved_faq_ids=[s.faq_id for s in response.retrieved[:k]],
        hit=hit,
        confident=response.confident,
        confidence=response.confidence,
        answered=answered,
        refusal_correct=refusal_correct,
        # Only meaningful when an answer was actually produced.
        language_correct=(response.language == case.lang) if answered else None,
        cited_faq_ids=list(response.cited_faq_ids),
        # Whether the model cited a FAQ that actually answers the question.
        # Distinct from `hit`: retrieval can succeed while the citation is wrong.
        citation_correct=(
            any(fid in case.expected_faq_ids for fid in response.cited_faq_ids)
            if (answered and case.expected_faq_ids)
            else None
        ),
        # A deterministic check that no forbidden value was invented, run
        # independently of the LLM judge so it cannot be talked out of a failure.
        fabricated=(
            case.fabricated(response.answer) if case.forbidden_pattern is not None else None
        ),
        answer=response.answer,
        latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
    )

    if judge is not None and answered:
        grounded, claims, reason = judge_groundedness(judge, response)
        result.grounded = grounded
        result.unsupported_claims = claims
        result.judge_reason = reason

    return result


def summarise(results: list[CaseResult], k: int) -> dict:
    """Aggregate case results into the report's metrics block."""
    scored = [r for r in results if r.hit is not None]
    hits = [r for r in scored if r.hit]
    out_of_scope = [r for r in results if r.kind == "out_of_scope"]
    in_scope = [r for r in results if r.kind != "out_of_scope"]
    judged = [r for r in results if r.grounded is not None]
    languages = [r for r in results if r.language_correct is not None]
    fabrication_checked = [r for r in results if r.fabricated is not None]
    cited = [r for r in results if r.citation_correct is not None]

    by_kind: dict[str, dict] = {}
    for kind in sorted({r.kind for r in results}):
        subset = [r for r in results if r.kind == kind and r.hit is not None]
        if subset:
            kind_hits = sum(1 for r in subset if r.hit)
            by_kind[kind] = {
                "cases": len(subset),
                "hits": kind_hits,
                "hit_rate": round(kind_hits / len(subset), 3),
            }

    return {
        "k": k,
        "total_cases": len(results),
        f"retrieval_hit_rate@{k}": round(len(hits) / len(scored), 3) if scored else None,
        "retrieval_scored_cases": len(scored),
        "hit_rate_by_kind": by_kind,
        "out_of_scope_refusal_rate": (
            round(sum(1 for r in out_of_scope if r.refusal_correct) / len(out_of_scope), 3)
            if out_of_scope
            else None
        ),
        "in_scope_answer_rate": (
            round(sum(1 for r in in_scope if r.answered) / len(in_scope), 3)
            if in_scope
            else None
        ),
        "groundedness_rate": (
            round(sum(1 for r in judged if r.grounded) / len(judged), 3) if judged else None
        ),
        "groundedness_judged_cases": len(judged),
        "citation_accuracy": (
            round(sum(1 for r in cited if r.citation_correct) / len(cited), 3)
            if cited
            else None
        ),
        "fabrication_free_rate": (
            round(
                sum(1 for r in fabrication_checked if not r.fabricated)
                / len(fabrication_checked),
                3,
            )
            if fabrication_checked
            else None
        ),
        "language_match_rate": (
            round(sum(1 for r in languages if r.language_correct) / len(languages), 3)
            if languages
            else None
        ),
        "median_latency_ms": (
            round(statistics.median(r.latency_ms for r in results), 1) if results else None
        ),
    }


def print_report(results: list[CaseResult], metrics: dict) -> None:
    """Print a human-readable evaluation report."""
    print("\n" + "=" * 78)
    print("PER-CASE RESULTS")
    print("=" * 78)
    for result in results:
        if result.kind == "out_of_scope":
            mark = "PASS" if result.refusal_correct else "FAIL"
            detail = "declined" if not result.answered else "ANSWERED (should decline)"
        else:
            passed = bool(result.hit) and not result.fabricated
            mark = "PASS" if passed else "FAIL"
            detail = f"expected {result.expected_faq_ids} got {result.retrieved_faq_ids}"

        flags = []
        if result.fabricated:
            flags.append("FABRICATED")
        if result.citation_correct is False:
            flags.append("MIS-CITED")
        if result.grounded is False:
            flags.append("UNGROUNDED")
        if result.language_correct is False:
            flags.append("WRONG-LANG")
        suffix = ("  [" + ", ".join(flags) + "]") if flags else ""

        print(f"\n[{mark}] ({result.kind}/{result.lang}) {result.question}")
        print(
            f"       {detail}  conf={result.confidence:.3f}  "
            f"{result.latency_ms:.0f}ms{suffix}"
        )
        for claim in result.unsupported_claims:
            print(f"       ! unsupported: {claim}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for key, value in metrics.items():
        if key == "hit_rate_by_kind":
            print(f"  {key}:")
            for kind, stats in value.items():
                print(
                    f"      {kind:<14} {stats['hits']}/{stats['cases']}  ({stats['hit_rate']})"
                )
        else:
            print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m faqrag.eval``."""
    parser = argparse.ArgumentParser(description="Evaluate the FAQ RAG system.")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="core",
        help="Question set to run: 'core' (MSA/English), 'saudi' (100 dialect questions), 'all'.",
    )
    parser.add_argument(
        "--k", type=int, default=None, help="top-k for hit-rate (default: config top_k)"
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and judging; measures retrieval only, with no LLM calls.",
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="Generate answers but skip the judge."
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the full report as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Log retrieval detail.")
    args = parser.parse_args(argv)

    ensure_utf8_stdout()
    settings: Settings = get_settings()
    configure_logging("INFO" if args.verbose else "WARNING")
    k = args.k or settings.top_k

    if args.retrieval_only:
        # Extractive generation keeps the pipeline shape identical while making
        # the run fully local and fast, so retrieval can be iterated on quickly.
        settings.llm_provider = "extractive"
        settings.rerank_enabled = False

    pipeline = RagPipeline.from_settings(settings)

    judge: LLMClient | None = None
    if not (args.retrieval_only or args.no_judge):
        try:
            judge = build_llm(settings)
        except LLMError as exc:
            print(f"warning: groundedness judge unavailable ({exc})", file=sys.stderr)

    cases = SUITES[args.suite]
    print(f"suite={args.suite}  cases={len(cases)}  k={k}", file=sys.stderr)

    results = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case.question[:60]}", file=sys.stderr)
        results.append(run_case(pipeline, case, k, judge))

    metrics = summarise(results, k)
    metrics["suite"] = args.suite
    print_report(results, metrics)

    if args.json:
        args.json.write_text(
            json.dumps(
                {"metrics": metrics, "cases": [asdict(r) for r in results]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote report to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
