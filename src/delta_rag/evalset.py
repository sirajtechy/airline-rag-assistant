"""Loading and validating the labelled evaluation set.

The validator is not ceremony. If a ``gold_span`` does not actually occur at its
``gold_locator``, every downstream metric silently becomes wrong, and the entire
ablation study is built on sand. So the eval set is checked against all three
parsers before any experiment is allowed to run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import BUSINESS_LINES, EVAL_DIR, SOURCE_DOCS
from .corpus import SourceUnit, load_corpus, normalise_for_match

QUERY_TYPES = ("keyword", "semantic")

# Labels are authored and gated against one parser so the ground truth is fixed
# while Stage 1 varies the parser under test. PyMuPDF is used because it recovers
# 100% of gold spans; see reports/results/stage1_parsing.md.
REFERENCE_PARSER = "pymupdf"


@dataclass
class EvalQuestion:
    id: str
    question: str
    route: str
    query_type: str
    gold_doc: str
    gold_locators: list[str]
    gold_span: str
    ideal_answer: str
    notes: str = ""
    meta: dict = field(default_factory=dict)


def load_eval_set(path: Path | None = None) -> list[EvalQuestion]:
    path = Path(path or EVAL_DIR / "eval_set.yaml")
    raw = yaml.safe_load(path.read_text())
    questions = [
        EvalQuestion(
            id=q["id"],
            question=q["question"],
            route=q["route"],
            query_type=q["query_type"],
            gold_doc=q["gold_doc"],
            gold_locators=list(q["gold_locators"]),
            gold_span=q["gold_span"],
            ideal_answer=q["ideal_answer"],
            notes=q.get("notes", ""),
        )
        for q in raw["questions"]
    ]
    _check_schema(questions)
    return questions


def _check_schema(questions: list[EvalQuestion]) -> None:
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise ValueError(f"duplicate question id {q.id!r}")
        seen.add(q.id)
        if q.route not in BUSINESS_LINES:
            raise ValueError(f"{q.id}: unknown route {q.route!r}")
        if q.query_type not in QUERY_TYPES:
            raise ValueError(f"{q.id}: unknown query_type {q.query_type!r}")
        if q.gold_doc not in SOURCE_DOCS:
            raise ValueError(f"{q.id}: unknown gold_doc {q.gold_doc!r}")
        if SOURCE_DOCS[q.gold_doc]["business_line"] != q.route:
            raise ValueError(
                f"{q.id}: route {q.route!r} disagrees with gold_doc business line "
                f"{SOURCE_DOCS[q.gold_doc]['business_line']!r}"
            )
        if not q.gold_locators or not q.gold_span.strip():
            raise ValueError(f"{q.id}: gold_locators and gold_span are required")


def validate_against_corpus(
    questions: list[EvalQuestion], units: list[SourceUnit]
) -> list[str]:
    """Return a list of human-readable problems; empty means the labels are sound."""
    by_key = {(u.doc_id, u.locator): u for u in units}
    by_doc: dict[str, list[SourceUnit]] = {}
    for u in units:
        by_doc.setdefault(u.doc_id, []).append(u)

    problems: list[str] = []
    for q in questions:
        span = normalise_for_match(q.gold_span)

        missing = [loc for loc in q.gold_locators if (q.gold_doc, loc) not in by_key]
        if missing:
            problems.append(f"{q.id}: locator(s) not present in corpus: {missing}")
            continue

        hit = any(
            span in normalise_for_match(by_key[(q.gold_doc, loc)].text)
            for loc in q.gold_locators
        )
        if not hit:
            # Distinguish "wrong locator" from "span not in the document at all",
            # because the two failures need very different fixes.
            elsewhere = [
                u.locator
                for u in by_doc.get(q.gold_doc, [])
                if span in normalise_for_match(u.text)
            ]
            if elsewhere:
                problems.append(
                    f"{q.id}: gold_span not at {q.gold_locators}, but found at {elsewhere}"
                )
            else:
                problems.append(
                    f"{q.id}: gold_span not found anywhere in {q.gold_doc}: {q.gold_span[:70]!r}"
                )
    return problems


def summarise(questions: list[EvalQuestion]) -> dict:
    from collections import Counter

    return {
        "total": len(questions),
        "by_route": dict(Counter(q.route for q in questions)),
        "by_query_type": dict(Counter(q.query_type for q in questions)),
        "by_doc": dict(Counter(q.gold_doc for q in questions)),
    }


def span_recovery(questions: list[EvalQuestion], units: list[SourceUnit]) -> float:
    """Fraction of gold spans still verbatim-recoverable from a parser's output.

    This is an informational Stage 1 signal, *not* a correctness gate. A parser
    that shatters a word ("t hose" instead of "those" — pypdf does this on
    contract page 4) still yields a page a human can read and an embedding model
    can largely match, so it is scored as degraded rather than absent. See
    ``delta_rag.metrics`` for how that distinction feeds NDCG@3 but not R@3.
    """
    if not questions:
        return 0.0
    return 1.0 - len(validate_against_corpus(questions, units)) / len(questions)


def main() -> int:
    """Gate the labels against the reference parser; report the rest as evidence."""
    questions = load_eval_set()
    print("eval set summary:", summarise(questions))

    problems = validate_against_corpus(questions, load_corpus(REFERENCE_PARSER))
    print(f"\n--- label gate (reference parser = {REFERENCE_PARSER}) ---")
    if problems:
        for p in problems:
            print("   FAIL:", p)
    else:
        print(f"   OK: all {len(questions)} gold spans verified at their gold locators")

    print("\n--- gold-span recoverability by parser (Stage 1 evidence) ---")
    for parser in ("pymupdf", "pypdf", "pdfplumber"):
        rate = span_recovery(questions, load_corpus(parser))
        print(f"   {parser:<12} {rate:6.1%}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
