"""Retrieval metrics: R@1, R@3, MRR@10, DCG@3/NDCG@3.

Implemented directly from the definitions in EVALUATION_METHODOLOGY.md Part A
rather than pulled from a library, so the exact grading semantics are auditable.

GRADED RELEVANCE
----------------
``grade_chunk`` returns 0-3 for a retrieved chunk against a labelled question:

    3  chunk contains the verbatim gold span      -> answer-bearing
    2  chunk is in the gold doc at a gold locator -> right place, span degraded
    1  chunk is in the gold doc, elsewhere        -> right document, wrong place
    0  chunk is from a different document         -> irrelevant

BINARY THRESHOLD (a deliberate design decision)
-----------------------------------------------
R@1, R@3 and MRR@10 count a chunk as relevant at ``grade >= 2``, i.e. "the
retriever landed on the right page/concept", *not* at grade 3.

Why: a parser can mangle a word inside an otherwise perfect page (pypdf renders
"those" as "t hose" on contract page 4). Requiring grade 3 would score that page
as a total retrieval miss, which overstates the damage — the page is still the
right page and still answers the question. Instead that defect costs NDCG@3
(graded) while leaving R@3 (binary) intact, so parser damage shows up
proportionately in exactly one place instead of being double-counted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from .corpus import normalise_for_match

RELEVANT_GRADE = 2  # binary-relevance threshold, see module docstring
MAX_GRADE = 3


class Retrieved(Protocol):
    """Minimal shape a retrieved chunk must expose to be scored."""

    doc_id: str
    locators: Sequence[str]
    text: str


class Labelled(Protocol):
    gold_doc: str
    gold_locators: Sequence[str]
    gold_span: str


def grade_chunk(chunk: Retrieved, question: Labelled) -> int:
    """Graded relevance in [0, 3]. See module docstring for the scale."""
    if chunk.doc_id != question.gold_doc:
        return 0
    if normalise_for_match(question.gold_span) in normalise_for_match(chunk.text):
        return 3
    if set(chunk.locators) & set(question.gold_locators):
        return 2
    return 1


def grade_ranking(chunks: Sequence[Retrieved], question: Labelled) -> list[int]:
    return [grade_chunk(c, question) for c in chunks]


# ── Per-query metrics (all take an already-graded ranking) ───────────────────
def recall_at_k(grades: Sequence[int], k: int) -> float:
    """1.0 if any relevant chunk appears in the top k, else 0.0.

    This follows the methodology's definition ("did the correct chunk appear in
    the top k"), which is a hit-rate rather than textbook set recall.
    """
    return float(any(g >= RELEVANT_GRADE for g in grades[:k]))


def reciprocal_rank(grades: Sequence[int], k: int = 10) -> float:
    for i, g in enumerate(grades[:k], start=1):
        if g >= RELEVANT_GRADE:
            return 1.0 / i
    return 0.0


def dcg_at_k(grades: Sequence[int], k: int = 3) -> float:
    """DCG@k = sum(rel_i / log2(i + 1)) — the formula named in the methodology."""
    return sum(g / math.log2(i + 1) for i, g in enumerate(grades[:k], start=1))


def ndcg_at_k(grades: Sequence[int], k: int = 3) -> float:
    """DCG@k normalised by the best achievable ordering of the same grades.

    The ideal ranking is built from the grades actually retrieved, so NDCG asks
    "given what you found, did you order it well?". A query whose gold chunk was
    never retrieved scores 0 because there is nothing positive to order.
    """
    ideal = dcg_at_k(sorted(grades, reverse=True), k)
    return dcg_at_k(grades, k) / ideal if ideal > 0 else 0.0


# ── Aggregation ──────────────────────────────────────────────────────────────
@dataclass
class RetrievalScores:
    n: int
    r_at_1: float
    r_at_3: float
    mrr_at_10: float
    dcg_at_3: float
    ndcg_at_3: float
    route_accuracy: float | None = None

    def as_row(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "R@1": round(self.r_at_1, 4),
            "R@3": round(self.r_at_3, 4),
            "MRR@10": round(self.mrr_at_10, 4),
            "DCG@3": round(self.dcg_at_3, 4),
            "NDCG@3": round(self.ndcg_at_3, 4),
            "Route acc": None if self.route_accuracy is None else round(self.route_accuracy, 4),
        }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def score_run(
    graded_rankings: Sequence[Sequence[int]],
    predicted_routes: Sequence[str] | None = None,
    gold_routes: Sequence[str] | None = None,
) -> RetrievalScores:
    """Aggregate per-query graded rankings into the reported metric set."""
    route_acc = None
    if predicted_routes is not None and gold_routes is not None:
        if len(predicted_routes) != len(gold_routes):
            raise ValueError("predicted_routes and gold_routes must be the same length")
        route_acc = _mean(float(p == g) for p, g in zip(predicted_routes, gold_routes))

    return RetrievalScores(
        n=len(graded_rankings),
        r_at_1=_mean(recall_at_k(g, 1) for g in graded_rankings),
        r_at_3=_mean(recall_at_k(g, 3) for g in graded_rankings),
        mrr_at_10=_mean(reciprocal_rank(g, 10) for g in graded_rankings),
        dcg_at_3=_mean(dcg_at_k(g, 3) for g in graded_rankings),
        ndcg_at_3=_mean(ndcg_at_k(g, 3) for g in graded_rankings),
        route_accuracy=route_acc,
    )


def score_by_facet(
    graded_rankings: Sequence[Sequence[int]],
    facets: Sequence[str],
) -> dict[str, RetrievalScores]:
    """Break a run out by a per-query label (``route`` or ``query_type``).

    Stage 5 depends on this: the methodology asks for R@3 split by keyword vs
    semantic queries, and this group's REQUIREMENT additionally asks for the
    split by which document the query should hit.
    """
    if len(graded_rankings) != len(facets):
        raise ValueError("graded_rankings and facets must be the same length")
    buckets: dict[str, list[Sequence[int]]] = {}
    for grades, facet in zip(graded_rankings, facets):
        buckets.setdefault(facet, []).append(grades)
    return {facet: score_run(rankings) for facet, rankings in sorted(buckets.items())}
