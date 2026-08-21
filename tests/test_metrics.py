"""Unit tests for the retrieval metrics, checked against hand-worked values.

Every ablation table in the report is produced by this module, so the arithmetic
is pinned against numbers computed by hand from the definitions in
EVALUATION_METHODOLOGY.md Part A.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from delta_rag.metrics import (
    RELEVANT_GRADE,
    dcg_at_k,
    grade_chunk,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_by_facet,
    score_run,
)


@dataclass
class FakeChunk:
    doc_id: str
    locators: list[str]
    text: str


@dataclass
class FakeQuestion:
    gold_doc: str = "contract_of_carriage"
    gold_locators: list[str] = field(default_factory=lambda: ["p17"])
    gold_span: str = "sell more tickets"


# ── grade_chunk ──────────────────────────────────────────────────────────────
def test_grade_3_when_chunk_contains_gold_span():
    chunk = FakeChunk("contract_of_carriage", ["p17"], "Delta may sell more tickets than seats.")
    assert grade_chunk(chunk, FakeQuestion()) == 3


def test_grade_3_survives_whitespace_and_curly_quote_noise():
    """PDF extraction emits curly quotes and ragged newlines; grading must not care."""
    chunk = FakeChunk("contract_of_carriage", ["p17"], "Delta\u2019s policy: SELL   MORE\nTICKETS now")
    q = FakeQuestion(gold_span="sell more tickets")
    assert grade_chunk(chunk, q) == 3


def test_grade_2_right_locator_but_span_absent():
    chunk = FakeChunk("contract_of_carriage", ["p17"], "Unrelated text on the correct page.")
    assert grade_chunk(chunk, FakeQuestion()) == 2


def test_grade_1_right_document_wrong_locator():
    chunk = FakeChunk("contract_of_carriage", ["p3"], "Definitions section.")
    assert grade_chunk(chunk, FakeQuestion()) == 1


def test_grade_0_wrong_document():
    chunk = FakeChunk("cargo_tariff", ["p17"], "Delta may sell more tickets than seats.")
    assert grade_chunk(chunk, FakeQuestion()) == 0


def test_multi_locator_chunk_matches_any_gold_locator():
    """Chunks that span a page boundary carry several locators."""
    chunk = FakeChunk("contract_of_carriage", ["p16", "p17"], "no span here")
    assert grade_chunk(chunk, FakeQuestion(gold_locators=["p17"])) == 2


# ── recall / MRR ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "grades,k,expected",
    [
        ([3, 0, 0], 1, 1.0),
        ([1, 0, 0], 1, 0.0),   # grade 1 is below the relevance threshold
        ([2, 0, 0], 1, 1.0),   # grade 2 is at the threshold
        ([0, 0, 3], 3, 1.0),
        ([0, 0, 0, 3], 3, 0.0),  # relevant chunk sits outside the cutoff
    ],
)
def test_recall_at_k(grades, k, expected):
    assert recall_at_k(grades, k) == expected


def test_relevance_threshold_is_grade_two():
    """Pins the documented design decision so it cannot drift silently."""
    assert RELEVANT_GRADE == 2


@pytest.mark.parametrize(
    "grades,expected",
    [
        ([3, 0, 0], 1.0),
        ([0, 3, 0], 0.5),
        ([0, 0, 2], 1 / 3),
        ([0] * 10, 0.0),
        ([1, 1, 1, 3], 0.25),   # grade-1 chunks are skipped, first relevant is rank 4
    ],
)
def test_reciprocal_rank(grades, expected):
    assert reciprocal_rank(grades, 10) == pytest.approx(expected)


def test_reciprocal_rank_ignores_hits_beyond_k():
    grades = [0] * 10 + [3]
    assert reciprocal_rank(grades, 10) == 0.0


# ── DCG / NDCG ───────────────────────────────────────────────────────────────
def test_dcg_matches_hand_computation():
    # DCG@3 = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.26186 + 0.5
    expected = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert dcg_at_k([3, 2, 1], 3) == pytest.approx(expected)
    assert dcg_at_k([3, 2, 1], 3) == pytest.approx(4.76186, abs=1e-5)


def test_ndcg_is_one_for_perfectly_ordered_results():
    assert ndcg_at_k([3, 2, 1], 3) == pytest.approx(1.0)


def test_ndcg_penalises_bad_ordering():
    """Same grades, worst order — must score strictly below the ideal ordering."""
    assert ndcg_at_k([1, 2, 3], 3) == pytest.approx(dcg_at_k([1, 2, 3], 3) / dcg_at_k([3, 2, 1], 3))
    assert ndcg_at_k([1, 2, 3], 3) < ndcg_at_k([3, 2, 1], 3)


def test_ndcg_is_zero_when_nothing_relevant_retrieved():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_bounded_in_unit_interval():
    for grades in ([3, 3, 3], [0, 1, 2], [2, 0, 3], [1, 1, 1]):
        assert 0.0 <= ndcg_at_k(grades, 3) <= 1.0


# ── aggregation ──────────────────────────────────────────────────────────────
def test_score_run_averages_across_queries():
    scores = score_run([[3, 0, 0], [0, 0, 0], [0, 2, 0]])
    assert scores.n == 3
    assert scores.r_at_1 == pytest.approx(1 / 3)
    assert scores.r_at_3 == pytest.approx(2 / 3)
    assert scores.mrr_at_10 == pytest.approx((1.0 + 0.0 + 0.5) / 3)


def test_route_accuracy_tracked_when_routes_supplied():
    scores = score_run(
        [[3], [3], [3], [3]],
        predicted_routes=["passenger", "cargo", "financial", "cargo"],
        gold_routes=["passenger", "cargo", "financial", "financial"],
    )
    assert scores.route_accuracy == pytest.approx(0.75)


def test_route_accuracy_none_when_not_supplied():
    assert score_run([[3]]).route_accuracy is None


def test_score_run_rejects_mismatched_route_lengths():
    with pytest.raises(ValueError):
        score_run([[3], [3]], predicted_routes=["passenger"], gold_routes=["passenger", "cargo"])


def test_score_by_facet_splits_correctly():
    """Stage 5 depends on this split being exact."""
    by_facet = score_by_facet([[3, 0, 0], [0, 0, 0], [3, 0, 0]], ["keyword", "keyword", "semantic"])
    assert set(by_facet) == {"keyword", "semantic"}
    assert by_facet["keyword"].n == 2
    assert by_facet["keyword"].r_at_1 == pytest.approx(0.5)
    assert by_facet["semantic"].r_at_1 == pytest.approx(1.0)


def test_score_by_facet_rejects_length_mismatch():
    with pytest.raises(ValueError):
        score_by_facet([[3]], ["keyword", "semantic"])


def test_empty_run_does_not_divide_by_zero():
    scores = score_run([])
    assert scores.n == 0 and scores.r_at_1 == 0.0 and scores.mrr_at_10 == 0.0
