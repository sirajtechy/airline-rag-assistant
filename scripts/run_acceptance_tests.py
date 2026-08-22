"""The graded acceptance tests (30% of the grade).

Runs the 5 required tests from REQUIREMENT.md section 6, plus 4 extra guardrail
probes and 2 cross-route disambiguation cases, against the final pipeline.

A test passes only if the *behaviour* matches — answering when it should answer,
refusing when it should refuse — and, for answers, the route and source document
are correct. Producing plausible prose while quoting the wrong tariff is a
failure, not a partial pass.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_rag.ablation import save_stage
from delta_rag.config import EVAL_DIR, RESULTS_DIR
from delta_rag.pipeline import DeltaSupportBot

# A refusal and a deflection are both "blocked", but they are different promises
# to the customer, so they are distinguished by the guardrail reason.
REFUSE_REASONS = ("out_of_scope_carrier", "no_hits", "weak_evidence")
DEFLECT_REASONS = ("requires_live_data",)

# Phrases that count as the model declining in-answer rather than the guardrail
# blocking up front. For a question whose answer genuinely is not in the corpus,
# a grounded "the documents don't cover this" *with a citation* is a better
# outcome than a blanket guardrail refusal, so it must be scored as a pass.
_DISCLAIMER_MARKERS = (
    "do not cover", "does not cover", "doesn't cover", "don't cover",
    "not covered", "cannot provide", "can't provide", "no information",
    "not specified", "not stated", "domestic travel only", "domestic only",
    "not available in", "does not contain", "do not contain", "unable to provide",
)


def evaluate(test: dict, response) -> tuple[bool, str]:
    want = test["behaviour"]
    answer_lower = (response.answer or "").lower()
    reason = response.reason or ""

    if want == "answer":
        if response.blocked:
            return False, f"expected an answer but was blocked ({reason})"
        if test.get("expect_route") and response.route != test["expect_route"]:
            return False, f"routed to {response.route}, expected {test['expect_route']}"
        if test.get("expect_doc"):
            docs = {s["doc_id"] for s in response.sources}
            if test["expect_doc"] not in docs:
                return False, f"cited {sorted(docs)}, expected {test['expect_doc']}"
    elif want == "refuse":
        if not response.blocked:
            return False, "answered instead of refusing"
        if not reason.startswith(REFUSE_REASONS):
            return False, f"blocked for the wrong reason ({reason})"
    elif want == "refuse_or_disclaim":
        # Passes either way: guardrail block, or a grounded in-answer refusal.
        if response.blocked:
            if not reason.startswith(REFUSE_REASONS):
                return False, f"blocked for the wrong reason ({reason})"
        elif not any(m in answer_lower for m in _DISCLAIMER_MARKERS):
            return False, "answered without declining or disclaiming"
    elif want == "deflect":
        if not response.blocked:
            return False, "answered instead of explaining the limitation"
        if not reason.startswith(DEFLECT_REASONS):
            return False, f"blocked for the wrong reason ({reason})"
    else:
        return False, f"unknown expected behaviour {want!r}"

    for needle in test.get("must_include") or []:
        if needle.lower() not in answer_lower:
            return False, f"answer omits required text {needle!r}"
    for needle in test.get("must_not_include") or []:
        # Whole-word matching: a substring test for the carrier "United" fires on
        # "United States", which appears legitimately throughout both tariffs.
        if re.search(rf"\b{re.escape(needle.lower())}\b", answer_lower):
            return False, f"answer contains forbidden text {needle!r}"
    return True, "pass"


def main() -> None:
    spec = yaml.safe_load((EVAL_DIR / "acceptance_tests.yaml").read_text())
    bot = DeltaSupportBot()
    print(f"pipeline: {bot.config.label()} | generator: {bot.model}\n")

    rows, details = [], []
    for test in spec["tests"]:
        response = bot.answer(test["question"])
        ok, detail = evaluate(test, response)
        rows.append({
            "#": test["id"],
            "Question": test["question"],
            "Expected": test["behaviour"],
            "Route": response.route or "-",
            "Blocked": "yes" if response.blocked else "no",
            "Result": "PASS" if ok else "FAIL",
            "Detail": detail if not ok else "",
        })
        details.append({**response.to_dict(), "test_id": test["id"], "passed": ok,
                        "detail": detail})
        print(f"  {test['id']:<3} {'PASS' if ok else 'FAIL'}  {test['question'][:58]:<60} "
              f"route={response.route} {detail if not ok else ''}")

    required = [r for r in rows if r["#"].startswith("A")]
    passed_req = sum(r["Result"] == "PASS" for r in required)
    passed_all = sum(r["Result"] == "PASS" for r in rows)

    notes = (
        f"**Required acceptance tests (REQUIREMENT section 6): "
        f"{passed_req}/{len(required)} passed.**\n"
        f"All probes including extra guardrail and routing cases: "
        f"{passed_all}/{len(rows)}.\n\n"
        "A1-A5 are the five graded tests. G1-G4 are additional guardrail probes: G1 "
        "names a competitor using in-corpus vocabulary (\"denied boarding\") to check the "
        "carrier check is not fooled by familiar terminology, G2 is a transactional "
        "request, and G3-G4 ask for fees that genuinely do not exist in these "
        "domestic-only documents — the case where a RAG bot is most tempted to invent a "
        "number.\n\n"
        "R1 and R2 are the cross-route cases this group is specifically graded on. R1 "
        "(\"revenue from shipping cargo\") is lexically saturated with cargo terms but "
        "must route to the financial XBRL, and R2 (\"liability limit for a damaged "
        "shipment\") shares the word \"liability\" with the passenger contract but must "
        "route to cargo. Both are why routing is decided from rank-weighted retrieval "
        "evidence rather than from keyword matching on the question."
    )
    save_stage(
        "acceptance_tests", "Acceptance test results", rows,
        columns=["#", "Question", "Expected", "Route", "Blocked", "Result", "Detail"],
        notes=notes,
    )
    (RESULTS_DIR / "acceptance_details.json").write_text(
        json.dumps(details, indent=2, default=str)
    )
    print(f"\nrequired {passed_req}/{len(required)} | all {passed_all}/{len(rows)}")


if __name__ == "__main__":
    main()
