"""Guardrails — the three refusals REQUIREMENT section 5 demands.

1. Never invent a refund policy, fee or rule that is not in the documents.
2. Decline questions about other airlines.
3. Explain the limitation for anything needing live booking access.

The first two run *before* retrieval, because they are cheap, deterministic and
must not depend on an LLM deciding to behave. The third is a retrieval-confidence
floor: if the best evidence is weak, the bot says so instead of letting the
generator improvise from loosely-related tariff text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .routing import detect_other_carrier, needs_live_data
from .stores import SearchHit

# Calibrated in scripts/calibrate_guardrail.py against the 30 answerable eval
# questions versus 8 deliberately unanswerable probes. Expressed on the dense
# cosine similarity of the single best-matching chunk, which is bounded [0, 1]
# and comparable across queries — unlike an RRF fusion score, which is a rank
# artefact and cannot support a fixed threshold.
#
# At 0.67 the gate answers 100% of answerable questions and refuses 25% of the
# unanswerable probes. The two populations overlap (answerable bottom out at
# 0.679, unanswerable top out at 0.726), so no threshold separates them cleanly.
#
# It is deliberately tuned to reject no answerable question rather than to
# maximise refusals, because this gate is only the *coarse* layer of a two-layer
# defence. The precise layer is the system prompt: acceptance probe G4 shows the
# model declining correctly ("Delta's published documents do not cover fees for
# checked bags on transatlantic flights ... domestic travel only [1]") on a
# question whose retrieval confidence was high. Cosine similarity measures
# topical closeness, not whether the answer is actually present, so tightening
# this gate buys refusals the prompt already handles while losing real answers.
# See reports/results/guardrail_calibration.md.
ABSTAIN_THRESHOLD = 0.67

REFUSAL_OTHER_CARRIER = (
    "I can only help with Delta Air Lines policies — my knowledge covers Delta's "
    "Domestic Contract of Carriage, Delta Cargo's U.S. domestic shipping tariff, "
    "and Delta's published financial filings. I don't have {carrier}'s policies, "
    "so I can't answer that. For {carrier} you'll need to check their own contract "
    "of carriage or customer service."
)

DEFLECT_LIVE_DATA = (
    "I can't look up live booking, flight status, or account information — I only "
    "answer general policy questions from Delta's published documents. For anything "
    "tied to a specific booking, please use the Fly Delta app, delta.com, or contact "
    "Delta Reservations directly.\n\n"
    "I can still help with the rules themselves, for example what Delta's policy is "
    "on delays and cancellations, or what you're entitled to if you're denied boarding."
)

ABSTAIN_NO_EVIDENCE = (
    "I couldn't find anything in Delta's published documents that answers that, so I "
    "don't want to guess — inventing a fee or rule would be worse than saying nothing.\n\n"
    "My sources cover Delta's U.S. domestic passenger rules (Contract of Carriage), "
    "Delta Cargo's U.S. domestic shipping tariff, and Delta's FY2025 financial "
    "filings. Note these are **domestic** documents, so international fees and routes "
    "aren't covered. Try rephrasing, or ask about one of those areas."
)


@dataclass
class GuardrailVerdict:
    """A pre-emptive answer, or ``None`` to let the pipeline proceed."""

    blocked: bool
    answer: str | None = None
    reason: str | None = None


def check_pre_retrieval(question: str) -> GuardrailVerdict:
    """Deterministic checks that must not depend on model behaviour.

    Carrier scope is tested before live-data intent so that "what's the status of
    my JetBlue flight" is refused as out-of-scope rather than merely deflected.
    """
    carrier = detect_other_carrier(question)
    if carrier:
        return GuardrailVerdict(
            blocked=True,
            answer=REFUSAL_OTHER_CARRIER.format(carrier=carrier),
            reason=f"out_of_scope_carrier:{carrier}",
        )
    if needs_live_data(question):
        return GuardrailVerdict(
            blocked=True, answer=DEFLECT_LIVE_DATA, reason="requires_live_data"
        )
    return GuardrailVerdict(blocked=False)


def check_evidence(
    hits: Sequence[SearchHit],
    dense_scores: Sequence[float] | None = None,
    threshold: float = ABSTAIN_THRESHOLD,
) -> GuardrailVerdict:
    """Abstain when the retrieved evidence is too weak to ground an answer."""
    if not hits:
        return GuardrailVerdict(True, ABSTAIN_NO_EVIDENCE, "no_hits")
    best = max(dense_scores) if dense_scores else None
    if best is not None and best < threshold:
        return GuardrailVerdict(
            True, ABSTAIN_NO_EVIDENCE, f"weak_evidence:{best:.3f}<{threshold}"
        )
    return GuardrailVerdict(blocked=False)


def verify_citations(answer: str, n_sources: int) -> tuple[bool, list[int]]:
    """Check the answer cites sources that were actually supplied.

    A citation pointing at a source that does not exist is a hallucinated
    citation, which is worse than no citation at all because it looks grounded.
    """
    import re

    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})
    invalid = [c for c in cited if c < 1 or c > n_sources]
    return (not invalid and bool(cited)), invalid
