"""Business-line routing.

The graded skill for this group is sending a passenger question to the Contract
of Carriage and a shipping question to the Cargo tariff without
cross-contamination.

Routing is done **from retrieval evidence**, not from a keyword classifier on the
question. A keyword router is exactly what fails the hard cases: "How much
revenue did Delta make from shipping cargo last year?" is lexically saturated
with cargo terms but is a financial question, and "What is Delta's liability
limit for a damaged shipment?" shares "liability" with the passenger contract.
Letting hybrid retrieval vote with rank-weighted evidence gets both right, and it
degrades gracefully instead of falling off a keyword cliff.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .config import SOURCE_DOCS
from .stores import SearchHit

# Carriers that are explicitly out of scope. The corpus is Delta-only, so any
# question about these must be declined rather than answered from Delta's rules.
OTHER_CARRIERS = {
    "jetblue": "JetBlue",
    "united": "United Airlines",
    "american airlines": "American Airlines",
    "southwest": "Southwest Airlines",
    "spirit": "Spirit Airlines",
    "frontier": "Frontier Airlines",
    "alaska airlines": "Alaska Airlines",
    "cape air": "Cape Air",
    "hawaiian": "Hawaiian Airlines",
    "allegiant": "Allegiant Air",
    "lufthansa": "Lufthansa",
    "emirates": "Emirates",
    "british airways": "British Airways",
    "air canada": "Air Canada",
    "ryanair": "Ryanair",
    "easyjet": "easyJet",
}

# Requests that need live account/booking access rather than published policy.
_LIVE_DATA_PATTERNS = [
    r"\bmy (flight|booking|reservation|ticket|bag|baggage|shipment|claim|refund|account)\b",
    r"\bstatus of (my|the) (flight|shipment|refund|claim)\b",
    r"\b(is|was|will) my \w+",
    r"\bflight status\b",
    r"\b(cancel|rebook|change|refund|book) my\b",
    r"\bwhere is my\b",
    r"\btrack (my|this) (shipment|package|bag)\b",
    r"\bconfirmation (number|code)\b",
    r"\bcheck me in\b",
]
_LIVE_DATA_RE = re.compile("|".join(_LIVE_DATA_PATTERNS), re.I)


@dataclass
class RouteDecision:
    business_line: str | None
    confidence: float
    votes: dict[str, float]
    top_score: float
    term_prior: dict[str, float] = None  # type: ignore[assignment]


# ── Document-declared vocabulary ─────────────────────────────────────────────
# Both tariffs open with a definitions section — Rule 3 in the passenger contract,
# G2 in the cargo tariff — where each document declares the terms it governs.
# That is a far better source of routing vocabulary than a hand-written keyword
# list, because the documents themselves decide what counts as their domain
# language, and it is extracted rather than invented.
_DEFINITION_RE = re.compile(
    r"^([A-Z][A-Za-z][A-Za-z /'\-]{1,38}?)\s*(?:shall mean|means|shall be|[-\u2013]\s)",
    re.M,
)
# Terms too generic to disambiguate even though both documents define them.
_STOP_TERMS = {"carrier", "days", "delta", "person", "property", "us", "u s",
               "united states of america", "currency", "numbers", "group"}


def extract_defined_terms() -> dict[str, str]:
    """Map a defined term to the business line that uniquely defines it.

    Terms defined by more than one document (e.g. "Carrier") are dropped, since
    they carry no routing signal.
    """
    from .corpus import load_corpus

    by_line: dict[str, set[str]] = {}
    for unit in load_corpus():
        if unit.business_line == FINANCIAL_LINE:
            continue
        for match in _DEFINITION_RE.finditer(unit.text):
            term = " ".join(match.group(1).split()).strip().lower()
            if len(term) < 4 or term in _STOP_TERMS:
                continue
            by_line.setdefault(unit.business_line, set()).add(term)

    counts: dict[str, list[str]] = {}
    for line, terms in by_line.items():
        for term in terms:
            counts.setdefault(term, []).append(line)
    return {t: lines[0] for t, lines in counts.items() if len(lines) == 1}


FINANCIAL_LINE = "financial"
_DEFINED_TERMS: dict[str, str] | None = None


def defined_terms() -> dict[str, str]:
    global _DEFINED_TERMS
    if _DEFINED_TERMS is None:
        _DEFINED_TERMS = extract_defined_terms()
    return _DEFINED_TERMS


def term_prior(question: str) -> dict[str, float]:
    """Business-line evidence from the documents' own defined vocabulary."""
    lowered = question.lower()
    prior: dict[str, float] = {}
    for term, line in defined_terms().items():
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            # Longer terms are more specific, so they count for more.
            prior[line] = prior.get(line, 0.0) + (1.0 + 0.4 * term.count(" "))
    return prior


def detect_other_carrier(question: str) -> str | None:
    """Return a competitor's display name if the question is about them."""
    lowered = question.lower()
    for needle, display in OTHER_CARRIERS.items():
        if re.search(rf"\b{re.escape(needle)}\b", lowered):
            return display
    return None


def needs_live_data(question: str) -> bool:
    """True when the question needs account access rather than published policy."""
    return bool(_LIVE_DATA_RE.search(question))


def route_from_hits(
    hits: Sequence[SearchHit],
    top_n: int = 5,
    question: str | None = None,
    prior_weight: float = 0.35,
) -> RouteDecision:
    """Rank-weighted vote over retrieved chunks, plus a vocabulary prior.

    Weighting by 1/rank means the strongest evidence dominates while lower hits
    still get a say, so a single stray top-1 result cannot hijack the route.

    When ``question`` is supplied, the documents' own defined vocabulary adds a
    prior. This exists because retrieval evidence alone confuses genuinely
    ambiguous cases: "What is Delta's liability limit for a damaged shipment?"
    retrieves the passenger contract's baggage-liability text, which is the most
    topically similar passage in the corpus, while "shipment" is a term the cargo
    tariff explicitly defines and the passenger contract does not. The prior is
    weighted below the retrieval evidence so it nudges ties rather than
    overriding what was actually found.
    """
    if not hits:
        return RouteDecision(None, 0.0, {}, 0.0, {})

    votes: dict[str, float] = {}
    for rank, hit in enumerate(hits[:top_n], start=1):
        line = hit.chunk.business_line
        votes[line] = votes.get(line, 0.0) + 1.0 / rank

    prior = term_prior(question) if question else {}
    if prior:
        scale = prior_weight * sum(votes.values()) / max(sum(prior.values()), 1e-9)
        for line, weight in prior.items():
            votes[line] = votes.get(line, 0.0) + weight * scale

    total = sum(votes.values())
    winner = max(votes, key=votes.get)
    return RouteDecision(
        business_line=winner,
        confidence=votes[winner] / total if total else 0.0,
        votes=votes,
        top_score=float(hits[0].score),
        term_prior=prior,
    )


def doc_to_business_line(doc_id: str) -> str | None:
    spec = SOURCE_DOCS.get(doc_id)
    return spec["business_line"] if spec else None
