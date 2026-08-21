"""The financial business line — turning raw SEC XBRL into retrievable prose.

An XBRL instance is a bag of ~1.4k machine-readable facts, each pointing at a
context (period + optional dimensional breakdown) and a unit. None of that is
useful to an embedding model as-is.

This module resolves facts against the companion label linkbase and renders one
readable record *per concept*, with every reported context listed underneath it.
Concept-level grouping is deliberate: one chunk per raw fact would produce
thousands of near-identical fragments that shred dense retrieval, while one chunk
per statement would be too coarse to cite precisely.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

XBRLI = "{http://www.xbrl.org/2003/instance}"
XBRLDI = "{http://xbrl.org/2006/xbrldi}"
XLINK = "{http://www.w3.org/1999/xlink}"
LINKBASE = "{http://www.xbrl.org/2003/linkbase}"

STANDARD_LABEL_ROLE = "http://www.xbrl.org/2003/role/label"

# Concepts a customer-support/finance question is realistically about. Used only
# to mark records as "headline" so the report can show the corpus is sensible.
HEADLINE_CONCEPTS = {
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "CostsAndExpenses",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
}

# Vocabulary bridge between how a US-GAAP concept is *named* and how a financial
# statement *labels* the same line. "Revenue from Contract with Customer,
# Excluding Assessed Tax" is the XBRL element name for what every 10-K income
# statement calls total operating revenue, and no retriever can be expected to
# make that leap unaided.
#
# These are standard line-item names taken from 10-K statement conventions, not
# from this project's evaluation questions. Their contribution is measured
# separately from the structural fixes (see reports/results/stage0_xbrl.md) so
# the two effects can be judged independently.
CONCEPT_ALIASES: dict[str, str] = {
    "RevenueFromContractWithCustomerExcludingAssessedTax":
        "total revenue, total operating revenue, operating revenue, revenues, sales, top line",
    "CostsAndExpenses":
        "total operating expense, total costs and expenses, operating expenses",
    "OperatingIncomeLoss":
        "operating income, operating profit, income from operations",
    "NetIncomeLoss":
        "net income, net profit, net earnings, bottom line",
    "EarningsPerShareDiluted": "diluted EPS, diluted earnings per share",
    "EarningsPerShareBasic": "basic EPS, basic earnings per share",
    "Assets": "total assets, balance sheet total",
    "Liabilities": "total liabilities",
    "StockholdersEquity":
        "total stockholders equity, shareholders equity, shareowners equity, book value",
    "CashAndCashEquivalentsAtCarryingValue":
        "cash, cash and cash equivalents, cash on hand, liquidity",
    "LongTermDebtNoncurrent": "long term debt, non-current debt",
    "OperatingLeaseLiabilityNoncurrent": "operating lease liabilities",
}

# Prose blobs rather than reported facts. They are long, boilerplate-heavy and
# act purely as retrieval distractors for a customer-support corpus.
_NON_FACT_MARKERS = ("TextBlock", "PolicyText", "Axis", "Domain")


@dataclass
class FinancialRecord:
    """One concept's worth of reported facts, rendered as retrievable text."""

    concept: str          # e.g. "RevenueFromContractWithCustomerExcludingAssessedTax"
    prefix: str           # e.g. "us-gaap"
    label: str            # human-readable standard label
    text: str
    fact_count: int
    headline: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def locator(self) -> str:
        return f"{self.prefix}:{self.concept}"


# ── Label linkbase ────────────────────────────────────────────────────────────
def load_labels(labels_path: Path) -> dict[str, str]:
    """Map ``prefix_ConceptName`` -> standard human-readable label."""
    from lxml import etree

    root = etree.parse(str(labels_path)).getroot()

    # loc: xlink:label -> concept key taken from the href fragment
    loc_to_concept: dict[str, str] = {}
    for loc in root.iter(f"{LINKBASE}loc"):
        href = loc.get(f"{XLINK}href") or ""
        if "#" in href:
            loc_to_concept[loc.get(f"{XLINK}label")] = href.split("#", 1)[1]

    # label: xlink:label -> text, preferring the standard label role
    lab_to_text: dict[str, str] = {}
    for lab in root.iter(f"{LINKBASE}label"):
        key = lab.get(f"{XLINK}label")
        if lab.get(f"{XLINK}role") == STANDARD_LABEL_ROLE or key not in lab_to_text:
            lab_to_text[key] = (lab.text or "").strip()

    out: dict[str, str] = {}
    for arc in root.iter(f"{LINKBASE}labelArc"):
        concept = loc_to_concept.get(arc.get(f"{XLINK}from"))
        text = lab_to_text.get(arc.get(f"{XLINK}to"))
        if concept and text:
            out[concept] = text
    return out


# ── Context resolution ────────────────────────────────────────────────────────
def _describe_period(context, legacy: bool = False) -> str:
    """Compact period label.

    Full calendar years collapse to "FY2025". The verbose
    "for the period 2025-01-01 to 2025-12-31" form repeated across hundreds of
    records was shared vocabulary that swamped each record's distinguishing
    signal for both BM25 and the embedding models. ``legacy=True`` restores that
    original phrasing so the cost of it can be measured rather than asserted.
    """
    period = context.find(f"{XBRLI}period")
    if period is None:
        return "period not stated"
    instant = period.find(f"{XBRLI}instant")
    if instant is not None:
        return f"as of {instant.text}"
    start = period.find(f"{XBRLI}startDate")
    end = period.find(f"{XBRLI}endDate")
    if start is not None and end is not None:
        s, e = start.text, end.text
        if legacy:
            return f"for the period {s} to {e}"
        if s.endswith("-01-01") and e.endswith("-12-31") and s[:4] == e[:4]:
            return f"FY{s[:4]}"
        return f"{s} to {e}"
    return "period not stated"


def _describe_dimensions(context, labels: dict[str, str]) -> str:
    parts = []
    for member in context.iter(f"{XBRLDI}explicitMember"):
        raw = (member.text or "").strip()
        key = raw.replace(":", "_")
        pretty = labels.get(key, raw.split(":")[-1])
        pretty = re.sub(r"\s*\[Member\]$", "", pretty)
        parts.append(pretty)
    return ", ".join(parts)


def _format_value(text: str, unit: str | None) -> str:
    raw = (text or "").strip()
    try:
        num = float(raw)
    except ValueError:
        return raw
    if unit and "usd" in unit.lower():
        if abs(num) >= 1_000_000_000:
            return f"USD {num:,.0f} ({num / 1_000_000_000:.2f} billion)"
        if abs(num) >= 1_000_000:
            return f"USD {num:,.0f} ({num / 1_000_000:.1f} million)"
        return f"USD {num:,.2f}"
    if unit and "shares" in unit.lower():
        return f"{num:,.0f} shares"
    if num.is_integer():
        return f"{num:,.0f}"
    return f"{num:,.2f}"


def parse_xbrl(
    data_path: Path,
    labels_path: Path,
    entity: str = "Delta Air Lines, Inc.",
    fiscal_year: str = "FY2025",
    drop_non_facts: bool = True,
    use_aliases: bool = True,
    legacy_rendering: bool = False,
) -> list[FinancialRecord]:
    """Render the XBRL instance as one text record per reported concept.

    ``drop_non_facts`` removes TextBlock/policy prose that acts as retrieval
    noise; ``use_aliases`` adds the statement-label vocabulary bridge;
    ``legacy_rendering`` restores the original verbose boilerplate. All three are
    switchable so each contribution is measured rather than assumed.
    """
    from lxml import etree

    labels = load_labels(Path(labels_path))
    root = etree.parse(str(data_path)).getroot()

    contexts = {c.get("id"): c for c in root.findall(f"{XBRLI}context")}
    units: dict[str, str] = {}
    for unit in root.findall(f"{XBRLI}unit"):
        measure = next((m.text for m in unit.iter(f"{XBRLI}measure")), "")
        units[unit.get("id")] = (measure or "").split(":")[-1]

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen: set[tuple] = set()

    for el in root:
        qname = etree.QName(el)
        namespace, local = qname.namespace or "", qname.localname
        if local in {"context", "unit"} or not el.get("contextRef"):
            continue
        prefix = (
            "us-gaap" if "us-gaap" in namespace
            else "dei" if "/dei/" in namespace
            else "dal" if "delta.com" in namespace
            else "srt" if "/srt/" in namespace
            else namespace.rstrip("/").split("/")[-1]
        )
        if drop_non_facts and any(marker in local for marker in _NON_FACT_MARKERS):
            continue
        context = contexts.get(el.get("contextRef"))
        if context is None:
            continue

        value = _format_value(el.text or "", units.get(el.get("unitRef") or ""))
        if not value or len(value) > 400:  # skip embedded HTML text blocks
            continue
        period = _describe_period(context, legacy=legacy_rendering)
        dims = _describe_dimensions(context, labels)
        scope = dims or ("Consolidated (no segment breakdown)" if legacy_rendering
                         else "Consolidated")

        line = f"  - {scope}, {period}: {value}"
        key = (prefix, local)
        if (key, line) in seen:
            continue
        seen.add((key, line))
        grouped[key].append(line)

    records: list[FinancialRecord] = []
    for (prefix, concept), lines in sorted(grouped.items()):
        label = labels.get(f"{prefix}_{concept}", concept)

        # Lead with the distinguishing label rather than the shared boilerplate:
        # the first tokens carry the most retrieval weight, and every record used
        # to open with the same entity/filing sentence.
        alias = CONCEPT_ALIASES.get(concept) if use_aliases else None
        if legacy_rendering:
            header = [
                f"{entity} — {fiscal_year} Form 10-K financial data (SEC XBRL).",
                f"Financial concept: {label} ({prefix}:{concept}).",
                "Reported values:",
            ]
        else:
            header = [f"{label} — {entity} {fiscal_year} Form 10-K."]
            if alias:
                header.append(f"Also known as: {alias}.")
            header.append(f"XBRL concept {prefix}:{concept}.")

        records.append(
            FinancialRecord(
                concept=concept,
                prefix=prefix,
                label=label,
                text="\n".join(header) + "\n" + "\n".join(lines),
                fact_count=len(lines),
                headline=concept in HEADLINE_CONCEPTS,
                meta={"entity": entity, "fiscal_year": fiscal_year,
                      "aliased": bool(alias)},
            )
        )
    return records
