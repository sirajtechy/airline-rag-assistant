"""Unified corpus layer.

Both business-line document types collapse into one shape, :class:`SourceUnit`,
addressed by a *locator* string:

    PDF page          -> "p17"
    XBRL concept      -> "us-gaap:NetIncomeLoss"

The locator is the join key between retrieved chunks and the ground-truth labels
in ``eval/eval_set.yaml``. Crucially it is stable across every parser and every
chunking strategy, which is what lets Stage 1 and Stage 2 be scored against the
same labels.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .config import SOURCE_DOCS
from .parsing import parse_pdf
from .xbrl import parse_xbrl


@dataclass
class SourceUnit:
    """The smallest independently-citable piece of a source document."""

    doc_id: str
    business_line: str
    locator: str
    text: str
    title: str
    meta: dict = field(default_factory=dict)


def normalise_for_match(text: str) -> str:
    """Aggressive normalisation used only for gold-span containment checks.

    Unicode punctuation is folded to ASCII because PDF extraction emits curly
    quotes and en-dashes that would otherwise cause spurious span misses.
    """
    text = unicodedata.normalize("NFKD", text)
    text = (
        text.replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2013", "-").replace("\u2014", "-")
        .replace("\u00a0", " ")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_corpus(parser: str = "pymupdf") -> list[SourceUnit]:
    """Load every source document as a flat list of locator-addressed units."""
    units: list[SourceUnit] = []
    for doc_id, spec in SOURCE_DOCS.items():
        if spec["kind"] == "pdf":
            for page in parse_pdf(spec["path"], doc_id, spec["business_line"], parser):
                units.append(
                    SourceUnit(
                        doc_id=doc_id,
                        business_line=spec["business_line"],
                        locator=f"p{page.page}",
                        text=page.text,
                        title=spec["title"],
                        meta={"page": page.page, "parser": parser},
                    )
                )
        elif spec["kind"] == "xbrl":
            for rec in parse_xbrl(spec["path"], spec["labels_path"]):
                units.append(
                    SourceUnit(
                        doc_id=doc_id,
                        business_line=spec["business_line"],
                        locator=rec.locator,
                        text=rec.text,
                        title=spec["title"],
                        meta={
                            "concept": rec.concept,
                            "label": rec.label,
                            "headline": rec.headline,
                            "fact_count": rec.fact_count,
                        },
                    )
                )
        else:
            raise ValueError(f"unhandled source kind {spec['kind']!r} for {doc_id}")
    return units


def citation_for(unit: SourceUnit) -> str:
    """Human-facing citation string shown next to generated answers."""
    if unit.locator.startswith("p") and unit.locator[1:].isdigit():
        return f"{unit.title}, page {unit.locator[1:]}"
    return f"{unit.title}, {unit.meta.get('label', unit.locator)}"
