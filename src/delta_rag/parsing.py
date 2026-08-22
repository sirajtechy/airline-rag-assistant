"""Stage 1 — PDF parsing strategies.

Three real alternatives are implemented so the Stage 1 ablation compares measured
output rather than reputation: ``pypdf``, ``pdfplumber``, and ``PyMuPDF`` (fitz).

Every parser returns the same shape — a list of :class:`ParsedPage` — so the rest
of the pipeline is parser-agnostic and the only thing that varies in the Stage 1
experiment is the extraction itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Characters that signal a failed glyph mapping / broken CID decode.
_REPLACEMENT_CHARS = ("\ufffd", "\x00")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")


@dataclass
class ParsedPage:
    """One page of extracted text, tagged with everything needed for citations."""

    doc_id: str
    business_line: str
    page: int  # 1-based, as a customer would cite it
    text: str
    parser: str
    meta: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text.strip())


# ── Individual parsers ────────────────────────────────────────────────────────
def _parse_pypdf(path: Path) -> list[str]:
    from pypdf import PdfReader

    return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


def _parse_pdfplumber(path: Path) -> list[str]:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        return [(p.extract_text() or "") for p in pdf.pages]


def _parse_pymupdf(path: Path) -> list[str]:
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return [page.get_text("text") for page in doc]


PARSERS: dict[str, Callable[[Path], list[str]]] = {
    "pypdf": _parse_pypdf,
    "pdfplumber": _parse_pdfplumber,
    "pymupdf": _parse_pymupdf,
}


def normalise(text: str) -> str:
    """Collapse the whitespace noise PDF extraction leaves behind.

    Kept deliberately conservative: de-hyphenate across line breaks, join hard-
    wrapped lines, and squeeze blank runs. Anything more aggressive would mask
    the very quality differences Stage 1 is trying to measure.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # de-hyphenate
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: Path, doc_id: str, business_line: str, parser: str) -> list[ParsedPage]:
    if parser not in PARSERS:
        raise ValueError(f"unknown parser {parser!r}; expected one of {sorted(PARSERS)}")
    raw_pages = PARSERS[parser](Path(path))
    return [
        ParsedPage(
            doc_id=doc_id,
            business_line=business_line,
            page=i + 1,
            text=normalise(raw),
            parser=parser,
            meta={"raw_char_count": len(raw)},
        )
        for i, raw in enumerate(raw_pages)
    ]


# ── Stage 1 quality measurement ───────────────────────────────────────────────
# ── Front matter and running-header removal ──────────────────────────────────
# A table-of-contents line is a heading followed by dotted leaders and a page
# number: "G32 Limit of Liability .......................... 10".
_TOC_LINE = re.compile(r"\.{4,}\s*\d+\s*$")
# "Page 17 of 23" style footers, and the standalone page numbers around them.
_PAGE_FOOTER = re.compile(r"^\s*(page\s+\d+\s+of\s+\d+|\d{1,3})\s*$", re.I)


def is_front_matter(text: str, toc_ratio: float = 0.30, min_lines: int = 5) -> bool:
    """True for cover and table-of-contents pages.

    These pages are actively harmful to retrieval rather than merely useless:
    a contents page lists every rule title in the document, so it matches almost
    any topical query with high lexical density while containing no answer at all.
    Both tariffs put their contents on pages 1-2, and no gold answer span lives
    there.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < min_lines:
        return True  # cover pages: a handful of title lines and nothing else
    toc_lines = sum(bool(_TOC_LINE.search(ln)) for ln in lines)
    return toc_lines / len(lines) >= toc_ratio


def find_running_lines(pages: list[str], threshold: float = 0.40) -> set[str]:
    """Lines repeated near the top/bottom of many pages, i.e. headers/footers.

    The Contract of Carriage stamps "Delta Domestic General Rules Tariff" on all
    23 pages. Left in place, that phrase is indexed 23 times and makes the
    passenger document a strong lexical match for any query containing "domestic",
    "rules" or "tariff" — including cargo shipping questions.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for text in pages:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines[:3] + lines[-3:]:
            # Mask digits so "Page 4 of 23" and "Page 17 of 23" collapse together.
            counts[re.sub(r"\d+", "#", line).lower()] += 1
    cutoff = max(2, int(len(pages) * threshold))
    return {key for key, n in counts.items() if n >= cutoff}


def strip_running_lines(text: str, running: set[str]) -> str:
    kept = [
        ln for ln in text.splitlines()
        if re.sub(r"\d+", "#", ln.strip()).lower() not in running
        and not _PAGE_FOOTER.match(ln)
    ]
    return "\n".join(kept).strip()


def page_is_clean(text: str, min_chars: int = 200, max_bad_ratio: float = 0.001) -> bool:
    """Heuristic used for the 'clean text %' column of the Stage 1 table.

    A page counts as cleanly extracted when it (a) yielded a plausible amount of
    text, (b) contains no replacement/NUL characters from failed glyph decoding,
    and (c) is mostly real words rather than shattered single characters.
    """
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    if any(c in stripped for c in _REPLACEMENT_CHARS):
        return False
    bad = sum(stripped.count(c) for c in _REPLACEMENT_CHARS)
    if bad / max(len(stripped), 1) > max_bad_ratio:
        return False
    words = _WORD_RE.findall(stripped)
    if not words:
        return False
    # Real prose averages >3 chars/word; shattered extraction trends toward 1-2.
    return sum(len(w) for w in words) / len(words) >= 3.0
