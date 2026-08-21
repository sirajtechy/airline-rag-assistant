"""Stage 2 — chunking strategies.

Four genuinely different strategies, not four sizes of the same one:

``fixed``       token windows with no overlap (the naive baseline)
``fixed_ov``    token windows with overlap (rescues facts split across a boundary)
``recursive``   LangChain RecursiveCharacterTextSplitter, structure-aware
``rule_aware``  splits on the tariffs' own rule headings ("RULE 20:", "G14"),
                a document-specific strategy that respects legal boundaries

Chunk sizes are measured in **tokens** via ``tiktoken`` so the numbers in the
Stage 2 table mean what the methodology says they mean.

Every chunk keeps the ``locators`` of the source units it came from, which is
what lets chunk-level retrieval be scored against locator-level ground truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .corpus import SourceUnit

# The financial route is already one record per concept at ~535 chars; splitting
# it would sever a concept from its own values. Those units pass through whole.
PRESPLIT_DOC_KINDS = {"financial_xbrl"}


@dataclass
class Chunk:
    """A retrievable unit, traceable back to the pages/concepts that produced it."""

    chunk_id: str
    doc_id: str
    business_line: str
    locators: list[str]
    text: str
    title: str
    meta: dict = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


# ── Strategy implementations ─────────────────────────────────────────────────
def _split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Token-window split. ``overlap=0`` gives the no-overlap baseline."""
    enc = _encoder()
    tokens = enc.encode(text)
    if not tokens:
        return []
    if overlap >= size:
        raise ValueError(f"overlap {overlap} must be smaller than size {size}")
    step = size - overlap
    out = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            break
        out.append(enc.decode(window))
        if start + size >= len(tokens):
            break
    return out


def _split_recursive(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# "RULE 20:" / "RULE 3" in the contract, "G14" / "G60" in the cargo tariff.
_RULE_HEADING = re.compile(r"^(?:RULE\s+\d+[:.]?|G\d{1,3}\b)", re.M)


def _split_rule_aware(text: str, size: int, overlap: int) -> list[str]:
    """Split on the documents' own rule headings, then size-cap each rule.

    Both tariffs are organised as numbered rules, so a rule boundary is a real
    semantic boundary. Oversized rules still get windowed so no chunk blows past
    the embedding model's context.
    """
    boundaries = [m.start() for m in _RULE_HEADING.finditer(text)]
    if not boundaries:
        return _split_recursive(text, size, overlap)
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(len(text))

    out: list[str] = []
    for start, end in zip(boundaries, boundaries[1:]):
        section = text[start:end].strip()
        if not section:
            continue
        if count_tokens(section) <= size:
            out.append(section)
        else:
            out.extend(_split_recursive(section, size, overlap))
    return out


STRATEGIES = {
    "fixed": lambda t, s, o: _split_fixed(t, s, 0),
    "fixed_ov": _split_fixed,
    "recursive": _split_recursive,
    "rule_aware": _split_rule_aware,
}


# ── Corpus-level chunking ────────────────────────────────────────────────────
def _group_units(units: list[SourceUnit]) -> dict[str, list[SourceUnit]]:
    grouped: dict[str, list[SourceUnit]] = {}
    for u in units:
        grouped.setdefault(u.doc_id, []).append(u)
    return grouped


def _locators_for(chunk_text: str, units: list[SourceUnit], offsets: list[tuple[int, int, str]],
                  search_from: int) -> tuple[list[str], int]:
    """Map a chunk back to the source locators whose character ranges it covers.

    Chunks are produced from the concatenated document text, so a chunk can span
    a page boundary and legitimately carry two locators.
    """
    idx = " ".join(u.text for u in units).find(chunk_text[:120], search_from)
    if idx < 0:
        idx = search_from
    start, end = idx, idx + len(chunk_text)
    hit = [loc for (s, e, loc) in offsets if s < end and start < e]
    return (hit or [offsets[0][2]]), max(start, 0)


def chunk_corpus(
    units: list[SourceUnit],
    strategy: str = "recursive",
    size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Chunk every document, preserving locator provenance for scoring."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(STRATEGIES)}")
    splitter = STRATEGIES[strategy]

    chunks: list[Chunk] = []
    for doc_id, doc_units in _group_units(units).items():
        # Pre-split sources (XBRL concepts) are already the right granularity.
        if doc_id in PRESPLIT_DOC_KINDS:
            for u in doc_units:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::{u.locator}",
                        doc_id=doc_id,
                        business_line=u.business_line,
                        locators=[u.locator],
                        text=u.text,
                        title=u.title,
                        meta={"strategy": "presplit", **u.meta},
                    )
                )
            continue

        joined, offsets, cursor = [], [], 0
        for u in doc_units:
            joined.append(u.text)
            offsets.append((cursor, cursor + len(u.text), u.locator))
            cursor += len(u.text) + 1  # +1 for the join space
        full_text = " ".join(joined)

        search_from = 0
        for i, piece in enumerate(splitter(full_text, size, overlap)):
            piece = piece.strip()
            if not piece:
                continue
            locators, search_from = _locators_for(piece, doc_units, offsets, search_from)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{strategy}::{i}",
                    doc_id=doc_id,
                    business_line=doc_units[0].business_line,
                    locators=locators,
                    text=piece,
                    title=doc_units[0].title,
                    meta={
                        "strategy": strategy,
                        "size": size,
                        "overlap": overlap,
                        "n_tokens": count_tokens(piece),
                    },
                )
            )
    return chunks
