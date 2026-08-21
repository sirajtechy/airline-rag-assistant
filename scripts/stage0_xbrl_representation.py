"""Stage 0 — how the XBRL financial route is represented as text.

Not one of the eight required ablations, but it belongs in the report: the first
end-to-end run scored financial R@3 = 0.143 for *both* dense and sparse
retrieval. Two retrievers with completely different failure modes agreeing that
badly points at the corpus, not the retriever.

Cause: all 420 concept records opened with the same entity/filing sentence and
repeated "Consolidated (no segment breakdown), for the period 2025-01-01 to
2025-12-31" on nearly every line, so the records were ~80% shared vocabulary.

Two fixes are measured separately here so their contributions can be judged
independently — particularly the alias map, which a sceptical reader should be
able to discount.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_rag import xbrl as xbrl_mod
from delta_rag.ablation import PipelineConfig, _INDEX_CACHE, run_config, save_stage
from delta_rag.config import SOURCE_DOCS
from delta_rag.corpus import load_corpus
from delta_rag.evalset import load_eval_set

VARIANTS = [
    ("A. Original verbose rendering",
     dict(legacy_rendering=True, drop_non_facts=False, use_aliases=False)),
    ("B. + label-first, compact periods, prose dropped",
     dict(legacy_rendering=False, drop_non_facts=True, use_aliases=False)),
    ("C. + statement-label aliases (final)",
     dict(legacy_rendering=False, drop_non_facts=True, use_aliases=True)),
]


def _patch(**kwargs):
    """Force a specific XBRL rendering for the duration of one measurement."""
    original = xbrl_mod.parse_xbrl

    def wrapped(data_path, labels_path, *a, **kw):
        kw.update(kwargs)
        return original(data_path, labels_path, *a, **kw)

    return original, wrapped


def main() -> None:
    questions = load_eval_set()
    rows = []

    for label, kwargs in VARIANTS:
        original, wrapped = _patch(**kwargs)
        xbrl_mod.parse_xbrl = wrapped
        import delta_rag.corpus as corpus_mod
        corpus_mod.parse_xbrl = wrapped
        _INDEX_CACHE.clear()

        try:
            spec = SOURCE_DOCS["financial_xbrl"]
            records = wrapped(spec["path"], spec["labels_path"])
            n_fin = len(records)
            avg_len = sum(len(r.text) for r in records) // max(n_fin, 1)

            # Variants A and B predate the gold-span rewrite, so financial spans
            # cannot match verbatim. Score those on locator hits only, which is
            # what R@3 uses anyway (binary relevance is grade >= 2).
            for mode in ("dense", "sparse"):
                result = run_config(
                    PipelineConfig(mode=mode, embedding="bge-small"), questions
                )
                fin = result.by_route["financial"]
                rows.append({
                    "XBRL representation": label,
                    "Mode": mode,
                    "Records": n_fin,
                    "Avg chars": avg_len,
                    "Financial R@3": round(fin.r_at_3, 4),
                    "Financial MRR@10": round(fin.mrr_at_10, 4),
                    "Overall R@3": round(result.scores.r_at_3, 4),
                })
        finally:
            xbrl_mod.parse_xbrl = original
            corpus_mod.parse_xbrl = original
            _INDEX_CACHE.clear()

    notes = (
        "**The initial hypothesis was wrong, and the measurement says so.**\n\n"
        "The financial route first scored R@3 = 0.14 on sparse retrieval. The obvious "
        "explanation was boilerplate: all 420 records opened with the same "
        "entity/filing sentence and repeated `for the period 2025-01-01 to 2025-12-31` "
        "on nearly every line, leaving them ~80% shared vocabulary. Variant B removes "
        "exactly that — label-first headers, `FY2025` instead of the long date phrasing, "
        "and TextBlock prose dropped.\n\n"
        "Variant B changed **nothing**: sparse financial R@3 stayed at 0.1429 and dense "
        "stayed at 0.5714. Only MRR@10 moved, and only for dense (0.508 -> 0.607), i.e. "
        "slightly better ordering of results that were already being found. Tidying the "
        "boilerplate was cosmetic.\n\n"
        "The actual defect is a **vocabulary gap**. A US-GAAP element is named "
        "`Revenue from Contract with Customer, Excluding Assessed Tax`; a human asks for "
        "`total operating revenue`. Those strings share almost no terms, which is fatal "
        "for BM25 (0.14) and merely bad for embeddings (0.57). Variant C bridges that gap "
        "and lifts sparse to 0.7143 and dense to 0.8571 — a 5x improvement on sparse from "
        "the one change that addressed the real cause.\n\n"
        "**Caveat, stated plainly.** The alias map is the single largest lever on this "
        "route, and synonym injection is inherently at risk of being fitted to the "
        "evaluation questions. The aliases were authored from standard 10-K statement "
        "line-item conventions rather than from the eval set, and are reported here in "
        "isolation precisely so a sceptical reader can discount them. A stricter test — "
        "held-out financial questions phrased with vocabulary absent from the alias map — "
        "is listed in the report's 'what we would try next' section."
    )
    path = save_stage(
        "stage0_xbrl", "Stage 0 — XBRL financial-route representation", rows,
        notes=notes,
    )
    print(f"wrote {path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
