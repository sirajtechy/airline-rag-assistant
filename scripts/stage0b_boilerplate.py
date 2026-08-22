"""Stage 0b — front matter and running-header removal.

Found by debugging a failing acceptance test, not by inspecting the corpus: A2
("What are the rules for shipping cargo domestically with Delta?") routed to the
*passenger* document. The retrieved evidence showed why:

  1. contract p1   "DELTA DOMESTIC GENERAL RULES TARIFF"
  2. contract p1-2 "RULE 24: GOVERNING LAW ... LIMITATION OF LIABILITY ........"
  3. cargo p2      "G26 Air Waybill and Shipping Documents ................."

Two distinct defects. The contract's **title page** contains the words
"domestic", "rules" and "tariff", and its running header repeats that phrase on
all 23 pages. And both documents' **contents pages** list every rule title in the
document, giving them enormous lexical density and zero answer content — the
worst possible ratio for a retriever.

Neither is exotic; both are what real PDFs look like. This measures the fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import delta_rag.ablation as ablation_mod
import delta_rag.corpus as corpus_mod
from delta_rag.ablation import PipelineConfig, run_config, save_stage
from delta_rag.evalset import load_eval_set

FINAL = PipelineConfig(
    parser="pymupdf", strategy="rule_aware", size=800, overlap=100,
    embedding="bge-small", store="faiss", mode="hybrid", fusion="rrf",
)


def main() -> None:
    questions = load_eval_set()
    original = corpus_mod.load_corpus
    rows = []

    for label, clean in [("A. Raw pages (front matter + headers kept)", False),
                         ("B. Front matter dropped, running headers stripped", True)]:
        def patched(parser="pymupdf", clean=clean, _o=original):
            return _o(parser, clean=clean)

        corpus_mod.load_corpus = patched
        ablation_mod.load_corpus = patched
        ablation_mod._INDEX_CACHE.clear()
        try:
            units = patched()
            n_pdf = len([u for u in units if u.doc_id != "financial_xbrl"])
            for mode in ("sparse", "hybrid"):
                r = run_config(PipelineConfig(**{**FINAL.__dict__, "mode": mode}),
                               questions)
                rows.append({
                    "Preprocessing": label,
                    "Mode": mode,
                    "PDF pages": n_pdf,
                    "R@1": round(r.scores.r_at_1, 4),
                    "R@3": round(r.scores.r_at_3, 4),
                    "MRR@10": round(r.scores.mrr_at_10, 4),
                    "NDCG@3": round(r.scores.ndcg_at_3, 4),
                    "R@3 (cargo)": round(r.by_route["cargo"].r_at_3, 4),
                    "R@3 (passenger)": round(r.by_route["passenger"].r_at_3, 4),
                })
        finally:
            corpus_mod.load_corpus = original
            ablation_mod.load_corpus = original
            ablation_mod._INDEX_CACHE.clear()

    save_stage(
        "stage0b_boilerplate", "Stage 0b — Front matter and running-header removal", rows,
        notes=(
            "Only **2 of 45 pages** are removed (contract p1, cargo p2), and no gold "
            "answer span lives on either, so the eval labels still validate 30/30.\n\n"
            "The mechanism is worth stating precisely, because it explains why a "
            "contents page is worse than useless. A table of contents lists every rule "
            "title in the document: `RULE 20: DENIED BOARDING COMPENSATION`, `G32 Limit "
            "of Liability`, and so on. To BM25 that page looks maximally relevant to "
            "almost any topical query, while containing no rule text to answer it with. "
            "It is a chunk engineered to win retrieval and then say nothing.\n\n"
            "The running header is the subtler of the two. `Delta Domestic General Rules "
            "Tariff` appears on all 23 pages of the passenger contract, so the terms "
            "'domestic', 'rules' and 'tariff' are indexed 23 times in the passenger "
            "document — which is why a *cargo* question phrased as 'rules for shipping "
            "cargo domestically' was pulled toward the passenger contract. The "
            "cross-contamination this group is graded on was partly caused by page "
            "furniture.\n\n"
            "This was found by debugging a failing acceptance test rather than by "
            "reading the PDFs, which is an argument for the methodology itself: the "
            "aggregate retrieval metrics were already respectable while a required "
            "acceptance test was failing for a cause no aggregate number surfaced."
        ),
    )
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
