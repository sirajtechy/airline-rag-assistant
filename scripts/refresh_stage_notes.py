"""Regenerate Stage 3 and Stage 6 prose from their cached JSON.

Both stages had written verdicts that no longer matched their own measurements:

* Stage 6's note was hard-coded to "Selected: Reciprocal Rank Fusion" even after
  the tie-break logic selected weighted alpha = 0.5, so the prose contradicted
  both the table beneath it and `final_retrieval_config.json`.
* Stage 3's note claimed bge-small beat MiniLM by +0.1333 R@3, a gap measured
  *before* the Stage 0b boilerplate cleaning. On the cleaned corpus they tie.

The measurements themselves are unchanged, so this rewrites the narrative from the
stored rows rather than re-running the sweeps. `run_retrieval_ablations.py` has
been fixed to generate the correct text directly on any future run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_rag.ablation import save_stage
from delta_rag.config import RESULTS_DIR

N_QUESTIONS = 30


def rows(stage: str) -> list[dict]:
    return json.loads((RESULTS_DIR / f"{stage}.json").read_text())


def refresh_stage3() -> None:
    data = rows("stage3_embeddings")
    best = max(data, key=lambda r: (r["R@3"], r["MRR@10"]))
    top = max(r["R@3"] for r in data)
    tied = [r["Key"] for r in data if r["R@3"] == top]

    tie_note = ""
    if len(tied) > 1:
        tie_note = (
            f"\n\n**This is a near-tie, not a clear win.** {len(tied)} of the "
            f"{len(data)} candidates ({', '.join(tied)}) share the top R@3 of "
            f"{top:.4f} and separate only on secondary metrics — where MiniLM actually "
            f"takes NDCG@3 and bge-small takes MRR@10.\n\n"
            "This **contradicts an earlier run of this same table**, in which bge-small "
            "led MiniLM by +0.1333 R@3. That gap existed *before* the front-matter and "
            "running-header removal in Stage 0b. Cleaning the boilerplate that was "
            "polluting retrieval made the embedding model substantially less important, "
            "which is a result in its own right: fixing the corpus reduced the leverage "
            "of a hyperparameter. It is also a caution about ablation order — a stage "
            "measured against a defective corpus can report a difference that evaporates "
            "once the defect is fixed."
        )

    save_stage(
        "stage3_embeddings", "Stage 3 — Embedding model", data,
        notes=(
            f"Winner: **{best['Key']}** (R@3 = {best['R@3']:.4f}, "
            f"MRR@10 = {best['MRR@10']:.4f}, {best['Latency (ms/query)']} ms/query)."
            + tie_note + "\n\n"
            "`bge-small` and `nomic` are asymmetric models: they are trained to embed "
            "queries and passages with different instruction prefixes. Those prefixes "
            "are applied here (`Represent this sentence for searching relevant "
            "passages: ` and `search_query:`/`search_document:` respectively). "
            "Benchmarking them without their prefixes is a common error that would have "
            "understated both and handed the stage to MiniLM by default.\n\n"
            "`nomic` is the interesting outlier: it takes the best R@1 and the best "
            "MRR@10 of the three, but the worst R@3, and costs ~9x the latency because "
            "it is served over HTTP by Ollama rather than running in-process. It ranks "
            "its best guess well and its next guesses poorly.\n\n"
            "Latency is measured after a warm-up call so model load time is excluded."
        ),
    )
    print("refreshed stage3_embeddings.md")


def refresh_stage6() -> None:
    data = rows("stage6_fusion")
    rrf = next(r for r in data if r["alpha"] is None)
    weighted = [r for r in data if r["alpha"] is not None]
    best_w = max(weighted, key=lambda r: (r["R@3"], r["MRR@10"]))

    top = max(r["R@3"] for r in data)
    rrf_tied = rrf["R@3"] == top
    margin = best_w["R@3"] - rrf["R@3"]
    sweep = " -> ".join(f"{r['R@3']:.4f}" for r in weighted)
    alphas = ", ".join(str(r["alpha"]) for r in weighted)

    if rrf_tied:
        verdict = (
            f"**Selected: Reciprocal Rank Fusion (k=60).** RRF ties weighted linear at "
            f"alpha = {best_w['alpha']} on R@3 ({rrf['R@3']:.4f}); the tie is broken on "
            "robustness, since RRF has no hyperparameter to overfit."
        )
    else:
        verdict = (
            f"**Selected: weighted linear at alpha = {best_w['alpha']}** "
            f"(R@3 = {best_w['R@3']:.4f}, MRR@10 = {best_w['MRR@10']:.4f}, "
            f"NDCG@3 = {best_w['NDCG@3']:.4f}), ahead of RRF at "
            f"R@3 = {rrf['R@3']:.4f}.\n\n"
            f"**The margin is {margin:+.4f} R@3 — exactly "
            f"{abs(margin) * N_QUESTIONS:.0f} question out of {N_QUESTIONS}.** That is "
            "the resolution limit of this evaluation set, so the win is real but weakly "
            "evidenced, and it is reported as such rather than as a decisive result."
        )

    save_stage(
        "stage6_fusion", "Stage 6 — Hybrid merge method and weighting", data,
        notes=(
            verdict + "\n\n"
            f"**The alpha sweep is the reason for caution.** Across alpha = {alphas} the "
            f"R@3 curve runs {sweep} — **non-monotonic and spiky**. If alpha were "
            "capturing a real property of the corpus the curve would be smooth, so the "
            "peak is more plausibly a coincidence of these particular 30 questions than "
            "a tuned optimum. That is precisely the overfitting the methodology warns "
            "against, and it is why **RRF remains the safer production choice** despite "
            "losing this comparison: it has no alpha to overfit and is invariant to the "
            "two retrievers' score scales.\n\n"
            "For reference: alpha weights the dense side, `score = alpha * dense_norm + "
            "(1 - alpha) * sparse_norm`. Both distributions are min-max normalised first "
            "because BM25 scores are unbounded while cosine similarity is capped at 1; "
            "combining them raw would let BM25 dominate through scale rather than "
            "relevance. RRF sidesteps normalisation entirely by fusing ranks, at the cost "
            "of being blind to margin."
        ),
    )
    print("refreshed stage6_fusion.md")


if __name__ == "__main__":
    refresh_stage3()
    refresh_stage6()
