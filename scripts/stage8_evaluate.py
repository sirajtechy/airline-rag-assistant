"""Stage 8, part 2 — judge the cached generations with RAGAS and DeepEval.

Reads the generation cache written by stage8_generate.py, so judging can be
re-run without re-generating. Retrieval was identical for every model, which is
what makes this a clean comparison of generators.

RAGAS is judged by qwen2.5:7b and DeepEval by llama3.1:8b — different families,
so their agreement carries information.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_rag.ablation import save_stage
from delta_rag.config import DEEPEVAL_JUDGE_MODEL, RAGAS_JUDGE_MODEL, RESULTS_DIR
from delta_rag.evaluation import EvalRecord, run_deepeval, run_ragas

from stage8_generate import CANDIDATES, generations_path

LABELS = {
    "llama3.2:3b": "Llama 3.2 3B",
    "qwen2.5:7b": "Qwen2.5 7B",
    "llama3.1:latest": "Llama 3.1 8B",
    "gemma4:latest": "Gemma 4 (~9B)",
}


def load_records(model: str) -> tuple[list[EvalRecord], dict]:
    payload = json.loads(generations_path(model).read_text())
    # Blocked (guardrail) answers have no retrieved context to be faithful to, so
    # including them would punish a model for the guardrail firing correctly.
    usable = [r for r in payload["records"] if not r["blocked"] and r["contexts"]]
    records = [
        EvalRecord(
            question=r["question"], answer=r["answer"],
            contexts=r["contexts"], ideal_answer=r["ideal_answer"],
        )
        for r in usable
    ]
    stats = {
        "n_scored": len(records),
        "n_blocked": sum(r["blocked"] for r in payload["records"]),
        "route_accuracy": sum(
            r["predicted_route"] == r["gold_route"] for r in payload["records"]
        ) / len(payload["records"]),
        "citations_valid": sum(r["citations_valid"] for r in usable) / max(len(usable), 1),
        "mean_generation_ms": sum(r["generation_ms"] for r in usable) / max(len(usable), 1),
        "answer_chars": sum(len(r["answer"]) for r in usable) / max(len(usable), 1),
    }
    return records, stats


def main() -> None:
    cache_path = RESULTS_DIR / "stage8_scores_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    rows = []
    for model in CANDIDATES:
        if not generations_path(model).exists():
            print(f"=== {model}: no generations, skipping ===")
            continue
        print(f"=== {model} ===", flush=True)
        records, stats = load_records(model)

        if model in cache:
            ragas_scores, deepeval_scores = cache[model]["ragas"], cache[model]["deepeval"]
            print("  cached scores")
        else:
            t = time.time()
            print(f"  RAGAS   (judge={RAGAS_JUDGE_MODEL}) on {len(records)} answers…",
                  flush=True)
            try:
                ragas_scores = run_ragas(records)
            except Exception as exc:
                print(f"    RAGAS FAILED: {type(exc).__name__}: {exc}")
                ragas_scores = {}
            print(f"    done in {time.time() - t:.0f}s -> {ragas_scores}", flush=True)

            t = time.time()
            print(f"  DeepEval (judge={DEEPEVAL_JUDGE_MODEL})…", flush=True)
            try:
                deepeval_scores = run_deepeval(records)
            except Exception as exc:
                print(f"    DeepEval FAILED: {type(exc).__name__}: {exc}")
                deepeval_scores = {}
            print(f"    done in {time.time() - t:.0f}s -> {deepeval_scores}", flush=True)

            cache[model] = {"ragas": ragas_scores, "deepeval": deepeval_scores}
            cache_path.write_text(json.dumps(cache, indent=2))

        def g(d, k):
            v = d.get(k)
            return None if v is None or v != v else round(float(v), 4)

        rows.append({
            "LLM": LABELS.get(model, model),
            "Faithfulness (RAGAS)": g(ragas_scores, "faithfulness"),
            "Answer Relevancy (RAGAS)": g(ragas_scores, "answer_relevancy"),
            "Context Precision (RAGAS)": g(ragas_scores, "context_precision"),
            "Context Recall (RAGAS)": g(ragas_scores, "context_recall"),
            "Faithfulness (DeepEval)": g(deepeval_scores, "faithfulness"),
            "Hallucination (DeepEval, lower=better)": g(deepeval_scores, "hallucination"),
            "G-Eval PolicyAccuracy (DeepEval)": g(deepeval_scores, "g_eval_policy_accuracy"),
            "Route acc": round(stats["route_accuracy"], 4),
            "Valid citations": round(stats["citations_valid"], 4),
            "Latency (ms/answer)": round(stats["mean_generation_ms"]),
            "Cost/query": "$0.00 (local)",
        })

    notes = (
        "Retrieval is byte-identical across every row — same parser, chunking, "
        "embeddings, store and fusion — so these differences are attributable to the "
        "generator alone. Context Precision and Context Recall are properties of the "
        "shared retrieval step and should be near-constant down the table; any spread "
        "in them is judge noise, which is itself a useful read on how much to trust the "
        "other columns.\n\n"
        f"**Two independent judges.** RAGAS is judged by `{RAGAS_JUDGE_MODEL}` and "
        f"DeepEval by `{DEEPEVAL_JUDGE_MODEL}` — deliberately different model families. "
        "Where the two faithfulness columns agree, the score is trustworthy; where they "
        "diverge sharply, the honest conclusion is that neither number is reliable "
        "rather than that the flattering one is right.\n\n"
        "**Cost is $0.00 for every candidate** because everything runs locally on "
        "Ollama, so cost cannot discriminate between models here. Latency and quality "
        "do the work instead. The G-Eval criterion is custom-written for this bot: it "
        "rewards exact figures, deadlines and dollar amounts and penalises invented "
        "fees, because a fluent answer that loses the $1,075 cap has failed the "
        "customer even though a generic relevancy metric would pass it.\n\n"
        "Guardrail-blocked answers are excluded from the generation metrics: they have "
        "no retrieval context to be faithful to, so scoring them would penalise a model "
        "for the guardrail working."
    )
    save_stage("stage8_generation", "Stage 8 — LLM for generation", rows, notes=notes)
    print("\nwrote stage8_generation.md")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
