"""Re-score only DeepEval's HallucinationMetric, with the correct ground truth.

Why this exists
---------------
`HallucinationMetric` emits one verdict per document in `context` and returns
`hallucination_count / number_of_verdicts`. The first run passed it the five
retrieved chunks, so a correct answer that legitimately drew on one chunk was
scored as contradicting the other four. That measures retrieval breadth, not
hallucination, and inflated the score to 0.587.

DeepEval intends `context` to be ground truth, so it is now given the curated
ideal answer from the evaluation set. `FaithfulnessMetric` is the metric that
belongs on `retrieval_context`, and it was already using it correctly.

Only this one metric is recomputed; faithfulness, answer relevancy and the G-Eval
criterion are kept from the completed cache, which saves ~45 minutes per model.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from delta_rag.config import DEEPEVAL_JUDGE_MODEL, RESULTS_DIR
from delta_rag.evaluation import _deepeval_model

from stage8_evaluate import cache_path, load_records
from stage8_generate import CANDIDATES, generations_path


def gold_context_by_question() -> dict[str, list[str]]:
    """Map each eval question to the source text at its gold locator(s).

    This is the ground truth HallucinationMetric actually wants: real source
    material, scoped to the passage that contains the answer.
    """
    from delta_rag.corpus import load_corpus
    from delta_rag.evalset import load_eval_set

    units = {(u.doc_id, u.locator): u.text for u in load_corpus()}
    mapping: dict[str, list[str]] = {}
    for q in load_eval_set():
        texts = [units[(q.gold_doc, loc)] for loc in q.gold_locators
                 if (q.gold_doc, loc) in units]
        mapping[q.question] = texts or [q.ideal_answer]
    return mapping


def hallucination_for(model: str, variant: str = "gold_source",
                      max_workers: int = 2) -> float:
    """Mean HallucinationMetric score under one choice of ``context``.

    Three defensible choices give three different answers, which is the point:

    ``retrieved``    all 5 retrieved chunks. One verdict per chunk, so a correct
                     answer using one chunk is scored against four it never
                     needed. Measures retrieval breadth. Original run: 0.5868.
    ``ideal_answer`` the short curated reference. One verdict, but an answer that
                     adds *more* correct detail than the summary is marked
                     unsupported, so it penalises completeness. Gave 0.6750.
    ``gold_source``  the document text at the gold locator. Real source material,
                     complete enough that extra correct detail is still supported.
                     This is the intended semantics.
    """
    from deepeval.metrics import HallucinationMetric
    from deepeval.test_case import LLMTestCase

    records, _ = load_records(model)
    gold = gold_context_by_question() if variant == "gold_source" else {}

    def score_one(record) -> float | None:
        if variant == "retrieved":
            context = list(record.contexts)
        elif variant == "ideal_answer":
            context = [record.ideal_answer]
        else:
            context = gold.get(record.question) or [record.ideal_answer]

        metric = HallucinationMetric(model=_deepeval_model(DEEPEVAL_JUDGE_MODEL),
                                     threshold=0.3)
        case = LLMTestCase(
            input=record.question,
            actual_output=record.answer,
            expected_output=record.ideal_answer,
            context=context,
        )
        try:
            metric.measure(case)
            return None if metric.score is None else float(metric.score)
        except Exception as exc:
            print(f"    failed on {record.question[:44]!r}: {type(exc).__name__}",
                  flush=True)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = [s for s in pool.map(score_one, records) if s is not None]
    return sum(scores) / len(scores) if scores else float("nan")


def main() -> None:
    path = cache_path("deepeval")
    if not path.exists():
        print(f"no deepeval cache at {path}; run stage8_evaluate.py first")
        return
    cache = json.loads(path.read_text())

    sensitivity: list[dict] = []
    for model in CANDIDATES:
        if model not in cache or not generations_path(model).exists():
            print(f"{model}: not in cache yet, skipping")
            continue

        original = cache[model].get("hallucination")
        row = {"Model": model, "context = 5 retrieved chunks": original}

        for variant, label in [("ideal_answer", "context = ideal answer"),
                               ("gold_source", "context = gold source text")]:
            print(f"{model}: scoring hallucination with {variant}…", flush=True)
            started = time.time()
            row[label] = round(hallucination_for(model, variant), 4)
            print(f"   -> {row[label]}  ({time.time() - started:.0f}s)", flush=True)

        cache[model]["hallucination"] = row["context = gold source text"]
        cache[model]["hallucination_context"] = "gold_source (document text at gold locator)"
        cache[model]["hallucination_sensitivity"] = {
            "retrieved_chunks": original,
            "ideal_answer": row["context = ideal answer"],
            "gold_source": row["context = gold source text"],
        }
        path.write_text(json.dumps(cache, indent=2))
        sensitivity.append(row)

    # The sensitivity spread is a finding in its own right, so it gets its own
    # artefact rather than being buried in a cache file.
    if sensitivity:
        from delta_rag.ablation import save_stage

        save_stage(
            "hallucination_sensitivity",
            "DeepEval HallucinationMetric — sensitivity to the choice of `context`",
            sensitivity,
            notes=(
                "`HallucinationMetric` emits **one verdict per document in `context`** and "
                "returns `hallucination_count / number_of_verdicts`. What you pass as "
                "`context` therefore does not merely shade the score — it changes what is "
                "being measured. Three defensible choices give three different answers on "
                "identical answers and an identical judge.\n\n"
                "**5 retrieved chunks** (the original, wrong configuration). A correct "
                "answer typically draws on one chunk, so the other four generate verdicts "
                "against material the answer never needed. This measures retrieval breadth: "
                "widen `top_k` and the apparent hallucination rate rises even though the "
                "answers are unchanged.\n\n"
                "**The curated ideal answer.** One verdict per record, but the ideal answers "
                "are short summaries. An answer that correctly includes *more* detail from "
                "the source than the summary contains is judged unsupported, so this variant "
                "penalises completeness — which is why the score went *up* rather than down.\n\n"
                "**The document text at the gold locator** (selected). Real source material, "
                "scoped to the passage that actually contains the answer, and complete enough "
                "that additional correct detail is still supported. This is the semantics "
                "DeepEval intends by `context`, and it is the value reported in Stage 8.\n\n"
                "The wider lesson is the dangerous one: a metric that runs without raising is "
                "not necessarily measuring what you think. The first configuration's error was "
                "**systematic and directional**, not random — which makes it far more "
                "hazardous than noise, because it reads as a finding. It nearly put a false "
                "claim into this report: that the local judges were self-contradictory, when "
                "in fact the harness was misconfigured."
            ),
        )

    print("\nfinal deepeval cache:")
    for model, scores in cache.items():
        print(f"  {model}: {scores}")


if __name__ == "__main__":
    main()
