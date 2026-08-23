"""Stage 8, part 1 — generate answers with each candidate LLM.

Retrieval is held completely fixed (the Stage 1-7 winner) and only the generator
varies, so any difference in the generation metrics is attributable to the model
rather than to what it was given to read.

Generations are cached to disk because they are the expensive step; the judging
pass in stage8_evaluate.py reads the cache, so RAGAS or DeepEval falling over
never costs a re-generation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_rag.config import RESULTS_DIR
from delta_rag.evalset import load_eval_set
from delta_rag.pipeline import DeltaSupportBot

# Generation ran for all four; judging is limited to three by Gemini's free-tier
# quota (20 requests/min per model). Three still spans a 3B-vs-7B-vs-8B size
# contrast across two model families, comfortably above the methodology's
# "at least 2" requirement. gemma4 generations remain on disk in
# reports/results/ and are reported on the quota-free metrics.
# Judged candidates. Two is the methodology's stated minimum and what this
# hardware supports: local judging costs ~235 s per answer (RAGAS 103 s +
# DeepEval 132 s), so each additional model adds ~2 h. These two give the widest
# contrast available - 3B vs 7B, two different model families - and qwen2.5:7b is
# the generator actually shipped in the app. Generations for llama3.1:8b and
# gemma4 remain on disk and are reported on the quota-free metrics.
CANDIDATES = ["qwen2.5:7b", "llama3.2:3b"]
ALL_GENERATED = ["llama3.2:3b", "qwen2.5:7b", "llama3.1:latest", "gemma4:latest"]


def generations_path(model: str) -> Path:
    return RESULTS_DIR / f"generations__{model.replace(':', '_')}.json"


def run_model(model: str, bot: DeltaSupportBot, questions) -> dict:
    path = generations_path(model)
    if path.exists():
        print(f"  [{model}] cached, skipping")
        return json.loads(path.read_text())

    bot.model = model
    records, started = [], time.time()
    for i, q in enumerate(questions, start=1):
        response = bot.answer(q.question)
        records.append({
            "id": q.id,
            "question": q.question,
            "route": q.route,
            "gold_route": q.route,
            "predicted_route": response.route,
            "ideal_answer": q.ideal_answer,
            "answer": response.answer,
            "contexts": response.contexts,
            "blocked": response.blocked,
            "citations_valid": response.citations_valid,
            "generation_ms": response.generation_ms,
            "retrieval_ms": response.retrieval_ms,
        })
        print(f"  [{model}] {i}/{len(questions)} {q.id} "
              f"({response.generation_ms:.0f} ms)", flush=True)

    payload = {
        "model": model,
        "total_seconds": time.time() - started,
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"  [{model}] wrote {path}")
    return payload


def main() -> None:
    questions = load_eval_set()
    bot = DeltaSupportBot()
    print(f"retrieval config (fixed): {bot.config.label()}")
    for model in CANDIDATES:
        print(f"=== {model} ===")
        try:
            run_model(model, bot, questions)
        except Exception as exc:  # a missing/failing model must not abort the sweep
            print(f"  [{model}] FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
