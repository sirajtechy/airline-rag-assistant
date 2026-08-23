"""Central configuration: paths, Ollama endpoints, and the model registry.

Everything in this project runs against a local Ollama server exposed through its
OpenAI-compatible ``/v1`` surface, so the whole pipeline is reproducible offline
with no hosted API spend.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # load .env so GOOGLE_API_KEY is picked up without exporting it by hand
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_INDEX = PROJECT_ROOT / "data" / "indexes"
EVAL_DIR = PROJECT_ROOT / "eval"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"

for _d in (DATA_INTERIM, DATA_INDEX, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_V1_URL = f"{OLLAMA_BASE_URL}/v1"
OLLAMA_API_KEY = "ollama"  # Ollama ignores the value but clients require one.

# ── Business lines (the routing target this group is graded on) ──────────────
PASSENGER = "passenger"
CARGO = "cargo"
FINANCIAL = "financial"
BUSINESS_LINES = (PASSENGER, CARGO, FINANCIAL)

SOURCE_DOCS: dict[str, dict] = {
    "contract_of_carriage": {
        "path": DATA_RAW / "Delta_contract_of_carriage.pdf",
        "business_line": PASSENGER,
        "title": "Delta Domestic General Rules Tariff (Contract of Carriage)",
        "kind": "pdf",
    },
    "cargo_tariff": {
        "path": DATA_RAW / "Delta_Cargo_Shipping_Rules_Tariff.pdf",
        "business_line": CARGO,
        "title": "Delta Cargo U.S. Domestic Shipping Rules Tariff",
        "kind": "pdf",
    },
    "financial_xbrl": {
        "path": DATA_RAW / "Delta_Financial_Data_XBRL.xml",
        "labels_path": DATA_RAW / "Delta_Financial_Labels_XBRL.xml",
        "business_line": FINANCIAL,
        "title": "Delta Air Lines FY2025 10-K XBRL Financial Facts",
        "kind": "xbrl",
    },
}


@dataclass(frozen=True)
class LLMSpec:
    """A generation/judge candidate, with measured latency recorded in reports."""

    name: str            # Ollama tag
    label: str           # human-readable name for report tables
    family: str          # used to keep the two eval judges independent
    params: str


# Measured in scripts/00_judge_benchmark.py — see reports/results/judge_benchmark.md.
# qwen3:14b and qwen3.6:35b-a3b are excluded from judge duty (100-300 s/call through
# /v1 because hidden reasoning tokens are still generated), but qwen3:14b is retained
# as a Stage 8 *generation* candidate where the call volume is far lower.
LLM_CANDIDATES: dict[str, LLMSpec] = {
    "qwen2.5:7b": LLMSpec("qwen2.5:7b", "Qwen2.5 7B", "qwen", "7B"),
    "llama3.1:latest": LLMSpec("llama3.1:latest", "Llama 3.1 8B", "llama", "8B"),
    "gemma4:latest": LLMSpec("gemma4:latest", "Gemma 4", "gemma", "~9B"),
    "qwen3:14b": LLMSpec("qwen3:14b", "Qwen3 14B", "qwen", "14B"),
    "llama3.2:3b": LLMSpec("llama3.2:3b", "Llama 3.2 3B", "llama", "3B"),
}

# ── Evaluation judges ────────────────────────────────────────────────────────
# Judging is the one place this project does not use local models. Two reasons,
# both measured:
#
# 1. Speed. Ollama allocates each model's maximum context unless told otherwise
#    (131,072 tokens for Llama 3.1, reserving ~22 GB), which made local judging
#    ~175 s per record. RAGAS reaches its judge through langchain-openai and
#    DeepEval through its own OllamaModel, and neither exposes num_ctx, so the cap
#    has to be baked into the model definition (scripts/create_judge_models.sh).
#    Even capped, a full 4-model sweep is hours.
# 2. Judge quality, which matters more. A local 7-8B judge is weak: DeepEval
#    scored one answer at faithfulness 0.79 while its own hallucination metric
#    said 0.60 — close to self-contradictory. Since these scores are 20% of the
#    grade, a stronger judge is worth more than local purity.
#
# Gemini Flash is used instead, with the local models retained as a fallback so
# the pipeline still runs fully offline if no key is present.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# Gemini judging is opt-in rather than automatic. It was measured and rejected for
# this workload: RAGAS needs roughly 330 LLM calls per candidate model (context
# precision alone costs one call per retrieved context), and the Gemini free tier
# allows ~20 requests/min per model. Client-side pacing still drew 429s, and each
# one carries a server-mandated ~48 s backoff, which stalled throughput to zero.
# Local judges have no quota at all, which is the binding constraint here.
# Set DELTA_RAG_JUDGE=gemini with a paid key to use it.
USE_GEMINI_JUDGE = bool(GOOGLE_API_KEY) and os.getenv("DELTA_RAG_JUDGE") == "gemini"

# Two *different* pinned model generations, not one shared judge. The methodology
# treats RAGAS and DeepEval as independent second opinions, and that only holds if
# the underlying judges differ — otherwise their agreement is one model agreeing
# with itself. Cross-generation is weaker independence than cross-vendor, and the
# report says so, but it is far better than identical judges.
#
# Versions are pinned rather than using the "-latest" aliases so a re-run months
# from now reproduces these numbers instead of silently swapping the judge.
# Verified callable with this API key. Note the /v1beta/models listing advertises
# older models (gemini-2.5-flash, gemini-3.5-flash) that new keys cannot actually
# call — they return 404 "no longer available to new users" — so the pair below was
# chosen by probing generateContent rather than by trusting the model list.
# gemini-3.7-flash was excluded: it returned 503 under load.
GEMINI_RAGAS_JUDGE = "gemini-3.6-flash"
GEMINI_DEEPEVAL_JUDGE = "gemini-3.5-flash-lite"

# Local fallbacks (Ollama derivatives with num_ctx pinned to 8192).
LOCAL_RAGAS_JUDGE_MODEL = "qwen2.5-7b-judge:latest"
LOCAL_DEEPEVAL_JUDGE_MODEL = "llama3.1-latest-judge:latest"

RAGAS_JUDGE_MODEL = GEMINI_RAGAS_JUDGE if USE_GEMINI_JUDGE else LOCAL_RAGAS_JUDGE_MODEL
DEEPEVAL_JUDGE_MODEL = (
    GEMINI_DEEPEVAL_JUDGE if USE_GEMINI_JUDGE else LOCAL_DEEPEVAL_JUDGE_MODEL
)

# Local embedding model served by Ollama (used as one Stage 3 candidate alongside
# the sentence-transformers models).
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"

RANDOM_SEED = 20260821
