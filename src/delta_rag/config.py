"""Central configuration: paths, Ollama endpoints, and the model registry.

Everything in this project runs against a local Ollama server exposed through its
OpenAI-compatible ``/v1`` surface, so the whole pipeline is reproducible offline
with no hosted API spend.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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

# Independent judges from *different* model families, so RAGAS/DeepEval agreement
# is a real signal rather than one model agreeing with itself.
RAGAS_JUDGE_MODEL = "qwen2.5:7b"
DEEPEVAL_JUDGE_MODEL = "llama3.1:latest"

# Local embedding model served by Ollama (used as one Stage 3 candidate alongside
# the sentence-transformers models).
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"

RANDOM_SEED = 20260821
