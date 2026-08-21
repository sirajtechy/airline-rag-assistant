# Project notes — Delta Air Lines RAG (Group 7 capstone)

## What this project is graded on
Per `data/raw/EVALUATION_METHODOLOGY.md` (the authoritative spec):

| Weight | Component |
|---|---|
| 40% | 8 ablation tables, real measured numbers, written winner justification per stage |
| 30% | Passenger-vs-cargo routing on the 5 acceptance tests |
| 20% | End-to-end RAGAS + DeepEval on the final pipeline |
| 10% | Code quality / app usability |

The evaluation study *is* the assignment. A working chatbot with no measured
evidence caps at 30%.

## Environment
- Python 3.12 venv at `.venv` (course suggests 3.11; 3.12 resolves everything cleanly).
- `numpy<2` is **not** needed — that pin exists in the course requirements only for
  `rapidocr`/`opencv`, which this project does not use. We run numpy 2.5.2.
- All LLM/embedding inference is local via **Ollama** on `http://127.0.0.1:11434`.
  Note the shell exports `OPENAI_API_KEY=ollama` and `OPENAI_BASE_URL` pointing at
  Ollama, so there is no hosted API spend and no real OpenAI access.

## Commands
```bash
.venv/bin/python -m pytest                      # unit tests (pytest.ini sets pythonpath=src)
PYTHONPATH=src .venv/bin/python -m delta_rag.evalset   # validate the labelled eval set
```

## Measured facts worth remembering
- **Judge latency through Ollama's `/v1`** (first call, includes model load):
  qwen2.5:7b 4.8s · llama3.1:8b 6.9s · gemma4 28.6s · qwen3:14b ~120s ·
  qwen3.6:35b-a3b 299s. The qwen3 family burns hidden reasoning tokens even when
  the output is JSON, making it unusable for judge duty (hundreds of calls).
  qwen3.6:35b-a3b was also the only model to return a *wrong* verdict.
- Judges are therefore **qwen2.5:7b for RAGAS** and **llama3.1:8b for DeepEval** —
  deliberately different model families so cross-framework agreement is a real signal.
- All models tested return clean parseable JSON with `response_format=json_object`;
  no `<think>` leakage through `/v1`.
- Source docs: contract 23 pages, cargo tariff 22 pages (matches the README).
  `file` misreports the contract as 20 pages — it reads a stale metadata field.
- XBRL yields **420 concept records / 1,394 facts**, averaging 535 chars — already
  chunk-sized, so the financial route is not further split.
- FY2025 headline facts: revenue $63.364B, operating income $5.822B, net income
  $5.005B, diluted EPS $7.66, total assets $81.317B, cargo revenue $900M.

## Key design decisions (and why)
1. **Ground truth is parser/chunker-invariant.** Labels are
   `(gold_doc, gold_locators, gold_span)` where a locator is `p<N>` for PDF pages or
   `<prefix>:<Concept>` for XBRL. Chunk-ID labels would be destroyed by the Stage 1
   and Stage 2 ablations, which is the trap this design avoids.
2. **Binary relevance threshold is grade >= 2, not 3.** pypdf shatters "those" into
   "t hose" on contract page 4. Requiring the verbatim span would score that page a
   total miss and double-count the damage. Instead it costs NDCG@3 (graded) while
   leaving R@3 (binary) intact. Pinned by `test_relevance_threshold_is_grade_two`.
3. **Labels are gated against one reference parser (`pymupdf`)** so ground truth is
   fixed while Stage 1 varies the parser under test. Per-parser span recoverability
   is reported as Stage 1 *evidence* rather than treated as a label failure.

## Gotchas
- `delta_rag.config` creates `data/interim`, `data/indexes`, `reports/results` on
  import; those are gitignored.
- Eval set has 30 questions (12 passenger / 11 cargo / 7 financial;
  14 keyword / 16 semantic). `F03` and `R1` are deliberate routing hard-negatives:
  cargo *revenue* is a financial fact, not a shipping rule.
