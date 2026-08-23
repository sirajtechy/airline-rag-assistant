# Delta Air Lines Customer Support Assistant — Evaluation-Driven RAG

**Group 7 · M10 RAG Capstone**

A retrieval-augmented support assistant over three real Delta Air Lines business
lines, built so that **every pipeline decision is backed by a measured number**
rather than by tutorial precedent.

| | |
|---|---|
| **R@3** | 0.9333 |
| **R@1** | 0.7333 |
| **MRR@10** | 0.8319 |
| **NDCG@3** | 0.8849 |
| **Routing accuracy** | 1.0000 |
| **Acceptance tests** | 5/5 required (10/11 including extra probes) |

Measured on a 30-question labelled evaluation set built *before* any design
decision was made.

---

## What this is

The brief looks like "build an airline chatbot". It isn't. The grading weights are
40% evaluation rigour, 30% routing correctness, 20% end-to-end generation scores,
10% code quality — and the methodology states plainly that a chatbot which merely
works, with no evaluation evidence, caps at the 30% line. So the deliverable is an
**8-stage ablation study**, and the chatbot is the winning configuration shipped as
an app.

**Corpus** — 45 real PDF pages and 2 real XML files, one airline, three business lines:

| Document | Business line |
|---|---|
| Delta Domestic General Rules Tariff (Contract of Carriage), 23 pages | passenger |
| Delta Cargo U.S. Domestic Shipping Rules & Tariffs, 22 pages | cargo |
| Delta FY2025 10-K XBRL facts + label linkbase | financial |

---

## Results at a glance

The final configuration, each element chosen on measured evidence:

| Stage | Winner | Deciding evidence |
|---|---|---|
| Parsing | PyMuPDF | 100% gold-span recovery vs 96.7% pypdf; 15× faster than pdfplumber |
| Chunking | recursive, 500 tok / 50 overlap | R@3 0.9000 — tied at top, margin under one question |
| Embeddings | bge-small-en-v1.5 | near-tie with MiniLM on R@3; wins MRR@10 |
| Vector store | FAISS | R@3 parity with Chroma, far faster index build |
| Retrieval mode | **Hybrid** | R@3 0.9000 vs 0.7667 dense / 0.8667 sparse |
| Fusion | weighted linear, α=0.5 | R@3 0.9333 vs 0.9000 RRF (one question — caveat documented) |
| Reranking | **none** | cross-encoder *lowered* NDCG@3 and cost 176 ms |
| Routing | top-1 chunk | 1.0000 on eval questions; beat two richer policies |

### Three findings worth reading the report for

**1. The largest win wasn't in the eight stages.** The financial route scored
R@3 = 0.1429 on sparse retrieval. Dense and sparse fail differently, so their
failing *identically* pointed at the corpus, not the config. The obvious culprit
was boilerplate — 420 XBRL records sharing a header and repeating the same period
phrasing on every line. Removing all of it changed **nothing**. The real defect was
a vocabulary gap: US-GAAP names the element `Revenue from Contract with Customer,
Excluding Assessed Tax` while a human asks for *total operating revenue*, and those
strings share almost no terms. Bridging it took sparse **0.1429 → 0.7143**.

**2. The reranker made things worse.** Adding `ms-marco-MiniLM` on tutorial
precedent would have cost both accuracy and latency: NDCG@3 fell from 0.8849 to
0.8343, and the damage grew with the candidate pool. Its training data (short web
passages) does not transfer to dense legal tariff prose, and with R@3 already at
0.90 there was no headroom to win — only correct chunks to demote.

**3. The simplest router won.** Two more sophisticated routing policies — a
rank-weighted vote, and a vote plus a prior extracted from the documents' own
definitions sections — both scored *worse* than simply trusting the top-1 retrieved
chunk.

Full evidence, including the negative results and the statistical caveats, is in
**[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md)** and the interactive
**[`PROJECT_SHOWCASE.html`](PROJECT_SHOWCASE.html)**.

---

## Quick start

Requires Python 3.12, Node 18+, and [Ollama](https://ollama.com) running locally.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

ollama pull qwen2.5:7b
ollama pull nomic-embed-text
bash scripts/create_judge_models.sh     # context-capped judge variants
```

### Run the app

```bash
# terminal 1 — API (wait for "ready"; ~18 s to embed the corpus and build FAISS)
.venv/bin/uvicorn backend.main:app --port 8000

# terminal 2 — UI
cd frontend && npm install && npm run dev     # http://localhost:5173
```

Try these to see routing and guardrails at work:

| Question | Expected behaviour |
|---|---|
| *What's Delta's policy on oversold flights?* | routes **passenger** |
| *What are the packaging requirements for shipping cargo?* | routes **cargo** |
| *How much revenue did Delta make from shipping cargo last year?* | routes **financial** — the routing trap |
| *What's JetBlue's baggage policy?* | refused before retrieval runs |

### Reproduce the evaluation

```bash
.venv/bin/python -m pytest                                # 30 metric unit tests
PYTHONPATH=src .venv/bin/python -m delta_rag.evalset      # validate all 30 gold spans

.venv/bin/python scripts/stage0_xbrl_representation.py    # financial-route text
.venv/bin/python scripts/stage0b_boilerplate.py           # front-matter removal
.venv/bin/python scripts/run_retrieval_ablations.py       # Stages 1-7
.venv/bin/python scripts/stage0c_routing.py               # routing policy
.venv/bin/python scripts/calibrate_guardrail.py           # abstain threshold
.venv/bin/python scripts/stage8_generate.py               # Stage 8 generations
.venv/bin/python scripts/stage8_evaluate.py               # RAGAS + DeepEval
.venv/bin/python scripts/run_acceptance_tests.py          # graded acceptance tests
.venv/bin/python scripts/build_report.py                  # rebuild the report
```

Stage 8 judging is the slow step (~1 hour per model per framework on local models).
Results cache per model, so it is resumable.

---

## How it works

```
question
   |
   +-- guardrails (deterministic, BEFORE retrieval)
   |     out-of-scope carrier -> refuse
   |     live booking/account -> explain limitation
   v
hybrid retrieval  (FAISS dense + BM25 sparse, weighted fusion a=0.5)
   |
   v
top-1 routing -> passenger | cargo | financial
   |
   v
scope context to routed business line     <- anti-contamination
   |
   v
abstain gate (dense cosine < 0.67 -> decline)
   |
   v
grounded generation, every claim cited [n]
```

Retrieval runs **unfiltered first** so the evidence decides the business line.
Filtering up front would require classifying the question, which is exactly what
fails on cargo-flavoured financial questions.

### Layout

```
src/delta_rag/      transport-agnostic pipeline package
  parsing.py        Stage 1 — three PDF parsers + boilerplate removal
  chunking.py       Stage 2 — four chunking strategies
  embeddings.py     Stage 3 — three backends, with asymmetric prefixes
  stores.py         Stage 4 — FAISS + ChromaDB
  retrieval.py      Stages 5-6 — dense/sparse/hybrid, RRF + weighted fusion
  rerank.py         Stage 7 — cross-encoder
  xbrl.py           financial route: XBRL -> retrievable prose
  routing.py        business-line routing
  guardrails.py     refusals, abstention, citation verification
  metrics.py        R@1, R@3, MRR@10, DCG/NDCG@3 (from the spec, not a library)
  pipeline.py       end-to-end assistant
backend/            FastAPI
frontend/           React + TypeScript + Vite
eval/               labelled eval set (30 q) + acceptance tests
scripts/            ablation + evaluation runners
notebooks/          4 notebooks walking the study
reports/results/    measured artefacts (JSON + markdown tables)
tests/              30 unit tests pinning the metric definitions
```

---

## A note on measurement integrity

Ground truth is deliberately recorded as `(gold_doc, gold_locators, gold_span)`
rather than chunk IDs. Stage 1 varies the parser and Stage 2 varies the chunker —
both change chunk boundaries and IDs, so chunk-level labels would be silently
invalidated by the first two experiments while still looking authoritative.

Binary relevance is thresholded at **grade ≥ 2**, not at the verbatim span. pypdf
extracts `those` as `t hose` on contract page 4; requiring the exact span would
score that mangled-but-correct page as a total retrieval miss and double-count a
parser already penalised elsewhere. A unit test pins the threshold so it cannot
drift.

Where the evidence is weak, the report says so: with n=30 one question is worth
0.0333 R@3, so Stages 2, 4 and 6 are all decided inside the noise floor and their
winners should be read as "no worse than the alternatives".

---

## Licence and attribution

Coursework for the *zero-to-genai-engineer* M10 RAG capstone. Source documents are
real published Delta Air Lines documents and SEC filings, included for academic
evaluation only.
