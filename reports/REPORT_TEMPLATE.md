# Evaluation Report — Delta Air Lines Customer Support Assistant

**Group 7 · M10 RAG Capstone**

Every table below contains numbers measured on this project's own 30-question
evaluation set. Tables are injected directly from the artefacts written by the
ablation scripts (`reports/results/*.md`) by `scripts/build_report.py`, so nothing
here is retyped and nothing can drift from what was actually run.

Reproduce end to end:

```bash
.venv/bin/python -m pytest                                # 30 metric unit tests
PYTHONPATH=src .venv/bin/python -m delta_rag.evalset      # validate the labels
.venv/bin/python scripts/stage0_xbrl_representation.py    # financial-route text
.venv/bin/python scripts/stage0b_boilerplate.py           # front matter removal
.venv/bin/python scripts/run_retrieval_ablations.py       # Stages 1-7
.venv/bin/python scripts/stage0c_routing.py               # routing policy
.venv/bin/python scripts/calibrate_guardrail.py           # abstain threshold
.venv/bin/python scripts/stage8_generate.py               # Stage 8 generations
.venv/bin/python scripts/stage8_evaluate.py               # RAGAS + DeepEval
.venv/bin/python scripts/run_acceptance_tests.py          # graded acceptance tests
.venv/bin/python scripts/build_report.py                  # rebuild this document
```

---

## 0. How ground truth was defined, and why it dictated the rest

Stage 1 varies the PDF parser; Stage 2 varies the chunking strategy. Both change
chunk boundaries and therefore chunk IDs, so ground truth recorded as "chunk #47 is
correct" is destroyed by the first two experiments — and every table after them
would be measuring noise while looking authoritative.

Labels are therefore recorded at a level invariant to both:

| Field | Meaning |
|---|---|
| `gold_doc` | stable document id |
| `gold_locators` | PDF page (`p17`) or XBRL concept (`us-gaap:NetIncomeLoss`) |
| `gold_span` | verbatim answer-bearing text |

Graded relevance is then derived mechanically: **3** the chunk contains the gold
span, **2** right document at a gold locator, **1** right document elsewhere,
**0** wrong document.

**One decision deserves flagging.** R@1/R@3/MRR@10 count relevance at **grade ≥ 2**,
not grade 3. pypdf extracts `those` as `t hose` on contract page 4, so one gold span
becomes unrecoverable from its output. Requiring grade 3 would score that
mangled-but-correct page as a total retrieval miss — overstating the damage and
double-counting it against a parser already penalised in the Stage 1 clean-text
column. At grade ≥ 2 the defect costs NDCG@3 (graded) and leaves R@3 (binary)
intact: counted once, in one place. A unit test pins the threshold so it cannot
drift.

The evaluation set is **30 questions** (12 passenger / 11 cargo / 7 financial;
14 keyword / 16 semantic), built before any pipeline decision, with all 30 gold
spans machine-verified against the PDFs. The keyword/semantic split exists because
Stage 5 requires it, so it had to be authored up front rather than reconstructed.

---

## Stage 0 — Representing the XBRL financial route

Not one of the eight required ablations, but the largest single quality change in
the project, and a case where the measurement contradicted my hypothesis.

{{stage0_xbrl}}

---

## Stage 0b — Front matter and running headers

Found by debugging a **failing required acceptance test**, not by reading the PDFs.

{{stage0b_boilerplate}}

---

## Stage 1 — Parsing strategy

{{stage1_parsing}}

---

## Stage 2 — Chunking strategy

{{stage2_chunking}}

**On the margin.** The top four configurations tie at R@3 = 0.9000 and are separated
by less than 0.01 MRR@10 — under a third of one question out of 30. `recursive
500/50` is selected as the winner on the metric priority (R@3, then MRR@10, then
NDCG@3), but the honest reading is that **chunking strategy is not a decisive lever
on this corpus** once front matter is removed. Both tariffs are written as short
numbered rules, so almost any sane splitter lands chunk boundaries near rule
boundaries. `rule_aware` — a custom strategy that splits on the documents' own
`RULE 20:` / `G14` headings — was built expecting it to win, and it did not, which
is a useful negative result about how much document-specific engineering this corpus
actually rewards.

---

## Stage 3 — Embedding model

{{stage3_embeddings}}

---

## Stage 4 — Vector database

{{stage4_vectordb}}

---

## Stage 5 — Dense vs sparse vs hybrid

This is the strongest evidence in the study, and the table this group's
`REQUIREMENT.md` specifically asks to be broken out by target document.

{{stage5_retrieval_mode}}

**Why hybrid is genuinely justified here.** The per-route columns show two
retrievers with *opposite* weaknesses, not one dominant retriever:

- **Sparse is perfect on passenger (1.0000) and collapses on financial (0.7143).**
  Passenger questions quote tariff vocabulary almost verbatim, which is BM25's home
  ground.
- **Dense is the reverse** — strong on financial (0.8571), weakest on cargo
  (0.5455).

Hybrid does not split the difference; on semantic queries it **beats both**
(0.9375 vs 0.7500 dense and 0.8750 sparse) and it takes the better of the two on
every route. That is the argument for paying the complexity cost of running two
retrievers and fusing them. Had one mode dominated every column, fusion would have
been unjustifiable complexity.

---

## Stage 6 — Hybrid merge method and weighting

{{stage6_fusion}}

**On the margin.** Weighted α = 0.5 leads RRF on R@3 by 0.0333 — which on 30
questions is *exactly one question*. That is the resolution limit of this eval set,
so the win is real but weakly evidenced. Two further cautions are worth recording:
the α sweep is **non-monotonic** (0.8667 → 0.8667 → 0.9333 → lower again), which is
what overfitting to a small set looks like rather than a tuned optimum; and RRF
carries no hyperparameter and is invariant to score scale. **RRF remains the safer
production choice**, and if this eval set grew to 100+ questions I would expect the
gap to close. The measured winner is reported as measured, with the caveat attached.

---

## Stage 7 — Reranking

{{stage7_reranking}}

---

## Stage 8 — LLM for generation

{{stage8_generation}}

---

## Supporting decisions, also measured

### Routing policy (30% of the grade)

{{stage0c_routing}}

### Guardrail abstain threshold

{{guardrail_calibration}}

---

## Acceptance test results

{{acceptance_tests}}

---

## Final configuration

| Stage | Choice | Deciding evidence |
|---|---|---|
| Parsing | **PyMuPDF** | 100% gold-span recovery vs 96.7% for pypdf; 15× faster than pdfplumber |
| Chunking | **recursive, 500 tokens, 50 overlap** | R@3 0.9000 — tied at the top; margin under one question |
| Embeddings | **BAAI/bge-small-en-v1.5** | R@3 0.9000 vs 0.7667 MiniLM, at 1.8 ms/query |
| Vector store | **FAISS** | R@3 parity with Chroma; far faster index build at this size |
| Retrieval mode | **Hybrid** | R@3 0.9000 vs 0.7667 dense / 0.8667 sparse; wins every route |
| Fusion | **Weighted linear, α = 0.5** | R@3 0.9333 vs 0.9000 RRF — one question, caveat above |
| Reranking | **None** | Cross-encoder *lowered* NDCG@3 and cost 120 ms |
| Routing | **Top-1 chunk** | 1.0000 on eval questions; beat both richer policies |
| Generation | see Stage 8 | fixed retrieval, generator varied |

### Synthesis

The headline retrieval result is **R@3 = 0.9333, R@1 = 0.7333, MRR@10 = 0.8319,
NDCG@3 = 0.8849, routing accuracy = 1.0000** on the 30-question set, and **5/5 on
the required acceptance tests** (10/11 including four extra guardrail probes and two
cross-route traps).

What actually moved the needle was **not** the eight canonical stages. Ranked by
measured impact:

1. **Vocabulary bridging on the financial route** — sparse financial R@3 went
   0.1429 → 0.7143 (5×) and dense 0.5714 → 0.8571. Nothing in Stages 1-7 came close.
2. **Removing 2 of 45 pages of front matter** — fixed a *failing required acceptance
   test* and lifted sparse R@3 0.7667 → 0.8333.
3. **Hybrid retrieval** — +0.1666 R@3 over dense, +0.0333 over sparse, and the only
   configuration that is strong on all three routes.
4. **Embedding model** — bge-small over MiniLM was worth +0.1333 R@3 for 0.4 ms.
5. **Chunking, fusion method, vector store** — differences at or inside the noise
   floor of a 30-question set.
6. **Reranking** — actively harmful.

The uncomfortable conclusion is that the two largest wins came from **looking at what
the retriever actually returned**, while the canonical hyperparameter sweeps mostly
produced ties. Both were found by investigating a failure, not by tuning: the
financial defect surfaced because dense and sparse failed *identically* (a tell that
the corpus, not the config, was at fault), and the boilerplate defect surfaced
because an acceptance test failed while every aggregate metric looked healthy.

### Where the evidence is weak, stated plainly

- **n = 30 is coarse.** One question is worth 0.0333 R@3, so Stage 2, Stage 4 and
  Stage 6 are all decided inside the noise floor. Those winners should be read as
  "no worse than the alternatives", not "better".
- **The alias map is the single largest lever and the most exposed to
  test-fitting.** It was written from 10-K statement conventions rather than from the
  eval questions, and is measured in isolation (Stage 0, variant B vs C) so it can be
  discounted — but a held-out financial question set is the only real check.
- **Both judges are 7-8B local models.** They are cheap and reproducible, but weaker
  than a frontier judge. Where RAGAS and DeepEval disagree, the correct conclusion is
  that neither number is reliable.
- **R2 is a known misroute.** "Liability limit for a damaged shipment" routes to the
  passenger contract, whose baggage-liability clause is genuinely the most similar
  text in the corpus and even quotes a dollar figure. The defined-term prior fixes it
  but breaks R1, so no fix was shipped on the strength of one case.

### What I would try next

Tied to the M06 "when RAG fails" material, in priority order:

1. **A held-out financial eval set** using vocabulary absent from the alias map. This
   is the biggest threat to the project's headline claim and the cheapest to test.
2. **LLM-based groundedness instead of a cosine floor.** Acceptance probe G4 showed
   the *model* refusing correctly at high retrieval confidence — it can read the
   context and see the answer is absent, which cosine similarity fundamentally
   cannot. The threshold gate is the weakest component: the answerable and
   unanswerable score distributions overlap, so no threshold separates them.
3. **A domain-tuned reranker.** The failure was diagnosed as MS MARCO training data
   not transferring to legal tariff prose. Fine-tuning a cross-encoder on
   tariff-style pairs would test that diagnosis directly, rather than assuming it.
4. **Multi-hop questions.** Every eval question is answerable from one chunk. Real
   questions like "I was bumped *and* my bag was damaged — what am I owed?" need
   Rule 20 *and* Rule 17 simultaneously, and the current top-k flat retrieval has no
   mechanism for composing two rules.
5. **Cross-route answers.** Routing currently commits to exactly one business line to
   prevent contamination. A shipper asking a question that legitimately spans cargo
   rules *and* the financial filings cannot be served at all — the anti-contamination
   fix has a cost that this eval set never probes.
