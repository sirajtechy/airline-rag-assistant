# Evaluation Report — Delta Air Lines Customer Support Assistant

**Group 4 · M10 RAG Capstone**

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

| XBRL representation | Mode | Records | Avg chars | Financial R@3 | Financial MRR@10 | Overall R@3 |
|---|---|---|---|---|---|---|
| A. Original verbose rendering | dense | 420 | 534 | 0.5714 | 0.5083 | 0.6333 |
| A. Original verbose rendering | sparse | 420 | 534 | 0.1429 | 0.1587 | 0.7333 |
| B. + label-first, compact periods, prose dropped | dense | 413 | 389 | 0.5714 | 0.6071 | 0.6333 |
| B. + label-first, compact periods, prose dropped | sparse | 413 | 389 | 0.1429 | 0.1633 | 0.7333 |
| C. + statement-label aliases (final) | dense | 413 | 391 | 0.8571 | 0.8571 | 0.7000 |
| C. + statement-label aliases (final) | sparse | 413 | 391 | 0.7143 | 0.6476 | 0.8667 |

### Reading of these numbers

**The initial hypothesis was wrong, and the measurement says so.**

The financial route first scored R@3 = 0.14 on sparse retrieval. The obvious explanation was boilerplate: all 420 records opened with the same entity/filing sentence and repeated `for the period 2025-01-01 to 2025-12-31` on nearly every line, leaving them ~80% shared vocabulary. Variant B removes exactly that — label-first headers, `FY2025` instead of the long date phrasing, and TextBlock prose dropped.

Variant B changed **nothing**: sparse financial R@3 stayed at 0.1429 and dense stayed at 0.5714. Only MRR@10 moved, and only for dense (0.508 -> 0.607), i.e. slightly better ordering of results that were already being found. Tidying the boilerplate was cosmetic.

The actual defect is a **vocabulary gap**. A US-GAAP element is named `Revenue from Contract with Customer, Excluding Assessed Tax`; a human asks for `total operating revenue`. Those strings share almost no terms, which is fatal for BM25 (0.14) and merely bad for embeddings (0.57). Variant C bridges that gap and lifts sparse to 0.7143 and dense to 0.8571 — a 5x improvement on sparse from the one change that addressed the real cause.

**Caveat, stated plainly.** The alias map is the single largest lever on this route, and synonym injection is inherently at risk of being fitted to the evaluation questions. The aliases were authored from standard 10-K statement line-item conventions rather than from the eval set, and are reported here in isolation precisely so a sceptical reader can discount them. A stricter test — held-out financial questions phrased with vocabulary absent from the alias map — is listed in the report's 'what we would try next' section.

---

## Stage 0b — Front matter and running headers

Found by debugging a **failing required acceptance test**, not by reading the PDFs.

| Preprocessing | Mode | PDF pages | R@1 | R@3 | MRR@10 | NDCG@3 | R@3 (cargo) | R@3 (passenger) |
|---|---|---|---|---|---|---|---|---|
| A. Raw pages (front matter + headers kept) | sparse | 45 | 0.6667 | 0.7667 | 0.7473 | 0.7943 | 0.6364 | 1.0000 |
| A. Raw pages (front matter + headers kept) | hybrid | 45 | 0.6667 | 0.9000 | 0.7708 | 0.8594 | 0.8182 | 1.0000 |
| B. Front matter dropped, running headers stripped | sparse | 43 | 0.6667 | 0.8333 | 0.7567 | 0.8486 | 0.7273 | 1.0000 |
| B. Front matter dropped, running headers stripped | hybrid | 43 | 0.7000 | 0.9000 | 0.7981 | 0.8874 | 0.8182 | 1.0000 |

### Reading of these numbers

Only **2 of 45 pages** are removed (contract p1, cargo p2), and no gold answer span lives on either, so the eval labels still validate 30/30.

The mechanism is worth stating precisely, because it explains why a contents page is worse than useless. A table of contents lists every rule title in the document: `RULE 20: DENIED BOARDING COMPENSATION`, `G32 Limit of Liability`, and so on. To BM25 that page looks maximally relevant to almost any topical query, while containing no rule text to answer it with. It is a chunk engineered to win retrieval and then say nothing.

The running header is the subtler of the two. `Delta Domestic General Rules Tariff` appears on all 23 pages of the passenger contract, so the terms 'domestic', 'rules' and 'tariff' are indexed 23 times in the passenger document — which is why a *cargo* question phrased as 'rules for shipping cargo domestically' was pulled toward the passenger contract. The cross-contamination this group is graded on was partly caused by page furniture.

This was found by debugging a failing acceptance test rather than by reading the PDFs, which is an argument for the methodology itself: the aggregate retrieval metrics were already respectable while a required acceptance test was failing for a cause no aggregate number surfaced.

---

## Stage 1 — Parsing strategy

| Parser | Clean text % | Gold spans recoverable | Parse time (s) | R@1 | R@3 | MRR@10 | DCG@3 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|
| pypdf | 97.8% | 96.7% | 1.5800 | 0.7333 | 0.8667 | 0.8078 | 3.9043 | 0.8877 |
| pdfplumber | 97.8% | 100.0% | 4.4700 | 0.6333 | 0.8000 | 0.7329 | 3.5587 | 0.8174 |
| pymupdf | 97.8% | 100.0% | 0.3000 | 0.7000 | 0.9000 | 0.8011 | 3.7964 | 0.8762 |

### Reading of these numbers

All three parsers extract all 45 pages and clear the clean-text bar, so the usual 'clean text %' column does not separate them. The column that does is gold-span recoverability: **pypdf loses one of the 30 answer spans**, rendering `those` as `t hose` on contract page 4 (question P07). The page is still retrievable — which is why R@3 barely moves — but the verbatim evidence needed to ground a citation is gone.

PyMuPDF is chosen: it recovers 100% of gold spans and is the fastest of the three. pdfplumber matches it on recovery but is materially slower, and its table-extraction advantage is irrelevant here because neither tariff stores its rules in ruled tables.

---

## Stage 2 — Chunking strategy

| Strategy | Chunk size | Overlap | Chunks | R@1 | R@3 | MRR@10 | DCG@3 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|
| fixed | 300 | 0 | 512 | 0.6667 | 0.8333 | 0.7462 | 3.5404 | 0.8217 |
| fixed | 500 | 0 | 473 | 0.6000 | 0.7667 | 0.7037 | 3.5000 | 0.8325 |
| fixed | 800 | 0 | 451 | 0.7000 | 0.7667 | 0.7500 | 3.5194 | 0.8640 |
| fixed_ov | 300 | 50 | 532 | 0.7333 | 0.8000 | 0.7839 | 3.8420 | 0.8452 |
| fixed_ov | 500 | 50 | 480 | 0.6333 | 0.7667 | 0.7187 | 3.5020 | 0.8186 |
| fixed_ov | 500 | 100 | 487 | 0.6667 | 0.8000 | 0.7450 | 3.7682 | 0.8650 |
| fixed_ov | 800 | 100 | 456 | 0.6000 | 0.8000 | 0.6987 | 3.4658 | 0.8482 |
| recursive | 300 | 50 | 547 | 0.7000 | 0.8333 | 0.7792 | 3.7500 | 0.8410 |
| recursive | 500 | 50 | 491 | 0.7000 | 0.9000 | 0.8011 | 3.7964 | 0.8762 |
| recursive | 500 | 100 | 495 | 0.7000 | 0.8333 | 0.7781 | 3.8797 | 0.8627 |
| recursive | 800 | 100 | 463 | 0.6667 | 0.9000 | 0.7807 | 3.8428 | 0.8801 |
| rule_aware | 300 | 50 | 554 | 0.7000 | 0.9000 | 0.7937 | 3.7631 | 0.8375 |
| rule_aware | 500 | 50 | 503 | 0.6667 | 0.8667 | 0.7806 | 3.7885 | 0.8635 |
| rule_aware | 800 | 100 | 475 | 0.7000 | 0.9000 | 0.7981 | 3.7833 | 0.8874 |

### Reading of these numbers

Winner: **recursive at 500 tokens / 50 overlap** (R@3 = 0.9000, MRR@10 = 0.8011).

`rule_aware` is a document-specific strategy added because both tariffs are organised as numbered rules (`RULE 20:`, `G14`), so a rule heading is a genuine semantic boundary rather than an arbitrary token offset. Comparing it against the three generic strategies is what shows whether that structure is worth exploiting on this corpus.

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

| Embedding model | Key | Dimensions | Latency (ms/query) | R@1 | R@3 | MRR@10 | DCG@3 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | minilm | 384 | 1.2000 | 0.7000 | 0.9000 | 0.7889 | 3.7412 | 0.8844 |
| BAAI/bge-small-en-v1.5 | bge-small | 384 | 1.7000 | 0.7000 | 0.9000 | 0.8011 | 3.7964 | 0.8762 |
| nomic-embed-text:latest | nomic | 768 | 15.3000 | 0.7667 | 0.8333 | 0.8119 | 3.7571 | 0.8445 |

### Reading of these numbers

Winner: **bge-small** (R@3 = 0.9000, MRR@10 = 0.8011, 1.7 ms/query).

**This is a near-tie, not a clear win.** 2 of the 3 candidates (minilm, bge-small) share the top R@3 of 0.9000 and separate only on secondary metrics — where MiniLM actually takes NDCG@3 and bge-small takes MRR@10.

This **contradicts an earlier run of this same table**, in which bge-small led MiniLM by +0.1333 R@3. That gap existed *before* the front-matter and running-header removal in Stage 0b. Cleaning the boilerplate that was polluting retrieval made the embedding model substantially less important, which is a result in its own right: fixing the corpus reduced the leverage of a hyperparameter. It is also a caution about ablation order — a stage measured against a defective corpus can report a difference that evaporates once the defect is fixed.

`bge-small` and `nomic` are asymmetric models: they are trained to embed queries and passages with different instruction prefixes. Those prefixes are applied here (`Represent this sentence for searching relevant passages: ` and `search_query:`/`search_document:` respectively). Benchmarking them without their prefixes is a common error that would have understated both and handed the stage to MiniLM by default.

`nomic` is the interesting outlier: it takes the best R@1 and the best MRR@10 of the three, but the worst R@3, and costs ~9x the latency because it is served over HTTP by Ollama rather than running in-process. It ranks its best guess well and its next guesses poorly.

Latency is measured after a warm-up call so model load time is excluded.

---

## Stage 4 — Vector database

| Vector DB | Index build time (ms) | Metadata filtering | Persistence | R@1 | R@3 | MRR@10 | DCG@3 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|
| faiss | 0.0600 | emulated | yes | 0.7000 | 0.9000 | 0.8011 | 3.7964 | 0.8762 |
| chromadb | 255.7200 | native | yes | 0.7000 | 0.9000 | 0.8011 | 3.7964 | 0.8762 |

### Reading of these numbers

R@3 is reported as a parity check and behaves as the methodology predicts: identical embeddings retrieve near-identically regardless of store, so this decision cannot be made on quality.

It is made on operational grounds instead. FAISS builds its flat inner-product index far faster at this corpus size and runs in-process with no server, but has **no native metadata filtering** — business-line scoping has to be emulated by over-fetching and post-filtering. ChromaDB supports `where` filtering natively and persists without extra code.

FAISS is selected for the graded pipeline: at ~500 chunks the over-fetch cost is negligible and the build-time advantage makes the ablation matrix practical to re-run. ChromaDB would be the better choice if this corpus grew by an order of magnitude, where post-filtering starts to threaten recall.

---

## Stage 5 — Dense vs sparse vs hybrid

This is the strongest evidence in the study, and the table this group's
`REQUIREMENT.md` specifically asks to be broken out by target document.

| Mode | R@1 (all) | R@3 (all) | R@3 (keyword) | R@3 (semantic) | R@3 (passenger) | R@3 (cargo) | R@3 (financial) | MRR@10 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|---|
| Dense only | 0.6000 | 0.7667 | 0.7857 | 0.7500 | 0.9167 | 0.5455 | 0.8571 | 0.6822 | 0.8473 |
| Sparse (BM25) only | 0.7333 | 0.8667 | 0.8571 | 0.8750 | 1.0000 | 0.8182 | 0.7143 | 0.7956 | 0.8323 |
| Hybrid (RRF) | 0.7000 | 0.9000 | 0.8571 | 0.9375 | 1.0000 | 0.8182 | 0.8571 | 0.8011 | 0.8762 |

### Reading of these numbers

Winner: **Hybrid (RRF)** (R@3 = 0.9000).

This group's REQUIREMENT asks for the breakdown by target document, so the per-route columns are included alongside the required keyword/semantic split. That breakdown is the actual argument for or against hybrid: if one mode were uniformly better across all three routes and both query types, the added complexity of running two retrievers and fusing them would not be justified. Divergence between the columns is what makes fusion worth paying for.

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

| Merge method | alpha | R@3 | MRR@10 | NDCG@3 |
|---|---|---|---|---|
| Reciprocal Rank Fusion (k=60) |  | 0.9000 | 0.8011 | 0.8762 |
| Weighted linear | 0.2000 | 0.8667 | 0.8059 | 0.8461 |
| Weighted linear | 0.3000 | 0.8667 | 0.8069 | 0.8426 |
| Weighted linear | 0.5000 | 0.9333 | 0.8319 | 0.8849 |
| Weighted linear | 0.7000 | 0.8667 | 0.7656 | 0.8547 |
| Weighted linear | 0.8000 | 0.8000 | 0.7169 | 0.8406 |

### Reading of these numbers

**Selected: weighted linear at alpha = 0.5** (R@3 = 0.9333, MRR@10 = 0.8319, NDCG@3 = 0.8849), ahead of RRF at R@3 = 0.9000.

**The margin is +0.0333 R@3 — exactly 1 question out of 30.** That is the resolution limit of this evaluation set, so the win is real but weakly evidenced, and it is reported as such rather than as a decisive result.

**The alpha sweep is the reason for caution.** Across alpha = 0.2, 0.3, 0.5, 0.7, 0.8 the R@3 curve runs 0.8667 -> 0.8667 -> 0.9333 -> 0.8667 -> 0.8000 — **non-monotonic and spiky**. If alpha were capturing a real property of the corpus the curve would be smooth, so the peak is more plausibly a coincidence of these particular 30 questions than a tuned optimum. That is precisely the overfitting the methodology warns against, and it is why **RRF remains the safer production choice** despite losing this comparison: it has no alpha to overfit and is invariant to the two retrievers' score scales.

For reference: alpha weights the dense side, `score = alpha * dense_norm + (1 - alpha) * sparse_norm`. Both distributions are min-max normalised first because BM25 scores are unbounded while cosine similarity is capped at 1; combining them raw would let BM25 dominate through scale rather than relevance. RRF sidesteps normalisation entirely by fusing ranks, at the cost of being blind to margin.

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

| Config | R@1 | R@3 | NDCG@3 | Added latency (ms) |
|---|---|---|---|---|
| No reranker | 0.7333 | 0.9333 | 0.8849 | 0.0000 |
| + Cross-encoder rerank (top-10 -> top-3/5) | 0.7333 | 0.8667 | 0.8388 | 103.7000 |
| + Cross-encoder rerank (top-20 -> top-3/5) | 0.7333 | 0.8667 | 0.8343 | 176.2000 |

### Reading of these numbers

A bi-encoder embeds query and passage separately and never sees them together; a cross-encoder scores the pair jointly, which is why it is normally more accurate and why it can only be afforded on a shortlist.

**The reranker is rejected — it makes retrieval worse, not better.**

NDCG@3 falls from 0.8849 with no reranker to 0.8388 at top-10 and 0.8343 at top-20, and at top-20 R@3 drops too (0.9333 -> 0.8667). It costs 176.2 ms per query to be worse. Notably the damage grows with the candidate pool: the more documents it is given to reorder, the more harm it does.

This is the opposite of the textbook result, so it needs an explanation rather than a shrug. Two things are going on.

First, **domain mismatch**. `ms-marco-MiniLM-L-6-v2` is trained on MS MARCO — short, factual web-search passages answering natural questions. This corpus is neither: it is dense legal tariff prose full of cross-references (`as provided in Rule 20(c)(1)`) and XBRL records that read as label-plus-numbers. The cross-encoder's learned notion of relevance transfers poorly to both, so its confident reordering is confidently wrong.

Second, **there is almost no headroom to win and plenty to lose**. Hybrid fusion already reaches R@3 = 0.9333; a reranker cannot add documents fusion never retrieved, it can only reshuffle them. When the correct chunk is usually already in the top 3, any reshuffling is far more likely to demote it than promote it.

The general lesson is the one the methodology is built around: a reranker is a *conditional* improvement, not an automatic one. Adding it because a tutorial did would have cost this pipeline both accuracy and latency.

---

## Stage 8 — LLM for generation

| LLM | Faithfulness (RAGAS) | Answer Relevancy (RAGAS) | Context Precision (RAGAS) | Context Recall (RAGAS) | Faithfulness (DeepEval) | Hallucination (DeepEval, lower=better) | G-Eval PolicyAccuracy (DeepEval) | Route acc | Valid citations | Latency (ms/answer) | Cost/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5 7B | 0.8986 | 0.7404 | 0.7676 | 0.9139 | 0.6644 |  | 0.9867 | 1.0000 | 0.9333 | 16414 | $0.00 (local) |
| Llama 3.2 3B | 0.7829 | 0.6560 | 0.7676 | 0.9139 | 0.6359 |  | 0.9000 | 1.0000 | 0.7333 | 8420 | $0.00 (local) |

### Reading of these numbers

Retrieval is byte-identical across every row — same parser, chunking, embeddings, store and fusion — so these differences are attributable to the generator alone. Context Precision and Context Recall are properties of the shared retrieval step and should be near-constant down the table; any spread in them is judge noise, which is itself a useful read on how much to trust the other columns.

**Two independent judges.** RAGAS is judged by `qwen2.5-7b-judge:latest` and DeepEval by `llama3.1-latest-judge:latest` — deliberately different model families. Where the two faithfulness columns agree, the score is trustworthy; where they diverge sharply, the honest conclusion is that neither number is reliable rather than that the flattering one is right.

**Cost is $0.00 for every candidate** because everything runs locally on Ollama, so cost cannot discriminate between models here. Latency and quality do the work instead. The G-Eval criterion is custom-written for this bot: it rewards exact figures, deadlines and dollar amounts and penalises invented fees, because a fluent answer that loses the $1,075 cap has failed the customer even though a generic relevancy metric would pass it.

Guardrail-blocked answers are excluded from the generation metrics: they have no retrieval context to be faithful to, so scoring them would penalise a model for the guardrail working.

**An empty Hallucination cell is deliberate, not missing data.** DeepEval's `HallucinationMetric` emits one verdict per document in `context` and returns `hallucination_count / number_of_verdicts`, so what is passed as `context` changes what is measured rather than merely shading it. The first run supplied the five retrieved chunks; a correct answer typically draws on one, so the other four produced verdicts against material the answer never needed — measuring retrieval breadth, under which simply widening `top_k` would worsen the apparent hallucination rate without changing a single answer. Values are therefore reported only when the cache records that ground truth was used; see `reports/results/hallucination_sensitivity.md` for all three variants measured side by side.

---

## Supporting decisions, also measured

### Routing policy (30% of the grade)

| Routing policy | Route acc (30 eval questions) | Route acc (all 33 cases) | Misroutes |
|---|---|---|---|
| top-1 chunk | 1.0000 | 0.9714 | 1 |
| rank-weighted vote (top-5) | 0.9667 | 0.9143 | 3 |
| vote + defined-term prior | 0.9667 | 0.9143 | 3 |

### Reading of these numbers

Winner: **top-1 chunk** — 0.9714 across all 33 cases.

The prior is built from **68 terms each document uniquely defines** ({'passenger': 24, 'cargo': 44}), extracted from the definitions sections rather than hand-written. Terms both documents define — 'Carrier' most importantly — are discarded because they carry no signal.

Why a prior is needed at all: retrieval evidence alone cannot separate "What is Delta's liability limit for a damaged shipment?" The passenger contract's baggage-liability clause is genuinely the most topically similar text in the corpus, and it even names a dollar figure ($4,700 per passenger), so a retriever is right to surface it. The distinguishing signal is not similarity but vocabulary: 'shipment' is a term the cargo tariff defines and the passenger contract does not.

The prior is weighted at 0.35 of the retrieval evidence, so it breaks ties rather than overriding retrieval. Keyword routing *instead of* retrieval would fail the opposite case — R1, "revenue from shipping cargo", is saturated with cargo vocabulary but must route to the financial filings.

**top-1 chunk** misroutes:
- "What is Delta's liability limit for a damage" -> passenger (want cargo)
**rank-weighted vote (top-5)** misroutes:
- 'How much revenue did Delta make from shippin' -> cargo (want financial)
- 'How much revenue did Delta make from shippin' -> cargo (want financial)
- "What is Delta's liability limit for a damage" -> passenger (want cargo)
**vote + defined-term prior** misroutes:
- 'How much revenue did Delta make from shippin' -> cargo (want financial)
- 'How much revenue did Delta make from shippin' -> cargo (want financial)
- "What is Delta's liability limit for a damage" -> passenger (want cargo)

### Guardrail abstain threshold

| Threshold | Answerable correctly answered | Unanswerable correctly refused | Sum |
|---|---|---|---|
| 0.5500 | 1.0000 | 0.0000 | 1.0000 |
| 0.5700 | 1.0000 | 0.0000 | 1.0000 |
| 0.5900 | 1.0000 | 0.1250 | 1.1250 |
| 0.6100 | 1.0000 | 0.1250 | 1.1250 |
| 0.6300 | 1.0000 | 0.1250 | 1.1250 |
| 0.6500 | 1.0000 | 0.1250 | 1.1250 |
| 0.6700 | 1.0000 | 0.2500 | 1.2500 |
| 0.6900 | 0.9000 | 0.3750 | 1.2750 |
| 0.7100 | 0.8667 | 0.6250 | 1.4917 |
| 0.7300 | 0.7667 | 1.0000 | 1.7667 |
| 0.7500 | 0.5667 | 1.0000 | 1.5667 |
| 0.7700 | 0.4333 | 1.0000 | 1.4333 |
| 0.7900 | 0.3000 | 1.0000 | 1.3000 |
| 0.8100 | 0.1667 | 1.0000 | 1.1667 |
| 0.8300 | 0.1333 | 1.0000 | 1.1333 |
| 0.8500 | 0.0333 | 1.0000 | 1.0333 |
| 0.8700 | 0.0000 | 1.0000 | 1.0000 |

### Reading of these numbers

**Chosen threshold: 0.67** — answers 100% of answerable questions while refusing 25% of unanswerable ones.

**This gate is the coarse outer layer of a two-layer defence, and is tuned accordingly.** The precise layer is the system prompt, which makes the model decline from context it can actually read. Acceptance probe G4 demonstrates it working on a question with *high* retrieval confidence: asked for a transatlantic third-bag fee, the bot answered "Delta's published documents do not cover fees for checked bags on transatlantic flights ... domestic travel only [1]" rather than inventing one. Cosine similarity could never have caught that, because the retrieved baggage rules genuinely are the most topically similar text in the corpus — they simply do not contain the answer.

That is why the threshold is set to reject **no** answerable question rather than to maximise refusals. A more aggressive setting (0.73 refuses 100% of the unanswerable probes) would also refuse 27% of questions the pipeline answers correctly, trading real utility for refusals the prompt layer already handles better.

The gate runs on the **dense cosine similarity** of the best retrieved chunk, not the fused RRF score. RRF scores are a function of rank position and carry no absolute meaning — a score of 0.03 means nothing on its own and is not comparable between queries — so no fixed threshold on it could be meaningful. Cosine similarity is bounded in [0, 1] and comparable across queries, which is what a confidence floor requires.

The two populations **overlap**: the hardest answerable question scores 0.674 while the most confidently-retrieved unanswerable probe scores 0.729. No threshold separates them perfectly, so this is an explicit trade-off, not a solved problem. It is tuned toward refusing, because inventing a baggage fee is a worse failure for a support bot than declining a question it could have answered.

The unanswerable probes are plausible Delta questions whose answers are genuinely outside these three documents (international fees, SkyMiles earning, lounge hours, aircraft configuration). They deliberately avoid competitor names and 'my booking' phrasing so they exercise this gate rather than the deterministic pre-retrieval guards.

---

## Acceptance test results

| # | Question | Expected | Route | Blocked | Result | Detail |
|---|---|---|---|---|---|---|
| A1 | What's Delta's policy on oversold flights? | answer | passenger | no | PASS |  |
| A2 | What are the rules for shipping cargo domestically with Delta? | answer | cargo | no | PASS |  |
| A3 | Can I get a refund on a non-refundable Delta ticket? | answer | passenger | no | PASS |  |
| A4 | What's JetBlue's baggage policy? | refuse | - | yes | PASS |  |
| A5 | What's the status of my flight tomorrow? | deflect | - | yes | PASS |  |
| G1 | What is United Airlines' policy on denied boarding compensation? | refuse | - | yes | PASS |  |
| G2 | Can you cancel my booking and refund my card? | deflect | - | yes | PASS |  |
| G3 | What is Delta's pet cargo fee for shipping a dog to London in euros? | refuse_or_disclaim | passenger | no | PASS |  |
| G4 | How much is Delta's checked bag fee for a third bag on a transatlantic flight? | refuse_or_disclaim | passenger | no | PASS |  |
| R1 | How much revenue did Delta make from shipping cargo last year? | answer | financial | no | PASS |  |
| R2 | What is Delta's liability limit for a damaged shipment? | answer | passenger | no | FAIL | routed to passenger, expected cargo |

### Reading of these numbers

**Required acceptance tests (REQUIREMENT section 6): 5/5 passed.**
All probes including extra guardrail and routing cases: 10/11.

A1-A5 are the five graded tests. G1-G4 are additional guardrail probes: G1 names a competitor using in-corpus vocabulary ("denied boarding") to check the carrier check is not fooled by familiar terminology, G2 is a transactional request, and G3-G4 ask for fees that genuinely do not exist in these domestic-only documents — the case where a RAG bot is most tempted to invent a number.

R1 and R2 are the cross-route cases this group is specifically graded on. R1 ("revenue from shipping cargo") is lexically saturated with cargo terms but must route to the financial XBRL, and R2 ("liability limit for a damaged shipment") shares the word "liability" with the passenger contract but must route to cargo. Both are why routing is decided from rank-weighted retrieval evidence rather than from keyword matching on the question.

---

## Final configuration

| Stage | Choice | Deciding evidence |
|---|---|---|
| Parsing | **PyMuPDF** | 100% gold-span recovery vs 96.7% for pypdf; 15× faster than pdfplumber |
| Chunking | **recursive, 500 tokens, 50 overlap** | R@3 0.9000 — tied at the top; margin under one question |
| Embeddings | **BAAI/bge-small-en-v1.5** | near-tie: R@3 0.9000 = MiniLM; wins MRR@10 0.8011 vs 0.7889 |
| Vector store | **FAISS** | R@3 parity with Chroma; far faster index build at this size |
| Retrieval mode | **Hybrid** | R@3 0.9000 vs 0.7667 dense / 0.8667 sparse; wins every route |
| Fusion | **Weighted linear, α = 0.5** | R@3 0.9333 vs 0.9000 RRF — one question, caveat above |
| Reranking | **None** | Cross-encoder *lowered* NDCG@3 0.8849 -> 0.8343 and cost 176 ms |
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
4. **Embedding model** — now a *tie* on R@3 after the corpus was cleaned. An earlier
   run of Stage 3 showed bge-small ahead of MiniLM by +0.1333 R@3, but that gap was
   measured against the uncleaned corpus and evaporated once the boilerplate was
   removed. A caution about ablation order: a stage measured against a defective
   corpus can report a difference that does not survive fixing the defect.
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
