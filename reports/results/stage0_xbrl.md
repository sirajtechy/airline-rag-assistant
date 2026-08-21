# Stage 0 — XBRL financial-route representation

| XBRL representation | Mode | Records | Avg chars | Financial R@3 | Financial MRR@10 | Overall R@3 |
|---|---|---|---|---|---|---|
| A. Original verbose rendering | dense | 420 | 534 | 0.5714 | 0.5083 | 0.6333 |
| A. Original verbose rendering | sparse | 420 | 534 | 0.1429 | 0.1587 | 0.7333 |
| B. + label-first, compact periods, prose dropped | dense | 413 | 389 | 0.5714 | 0.6071 | 0.6333 |
| B. + label-first, compact periods, prose dropped | sparse | 413 | 389 | 0.1429 | 0.1633 | 0.7333 |
| C. + statement-label aliases (final) | dense | 413 | 391 | 0.8571 | 0.8571 | 0.7000 |
| C. + statement-label aliases (final) | sparse | 413 | 391 | 0.7143 | 0.6476 | 0.8667 |

## Reading of these numbers

**The initial hypothesis was wrong, and the measurement says so.**

The financial route first scored R@3 = 0.14 on sparse retrieval. The obvious explanation was boilerplate: all 420 records opened with the same entity/filing sentence and repeated `for the period 2025-01-01 to 2025-12-31` on nearly every line, leaving them ~80% shared vocabulary. Variant B removes exactly that — label-first headers, `FY2025` instead of the long date phrasing, and TextBlock prose dropped.

Variant B changed **nothing**: sparse financial R@3 stayed at 0.1429 and dense stayed at 0.5714. Only MRR@10 moved, and only for dense (0.508 -> 0.607), i.e. slightly better ordering of results that were already being found. Tidying the boilerplate was cosmetic.

The actual defect is a **vocabulary gap**. A US-GAAP element is named `Revenue from Contract with Customer, Excluding Assessed Tax`; a human asks for `total operating revenue`. Those strings share almost no terms, which is fatal for BM25 (0.14) and merely bad for embeddings (0.57). Variant C bridges that gap and lifts sparse to 0.7143 and dense to 0.8571 — a 5x improvement on sparse from the one change that addressed the real cause.

**Caveat, stated plainly.** The alias map is the single largest lever on this route, and synonym injection is inherently at risk of being fitted to the evaluation questions. The aliases were authored from standard 10-K statement line-item conventions rather than from the eval set, and are reported here in isolation precisely so a sceptical reader can discount them. A stricter test — held-out financial questions phrased with vocabulary absent from the alias map — is listed in the report's 'what we would try next' section.
