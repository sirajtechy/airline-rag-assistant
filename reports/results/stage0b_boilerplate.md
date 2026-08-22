# Stage 0b — Front matter and running-header removal

| Preprocessing | Mode | PDF pages | R@1 | R@3 | MRR@10 | NDCG@3 | R@3 (cargo) | R@3 (passenger) |
|---|---|---|---|---|---|---|---|---|
| A. Raw pages (front matter + headers kept) | sparse | 45 | 0.6667 | 0.7667 | 0.7473 | 0.7943 | 0.6364 | 1.0000 |
| A. Raw pages (front matter + headers kept) | hybrid | 45 | 0.6667 | 0.9000 | 0.7708 | 0.8594 | 0.8182 | 1.0000 |
| B. Front matter dropped, running headers stripped | sparse | 43 | 0.6667 | 0.8333 | 0.7567 | 0.8486 | 0.7273 | 1.0000 |
| B. Front matter dropped, running headers stripped | hybrid | 43 | 0.7000 | 0.9000 | 0.7981 | 0.8874 | 0.8182 | 1.0000 |

## Reading of these numbers

Only **2 of 45 pages** are removed (contract p1, cargo p2), and no gold answer span lives on either, so the eval labels still validate 30/30.

The mechanism is worth stating precisely, because it explains why a contents page is worse than useless. A table of contents lists every rule title in the document: `RULE 20: DENIED BOARDING COMPENSATION`, `G32 Limit of Liability`, and so on. To BM25 that page looks maximally relevant to almost any topical query, while containing no rule text to answer it with. It is a chunk engineered to win retrieval and then say nothing.

The running header is the subtler of the two. `Delta Domestic General Rules Tariff` appears on all 23 pages of the passenger contract, so the terms 'domestic', 'rules' and 'tariff' are indexed 23 times in the passenger document — which is why a *cargo* question phrased as 'rules for shipping cargo domestically' was pulled toward the passenger contract. The cross-contamination this group is graded on was partly caused by page furniture.

This was found by debugging a failing acceptance test rather than by reading the PDFs, which is an argument for the methodology itself: the aggregate retrieval metrics were already respectable while a required acceptance test was failing for a cause no aggregate number surfaced.
