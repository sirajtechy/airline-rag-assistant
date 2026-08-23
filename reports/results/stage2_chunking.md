# Stage 2 — Chunking strategy

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

## Reading of these numbers

Winner: **recursive at 500 tokens / 50 overlap** (R@3 = 0.9000, MRR@10 = 0.8011).

`rule_aware` is a document-specific strategy added because both tariffs are organised as numbered rules (`RULE 20:`, `G14`), so a rule heading is a genuine semantic boundary rather than an arbitrary token offset. Comparing it against the three generic strategies is what shows whether that structure is worth exploiting on this corpus.
