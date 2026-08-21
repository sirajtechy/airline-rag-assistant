# Stage 2 — Chunking strategy

| Strategy | Chunk size | Overlap | Chunks | R@1 | R@3 | MRR@10 | DCG@3 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|
| fixed | 300 | 0 | 517 | 0.6333 | 0.7667 | 0.7229 | 3.5289 | 0.8323 |
| fixed | 500 | 0 | 476 | 0.6000 | 0.7667 | 0.6906 | 3.5158 | 0.8534 |
| fixed | 800 | 0 | 452 | 0.5667 | 0.7333 | 0.6704 | 3.3123 | 0.8242 |
| fixed_ov | 300 | 50 | 538 | 0.7000 | 0.8000 | 0.7639 | 3.6869 | 0.8449 |
| fixed_ov | 500 | 50 | 482 | 0.6333 | 0.8000 | 0.7261 | 3.6805 | 0.8757 |
| fixed_ov | 500 | 100 | 491 | 0.6667 | 0.7667 | 0.7205 | 3.6369 | 0.8409 |
| fixed_ov | 800 | 100 | 458 | 0.6333 | 0.7667 | 0.7033 | 3.4317 | 0.8142 |
| recursive | 300 | 50 | 558 | 0.7000 | 0.8333 | 0.7714 | 3.7428 | 0.8444 |
| recursive | 500 | 50 | 500 | 0.7333 | 0.8333 | 0.7936 | 3.7658 | 0.8568 |
| recursive | 500 | 100 | 501 | 0.7333 | 0.8333 | 0.7950 | 3.8464 | 0.8762 |
| recursive | 800 | 100 | 470 | 0.6333 | 0.8667 | 0.7361 | 3.6369 | 0.8710 |
| rule_aware | 300 | 50 | 638 | 0.6667 | 0.8333 | 0.7636 | 3.7341 | 0.8190 |
| rule_aware | 500 | 50 | 595 | 0.6333 | 0.9000 | 0.7583 | 3.7043 | 0.8523 |
| rule_aware | 800 | 100 | 566 | 0.6667 | 0.9000 | 0.7708 | 3.6956 | 0.8594 |

## Reading of these numbers

Winner: **rule_aware at 800 tokens / 100 overlap** (R@3 = 0.9000, MRR@10 = 0.7708).

`rule_aware` is a document-specific strategy added because both tariffs are organised as numbered rules (`RULE 20:`, `G14`), so a rule heading is a genuine semantic boundary rather than an arbitrary token offset. Comparing it against the three generic strategies is what shows whether that structure is worth exploiting on this corpus.
