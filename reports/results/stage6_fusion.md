# Stage 6 — Hybrid merge method and weighting

| Merge method | alpha | R@3 | MRR@10 | NDCG@3 |
|---|---|---|---|---|
| Reciprocal Rank Fusion (k=60) |  | 0.9000 | 0.8011 | 0.8762 |
| Weighted linear | 0.2000 | 0.8667 | 0.8059 | 0.8461 |
| Weighted linear | 0.3000 | 0.8667 | 0.8069 | 0.8426 |
| Weighted linear | 0.5000 | 0.9333 | 0.8319 | 0.8849 |
| Weighted linear | 0.7000 | 0.8667 | 0.7656 | 0.8547 |
| Weighted linear | 0.8000 | 0.8000 | 0.7169 | 0.8406 |

## Reading of these numbers

**Selected: weighted linear at alpha = 0.5** (R@3 = 0.9333, MRR@10 = 0.8319, NDCG@3 = 0.8849), ahead of RRF at R@3 = 0.9000.

**The margin is +0.0333 R@3 — exactly 1 question out of 30.** That is the resolution limit of this evaluation set, so the win is real but weakly evidenced, and it is reported as such rather than as a decisive result.

**The alpha sweep is the reason for caution.** Across alpha = 0.2, 0.3, 0.5, 0.7, 0.8 the R@3 curve runs 0.8667 -> 0.8667 -> 0.9333 -> 0.8667 -> 0.8000 — **non-monotonic and spiky**. If alpha were capturing a real property of the corpus the curve would be smooth, so the peak is more plausibly a coincidence of these particular 30 questions than a tuned optimum. That is precisely the overfitting the methodology warns against, and it is why **RRF remains the safer production choice** despite losing this comparison: it has no alpha to overfit and is invariant to the two retrievers' score scales.

For reference: alpha weights the dense side, `score = alpha * dense_norm + (1 - alpha) * sparse_norm`. Both distributions are min-max normalised first because BM25 scores are unbounded while cosine similarity is capped at 1; combining them raw would let BM25 dominate through scale rather than relevance. RRF sidesteps normalisation entirely by fusing ranks, at the cost of being blind to margin.
