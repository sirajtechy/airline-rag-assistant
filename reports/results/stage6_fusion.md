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

**Selected: Reciprocal Rank Fusion (k=60).** RRF and weighted linear at alpha = 0.5 tie exactly on R@3 (0.9000). They then split the secondary metrics: weighted leads MRR@10 by +0.0308 while RRF leads NDCG@3 by -0.0087.

**Neither gap is real.** With 30 questions a single question is worth 0.0333 of R@3, so differences of ~0.02 on a secondary metric are smaller than the resolution of this eval set. Declaring a winner on that margin would be reading noise.

The tie is therefore broken on robustness, and the alpha sweep itself is the evidence: R@3 goes 0.8333 -> 0.8333 -> 0.9000 -> 0.8333 -> 0.8667 across alpha = 0.2 -> 0.8. That curve is **non-monotonic and spiky**. If alpha were capturing a real property of the corpus the curve would be smooth, so the peak at 0.5 is far more likely to be a coincidence of these 30 questions than a tuned optimum — precisely the overfitting the methodology warns against. RRF has no alpha to overfit and is invariant to the two retrievers' score scales, so it is the safer choice at identical measured quality.

For reference: alpha weights the dense side, `score = alpha * dense_norm + (1 - alpha) * sparse_norm`. Both distributions are min-max normalised first because BM25 scores are unbounded while cosine similarity is capped at 1; combining them raw would let BM25 dominate through scale rather than relevance. RRF sidesteps normalisation entirely by fusing ranks, at the cost of being blind to margin.
