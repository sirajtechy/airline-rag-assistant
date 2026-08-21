# Stage 5 — Dense vs sparse vs hybrid

| Mode | R@1 (all) | R@3 (all) | R@3 (keyword) | R@3 (semantic) | R@3 (passenger) | R@3 (cargo) | R@3 (financial) | MRR@10 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|---|
| Dense only | 0.6000 | 0.8333 | 0.8571 | 0.8125 | 0.8333 | 0.8182 | 0.8571 | 0.7140 | 0.7949 |
| Sparse (BM25) only | 0.6667 | 0.7667 | 0.8571 | 0.6875 | 1.0000 | 0.6364 | 0.5714 | 0.7473 | 0.7943 |
| Hybrid (RRF) | 0.6667 | 0.9000 | 0.8571 | 0.9375 | 1.0000 | 0.8182 | 0.8571 | 0.7708 | 0.8594 |

## Reading of these numbers

Winner: **Hybrid (RRF)** (R@3 = 0.9000).

This group's REQUIREMENT asks for the breakdown by target document, so the per-route columns are included alongside the required keyword/semantic split. That breakdown is the actual argument for or against hybrid: if one mode were uniformly better across all three routes and both query types, the added complexity of running two retrievers and fusing them would not be justified. Divergence between the columns is what makes fusion worth paying for.
