# Stage 5 — Dense vs sparse vs hybrid

| Mode | R@1 (all) | R@3 (all) | R@3 (keyword) | R@3 (semantic) | R@3 (passenger) | R@3 (cargo) | R@3 (financial) | MRR@10 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|---|
| Dense only | 0.6000 | 0.7667 | 0.7857 | 0.7500 | 0.9167 | 0.5455 | 0.8571 | 0.6822 | 0.8473 |
| Sparse (BM25) only | 0.7333 | 0.8667 | 0.8571 | 0.8750 | 1.0000 | 0.8182 | 0.7143 | 0.7956 | 0.8323 |
| Hybrid (RRF) | 0.7000 | 0.9000 | 0.8571 | 0.9375 | 1.0000 | 0.8182 | 0.8571 | 0.8011 | 0.8762 |

## Reading of these numbers

Winner: **Hybrid (RRF)** (R@3 = 0.9000).

This group's REQUIREMENT asks for the breakdown by target document, so the per-route columns are included alongside the required keyword/semantic split. That breakdown is the actual argument for or against hybrid: if one mode were uniformly better across all three routes and both query types, the added complexity of running two retrievers and fusing them would not be justified. Divergence between the columns is what makes fusion worth paying for.
