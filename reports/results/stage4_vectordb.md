# Stage 4 — Vector database

| Vector DB | Index build time (ms) | Metadata filtering | Persistence | R@1 | R@3 | MRR@10 | DCG@3 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|
| faiss | 0.0600 | emulated | yes | 0.6667 | 0.9000 | 0.7708 | 3.6956 | 0.8594 |
| chromadb | 323.6300 | native | yes | 0.6667 | 0.9000 | 0.7708 | 3.6956 | 0.8594 |

## Reading of these numbers

R@3 is reported as a parity check and behaves as the methodology predicts: identical embeddings retrieve near-identically regardless of store, so this decision cannot be made on quality.

It is made on operational grounds instead. FAISS builds its flat inner-product index far faster at this corpus size and runs in-process with no server, but has **no native metadata filtering** — business-line scoping has to be emulated by over-fetching and post-filtering. ChromaDB supports `where` filtering natively and persists without extra code.

FAISS is selected for the graded pipeline: at ~500 chunks the over-fetch cost is negligible and the build-time advantage makes the ablation matrix practical to re-run. ChromaDB would be the better choice if this corpus grew by an order of magnitude, where post-filtering starts to threaten recall.
