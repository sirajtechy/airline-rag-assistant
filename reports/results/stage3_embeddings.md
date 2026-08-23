# Stage 3 — Embedding model

| Embedding model | Key | Dimensions | Latency (ms/query) | R@1 | R@3 | MRR@10 | DCG@3 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | minilm | 384 | 1.2000 | 0.7000 | 0.9000 | 0.7889 | 3.7412 | 0.8844 |
| BAAI/bge-small-en-v1.5 | bge-small | 384 | 1.7000 | 0.7000 | 0.9000 | 0.8011 | 3.7964 | 0.8762 |
| nomic-embed-text:latest | nomic | 768 | 15.3000 | 0.7667 | 0.8333 | 0.8119 | 3.7571 | 0.8445 |

## Reading of these numbers

Winner: **bge-small** (R@3 = 0.9000, MRR@10 = 0.8011, 1.7 ms/query).

**This is a near-tie, not a clear win.** 2 of the 3 candidates (minilm, bge-small) share the top R@3 of 0.9000 and separate only on secondary metrics — where MiniLM actually takes NDCG@3 and bge-small takes MRR@10.

This **contradicts an earlier run of this same table**, in which bge-small led MiniLM by +0.1333 R@3. That gap existed *before* the front-matter and running-header removal in Stage 0b. Cleaning the boilerplate that was polluting retrieval made the embedding model substantially less important, which is a result in its own right: fixing the corpus reduced the leverage of a hyperparameter. It is also a caution about ablation order — a stage measured against a defective corpus can report a difference that evaporates once the defect is fixed.

`bge-small` and `nomic` are asymmetric models: they are trained to embed queries and passages with different instruction prefixes. Those prefixes are applied here (`Represent this sentence for searching relevant passages: ` and `search_query:`/`search_document:` respectively). Benchmarking them without their prefixes is a common error that would have understated both and handed the stage to MiniLM by default.

`nomic` is the interesting outlier: it takes the best R@1 and the best MRR@10 of the three, but the worst R@3, and costs ~9x the latency because it is served over HTTP by Ollama rather than running in-process. It ranks its best guess well and its next guesses poorly.

Latency is measured after a warm-up call so model load time is excluded.
