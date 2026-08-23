# Stage 3 — Embedding model

| Embedding model | Key | Dimensions | Latency (ms/query) | R@1 | R@3 | MRR@10 | DCG@3 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | minilm | 384 | 1.2000 | 0.7000 | 0.9000 | 0.7889 | 3.7412 | 0.8844 |
| BAAI/bge-small-en-v1.5 | bge-small | 384 | 1.7000 | 0.7000 | 0.9000 | 0.8011 | 3.7964 | 0.8762 |
| nomic-embed-text:latest | nomic | 768 | 15.3000 | 0.7667 | 0.8333 | 0.8119 | 3.7571 | 0.8445 |

## Reading of these numbers

Winner: **bge-small** (R@3 = 0.9000, MRR@10 = 0.8011, 1.7 ms/query).

`bge-small` and `nomic` are asymmetric models: they are trained to embed queries and passages with different instruction prefixes. Those prefixes are applied here (`Represent this sentence for searching relevant passages: ` and `search_query:`/`search_document:` respectively). Benchmarking them without their prefixes is a common error that would have understated both and handed the stage to MiniLM by default.

Latency is measured after a warm-up call so model load time is excluded; `nomic` pays a network round-trip to Ollama that the two in-process sentence-transformers models do not.
