# Stage 3 — Embedding model

| Embedding model | Key | Dimensions | Latency (ms/query) | R@1 | R@3 | MRR@10 | DCG@3 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | minilm | 384 | 1.3000 | 0.5667 | 0.7667 | 0.7003 | 3.3412 | 0.7802 |
| BAAI/bge-small-en-v1.5 | bge-small | 384 | 1.6000 | 0.6667 | 0.9000 | 0.7708 | 3.6956 | 0.8594 |
| nomic-embed-text:latest | nomic | 768 | 10.0000 | 0.6333 | 0.8667 | 0.7639 | 3.5631 | 0.8099 |

## Reading of these numbers

Winner: **bge-small** (R@3 = 0.9000, MRR@10 = 0.7708, 1.6 ms/query).

`bge-small` and `nomic` are asymmetric models: they are trained to embed queries and passages with different instruction prefixes. Those prefixes are applied here (`Represent this sentence for searching relevant passages: ` and `search_query:`/`search_document:` respectively). Benchmarking them without their prefixes is a common error that would have understated both and handed the stage to MiniLM by default.

Latency is measured after a warm-up call so model load time is excluded; `nomic` pays a network round-trip to Ollama that the two in-process sentence-transformers models do not.
