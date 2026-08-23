# Stage 1 — Parsing strategy

| Parser | Clean text % | Gold spans recoverable | Parse time (s) | R@1 | R@3 | MRR@10 | DCG@3 | NDCG@3 |
|---|---|---|---|---|---|---|---|---|
| pypdf | 97.8% | 96.7% | 1.5800 | 0.7333 | 0.8667 | 0.8078 | 3.9043 | 0.8877 |
| pdfplumber | 97.8% | 100.0% | 4.4700 | 0.6333 | 0.8000 | 0.7329 | 3.5587 | 0.8174 |
| pymupdf | 97.8% | 100.0% | 0.3000 | 0.7000 | 0.9000 | 0.8011 | 3.7964 | 0.8762 |

## Reading of these numbers

All three parsers extract all 45 pages and clear the clean-text bar, so the usual 'clean text %' column does not separate them. The column that does is gold-span recoverability: **pypdf loses one of the 30 answer spans**, rendering `those` as `t hose` on contract page 4 (question P07). The page is still retrievable — which is why R@3 barely moves — but the verbatim evidence needed to ground a citation is gone.

PyMuPDF is chosen: it recovers 100% of gold spans and is the fastest of the three. pdfplumber matches it on recovery but is materially slower, and its table-extraction advantage is irrelevant here because neither tariff stores its rules in ruled tables.
