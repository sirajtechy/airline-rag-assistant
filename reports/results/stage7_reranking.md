# Stage 7 — Cross-encoder reranking

| Config | R@1 | R@3 | NDCG@3 | Added latency (ms) |
|---|---|---|---|---|
| No reranker | 0.7333 | 0.9333 | 0.8849 | 0.0000 |
| + Cross-encoder rerank (top-10 -> top-3/5) | 0.7333 | 0.8667 | 0.8388 | 103.7000 |
| + Cross-encoder rerank (top-20 -> top-3/5) | 0.7333 | 0.8667 | 0.8343 | 176.2000 |

## Reading of these numbers

A bi-encoder embeds query and passage separately and never sees them together; a cross-encoder scores the pair jointly, which is why it is normally more accurate and why it can only be afforded on a shortlist.

**The reranker is rejected — it makes retrieval worse, not better.**

NDCG@3 falls from 0.8849 with no reranker to 0.8388 at top-10 and 0.8343 at top-20, and at top-20 R@3 drops too (0.9333 -> 0.8667). It costs 176.2 ms per query to be worse. Notably the damage grows with the candidate pool: the more documents it is given to reorder, the more harm it does.

This is the opposite of the textbook result, so it needs an explanation rather than a shrug. Two things are going on.

First, **domain mismatch**. `ms-marco-MiniLM-L-6-v2` is trained on MS MARCO — short, factual web-search passages answering natural questions. This corpus is neither: it is dense legal tariff prose full of cross-references (`as provided in Rule 20(c)(1)`) and XBRL records that read as label-plus-numbers. The cross-encoder's learned notion of relevance transfers poorly to both, so its confident reordering is confidently wrong.

Second, **there is almost no headroom to win and plenty to lose**. Hybrid fusion already reaches R@3 = 0.9333; a reranker cannot add documents fusion never retrieved, it can only reshuffle them. When the correct chunk is usually already in the top 3, any reshuffling is far more likely to demote it than promote it.

The general lesson is the one the methodology is built around: a reranker is a *conditional* improvement, not an automatic one. Adding it because a tutorial did would have cost this pipeline both accuracy and latency.
