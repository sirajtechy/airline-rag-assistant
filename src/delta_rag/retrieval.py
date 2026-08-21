"""Stages 5 & 6 — dense, sparse, and hybrid retrieval.

Dense  : embeddings + vector store (semantic similarity)
Sparse : BM25 over stemmed tokens (exact-term matching)
Hybrid : both, fused by Reciprocal Rank Fusion or weighted linear combination

Stage 6 exists because RRF and weighted fusion fail differently. RRF only sees
*ranks*, so it is immune to the two retrievers' scores being on incomparable
scales, but it also throws away the information in "this hit was far better than
that one". Weighted fusion keeps that magnitude but requires normalising two
distributions that genuinely are not comparable. Which wins is an empirical
question, which is exactly why the methodology demands the sweep.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder
from .stores import SearchHit, VectorStore

RRF_K = 60  # standard constant from the RRF literature; k~60 is the usual default


@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    seconds: float


# ── Sparse (BM25) ────────────────────────────────────────────────────────────
class BM25Retriever:
    """BM25 over Snowball-stemmed tokens, via the course's `bm25s` library."""

    def __init__(self):
        self._retriever = None
        self._chunks: list[Chunk] = []
        self._stemmer = None

    def build(self, chunks: Sequence[Chunk]) -> float:
        import bm25s
        import Stemmer

        start = time.perf_counter()
        self._chunks = list(chunks)
        self._stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize(
            [c.text for c in self._chunks], stopwords="en", stemmer=self._stemmer,
            show_progress=False,
        )
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        self._retriever = retriever
        return time.perf_counter() - start

    def search(self, query: str, k: int, business_line: str | None = None) -> list[SearchHit]:
        import bm25s

        fetch = k if business_line is None else min(k * 12, len(self._chunks))
        tokens = bm25s.tokenize(query, stopwords="en", stemmer=self._stemmer,
                                show_progress=False)
        idxs, scores = self._retriever.retrieve(
            tokens, k=min(fetch, len(self._chunks)), show_progress=False
        )
        hits = []
        for idx, score in zip(idxs[0], scores[0]):
            chunk = self._chunks[int(idx)]
            if business_line and chunk.business_line != business_line:
                continue
            hits.append(SearchHit(chunk=chunk, score=float(score)))
            if len(hits) == k:
                break
        return hits


# ── Fusion ───────────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]], k: int, rrf_k: int = RRF_K
) -> list[SearchHit]:
    """score(d) = sum over lists of 1 / (rrf_k + rank(d)), rank being 1-based."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks[cid] = hit.chunk
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [SearchHit(chunk=chunks[cid], score=score) for cid, score in ordered]


def _min_max(values: dict[str, float]) -> dict[str, float]:
    """Normalise to [0,1]. Ties (or a single hit) collapse to 1.0, not 0/0."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {key: 1.0 for key in values}
    return {key: (v - lo) / (hi - lo) for key, v in values.items()}


def weighted_fusion(
    dense: Sequence[SearchHit], sparse: Sequence[SearchHit], k: int, alpha: float
) -> list[SearchHit]:
    """score(d) = alpha * dense_norm + (1 - alpha) * sparse_norm.

    Both score distributions are min-max normalised first because BM25 scores are
    unbounded while cosine similarity is not; combining them raw would let BM25
    dominate purely through scale.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    dense_scores = _min_max({h.chunk.chunk_id: h.score for h in dense})
    sparse_scores = _min_max({h.chunk.chunk_id: h.score for h in sparse})
    chunks = {h.chunk.chunk_id: h.chunk for h in list(dense) + list(sparse)}

    combined = {
        cid: alpha * dense_scores.get(cid, 0.0) + (1 - alpha) * sparse_scores.get(cid, 0.0)
        for cid in chunks
    }
    ordered = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [SearchHit(chunk=chunks[cid], score=score) for cid, score in ordered]


# ── Unified retriever ────────────────────────────────────────────────────────
class Retriever:
    """Dense / sparse / hybrid retrieval over one chunked corpus."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        bm25: BM25Retriever | None = None,
    ):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.build_seconds: dict[str, float] = {}

    @property
    def supports_dense(self) -> bool:
        return self.embedder is not None and self.store is not None

    def retrieve(
        self,
        query: str,
        k: int = 10,
        mode: str = "hybrid",
        business_line: str | None = None,
        fusion: str = "rrf",
        alpha: float = 0.5,
        candidate_k: int | None = None,
    ) -> RetrievalResult:
        """Run one query. ``candidate_k`` controls the per-retriever fan-out
        before fusion; fusing only the top-k of each would discard hits that the
        other retriever ranks highly."""
        start = time.perf_counter()
        fan_out = candidate_k or max(k * 2, 20)

        if mode == "dense":
            hits = self._dense(query, k, business_line)
        elif mode == "sparse":
            hits = self.bm25.search(query, k, business_line)
        elif mode == "hybrid":
            dense = self._dense(query, fan_out, business_line)
            sparse = self.bm25.search(query, fan_out, business_line)
            if fusion == "rrf":
                hits = reciprocal_rank_fusion([dense, sparse], k)
            elif fusion == "weighted":
                hits = weighted_fusion(dense, sparse, k, alpha)
            else:
                raise ValueError(f"unknown fusion {fusion!r}; expected 'rrf' or 'weighted'")
        else:
            raise ValueError(f"unknown mode {mode!r}; expected dense|sparse|hybrid")

        return RetrievalResult(hits=hits, seconds=time.perf_counter() - start)

    def _dense(self, query: str, k: int, business_line: str | None) -> list[SearchHit]:
        if not self.supports_dense:
            raise RuntimeError("dense retrieval requires both an embedder and a store")
        vector = self.embedder.encode_queries([query]).vectors[0]
        return self.store.search(vector, k, business_line)


def build_retriever(
    chunks: Sequence[Chunk],
    embedder: Embedder | None,
    store: VectorStore | None,
    with_sparse: bool = True,
) -> Retriever:
    """Embed the corpus, build the vector index and the BM25 index."""
    retriever = Retriever(chunks, embedder, store)

    if embedder is not None and store is not None:
        embedded = embedder.encode_documents([c.text for c in chunks])
        retriever.build_seconds["embed"] = embedded.seconds
        retriever.build_seconds["index"] = store.build(chunks, embedded.vectors)

    if with_sparse:
        bm25 = BM25Retriever()
        retriever.build_seconds["bm25"] = bm25.build(chunks)
        retriever.bm25 = bm25

    return retriever
