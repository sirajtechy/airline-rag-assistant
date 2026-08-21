"""Stage 7 — cross-encoder reranking.

A bi-encoder embeds the query and the passage independently, so it never sees
them together. A cross-encoder scores the pair jointly and is markedly more
accurate — at the cost of one forward pass per candidate, which is why it is
only ever applied to a shortlist rather than the whole corpus.

The measured question for Stage 7 is whether that accuracy is worth the latency
for a customer-support bot, so this module reports both.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from .stores import SearchHit

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankResult:
    hits: list[SearchHit]
    seconds: float


@lru_cache(maxsize=2)
def _cross_encoder(model_id: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id)


class CrossEncoderReranker:
    def __init__(self, model_id: str = CROSS_ENCODER_MODEL):
        self.model_id = model_id

    def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> RerankResult:
        """Rescore a shortlist jointly against the query and keep the best ``top_k``."""
        if not hits:
            return RerankResult(hits=[], seconds=0.0)

        model = _cross_encoder(self.model_id)
        start = time.perf_counter()
        scores = model.predict(
            [(query, h.chunk.text) for h in hits], show_progress_bar=False
        )
        elapsed = time.perf_counter() - start

        rescored = [SearchHit(chunk=h.chunk, score=float(s)) for h, s in zip(hits, scores)]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return RerankResult(hits=rescored[:top_k], seconds=elapsed)
