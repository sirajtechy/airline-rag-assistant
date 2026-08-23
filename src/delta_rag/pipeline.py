"""The end-to-end Delta support assistant.

Order of operations, and why:

1. **Guardrails first.** Out-of-scope carriers and live-data requests are caught
   before retrieval — deterministically, so refusals never depend on the model
   choosing to behave, and cheaply, so we do not embed a query we will refuse.
2. **Retrieve broadly, then route.** Retrieval runs unfiltered so the evidence
   itself decides the business line. Filtering first would require classifying
   the question up front, which is precisely what fails on cargo-flavoured
   financial questions.
3. **Scope the context to the routed line.** This is the anti-contamination
   step: once the route is decided, chunks from other business lines are dropped
   so the generator cannot quote the cargo tariff at a passenger.
4. **Abstain on weak evidence**, then generate with enforced citations.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .ablation import PipelineConfig, get_index
from .config import RESULTS_DIR
from .generate import GeneratedAnswer, generate
from .guardrails import (
    ABSTAIN_THRESHOLD,
    check_evidence,
    check_pre_retrieval,
    verify_citations,
)
from .routing import RouteDecision, route_from_hits
from .stores import SearchHit

DEFAULT_GENERATION_MODEL = "qwen2.5:7b"


def load_final_config() -> PipelineConfig:
    """The configuration chosen by the Stage 1-7 ablations."""
    path = RESULTS_DIR / "final_retrieval_config.json"
    if path.exists():
        data = json.loads(path.read_text())["config"]
        return PipelineConfig(**{k: v for k, v in data.items()
                                 if k in PipelineConfig.__dataclass_fields__})
    # Measured winners, hard-coded as a fallback so the app runs without the
    # ablation artefacts present. Mirrors reports/results/final_retrieval_config.json.
    return PipelineConfig(
        parser="pymupdf", strategy="recursive", size=500, overlap=50,
        embedding="bge-small", store="faiss", mode="hybrid", fusion="weighted",
        alpha=0.5, rerank=False,
    )


@dataclass
class BotResponse:
    question: str
    answer: str
    route: str | None
    blocked: bool
    reason: str | None
    sources: list[dict] = field(default_factory=list)
    route_votes: dict = field(default_factory=dict)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    model: str = ""
    citations_valid: bool = True
    contexts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class DeltaSupportBot:
    """Delta customer support assistant over the passenger, cargo and financial lines."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        model: str = DEFAULT_GENERATION_MODEL,
        top_k: int = 5,
        abstain_threshold: float = ABSTAIN_THRESHOLD,
    ):
        self.config = config or load_final_config()
        self.model = model
        self.top_k = top_k
        self.abstain_threshold = abstain_threshold
        self.chunks, self.retriever = get_index(self.config)

    # ── internals ───────────────────────────────────────────────────────────
    def _dense_confidence(self, question: str, hits: Sequence[SearchHit]) -> list[float]:
        """Cosine similarity of the retrieved chunks, for the abstain gate.

        The fused RRF score is a rank artefact with no absolute meaning, so it
        cannot support a threshold. Cosine similarity is bounded and comparable
        across queries, which is what a confidence floor needs.
        """
        if not hits or not self.retriever.supports_dense:
            return []
        qv = self.retriever.embedder.encode_queries([question]).vectors[0]

        cached = self.retriever.doc_vectors
        missing = [h for h in hits if h.chunk.chunk_id not in cached]
        if missing:  # only pay for chunks the index did not already embed
            fresh = self.retriever.embedder.encode_documents(
                [h.chunk.text for h in missing]
            ).vectors
            for hit, vector in zip(missing, fresh):
                cached[hit.chunk.chunk_id] = vector

        # Both sides are L2-normalised at encode time, so a dot product is cosine.
        return [float(qv @ cached[h.chunk.chunk_id]) for h in hits]

    # ── public API ──────────────────────────────────────────────────────────
    def answer(self, question: str, generate_answer: bool = True) -> BotResponse:
        guard = check_pre_retrieval(question)
        if guard.blocked:
            return BotResponse(
                question=question, answer=guard.answer, route=None, blocked=True,
                reason=guard.reason, model=self.model,
            )

        result = self.retriever.retrieve(
            question, k=max(self.top_k * 3, 15), mode=self.config.mode,
            fusion=self.config.fusion, alpha=self.config.alpha,
        )
        # Top-1 routing, chosen on measurement: it scores 1.000 on the 30 eval
        # questions and 0.971 across all 35 routing cases, beating a rank-weighted
        # vote (0.914) and a vote plus defined-term prior (0.943). Both richer
        # policies misroute R1 ("revenue from shipping cargo"), where lower-ranked
        # cargo-tariff chunks outvote the correct financial record. See
        # reports/results/stage0c_routing.md.
        decision: RouteDecision = route_from_hits(
            result.hits, top_n=1, question=question, prior_weight=0.0
        )

        # Anti-contamination: keep only the routed business line.
        scoped = [h for h in result.hits if h.chunk.business_line == decision.business_line]
        scoped = (scoped or list(result.hits))[: self.top_k]

        confidences = self._dense_confidence(question, scoped)
        evidence = check_evidence(scoped, confidences, self.abstain_threshold)
        if evidence.blocked:
            return BotResponse(
                question=question, answer=evidence.answer, route=decision.business_line,
                blocked=True, reason=evidence.reason, route_votes=decision.votes,
                retrieval_ms=result.seconds * 1000, model=self.model,
            )

        contexts = [h.chunk.text for h in scoped]
        if not generate_answer:
            from .generate import format_sources
            _, meta = format_sources(scoped)
            return BotResponse(
                question=question, answer="", route=decision.business_line,
                blocked=False, reason=None, sources=meta,
                route_votes=decision.votes, retrieval_ms=result.seconds * 1000,
                model=self.model, contexts=contexts,
            )

        gen: GeneratedAnswer = generate(question, scoped, self.model)
        valid, _ = verify_citations(gen.text, len(scoped))
        return BotResponse(
            question=question, answer=gen.text, route=decision.business_line,
            blocked=False, reason=None, sources=gen.sources,
            route_votes=decision.votes, retrieval_ms=result.seconds * 1000,
            generation_ms=gen.seconds * 1000, model=self.model,
            citations_valid=valid, contexts=contexts,
        )
