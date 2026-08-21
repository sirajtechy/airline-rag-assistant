"""The ablation harness — one place that runs a config and scores it.

Every table in the report comes out of :func:`run_config`, so all eight stages
are measured by identical code and differ only in the config under test. That is
the whole point: if each stage had its own bespoke measurement path, the numbers
would not be comparable across stages.

Built indexes are cached by their construction key, because Stage 5, 6 and 7 all
re-query the same index and rebuilding it each time would dominate the runtime.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .chunking import Chunk, chunk_corpus
from .config import RESULTS_DIR
from .corpus import load_corpus
from .embeddings import get_embedder
from .evalset import EvalQuestion, load_eval_set
from .metrics import RetrievalScores, grade_ranking, score_by_facet, score_run
from .rerank import CrossEncoderReranker
from .retrieval import build_retriever
from .stores import get_store


@dataclass(frozen=True)
class PipelineConfig:
    """One fully-specified retrieval pipeline."""

    parser: str = "pymupdf"
    strategy: str = "recursive"
    size: int = 500
    overlap: int = 50
    embedding: str = "bge-small"
    store: str = "faiss"
    mode: str = "hybrid"          # dense | sparse | hybrid
    fusion: str = "rrf"           # rrf | weighted
    alpha: float = 0.5            # only used when fusion == "weighted"
    rerank: bool = False
    rerank_candidates: int = 20
    top_k: int = 10

    @property
    def index_key(self) -> tuple:
        """Everything that affects the built index (not the query-time config)."""
        return (self.parser, self.strategy, self.size, self.overlap,
                self.embedding, self.store, self.mode)

    def label(self) -> str:
        bits = [self.parser, f"{self.strategy}/{self.size}/{self.overlap}",
                self.embedding, self.store, self.mode]
        if self.mode == "hybrid":
            bits.append(self.fusion if self.fusion == "rrf" else f"weighted a={self.alpha}")
        if self.rerank:
            bits.append(f"rerank@{self.rerank_candidates}")
        return " | ".join(bits)


@dataclass
class RunResult:
    config: PipelineConfig
    scores: RetrievalScores
    by_route: dict[str, RetrievalScores]
    by_query_type: dict[str, RetrievalScores]
    n_chunks: int
    build_seconds: dict[str, float]
    query_ms_mean: float
    rerank_ms_mean: float = 0.0
    per_question: list[dict] = field(default_factory=list)


_INDEX_CACHE: dict[tuple, tuple] = {}


def _needs_dense(mode: str) -> bool:
    return mode in {"dense", "hybrid"}


def get_index(config: PipelineConfig):
    """Build (or reuse) the chunked corpus + retriever for a config."""
    key = config.index_key
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    units = load_corpus(config.parser)
    chunks = chunk_corpus(units, config.strategy, config.size, config.overlap)

    embedder = get_embedder(config.embedding) if _needs_dense(config.mode) else None
    store = get_store(config.store) if _needs_dense(config.mode) else None
    retriever = build_retriever(chunks, embedder, store, with_sparse=True)

    _INDEX_CACHE[key] = (chunks, retriever)
    return chunks, retriever


def run_config(
    config: PipelineConfig,
    questions: Sequence[EvalQuestion] | None = None,
    keep_per_question: bool = False,
) -> RunResult:
    """Execute one pipeline over the whole eval set and score it."""
    questions = list(questions or load_eval_set())
    chunks, retriever = get_index(config)
    reranker = CrossEncoderReranker() if config.rerank else None

    graded: list[list[int]] = []
    per_question: list[dict] = []
    query_times: list[float] = []
    rerank_times: list[float] = []

    for q in questions:
        fetch = config.rerank_candidates if config.rerank else config.top_k
        start = time.perf_counter()
        result = retriever.retrieve(
            q.question, k=fetch, mode=config.mode, fusion=config.fusion,
            alpha=config.alpha,
        )
        hits = result.hits
        query_times.append((time.perf_counter() - start) * 1000.0)

        if reranker is not None:
            reranked = reranker.rerank(q.question, hits, config.top_k)
            hits = reranked.hits
            rerank_times.append(reranked.seconds * 1000.0)

        grades = grade_ranking([h.chunk for h in hits], q)
        graded.append(grades)

        if keep_per_question:
            per_question.append({
                "id": q.id,
                "question": q.question,
                "route": q.route,
                "query_type": q.query_type,
                "grades": grades,
                "top_doc": hits[0].chunk.doc_id if hits else None,
                "top_locators": hits[0].chunk.locators if hits else [],
            })

    # Routing is judged by which document the top hit came from.
    predicted = [
        (p["top_doc"] if keep_per_question else None) for p in per_question
    ] if keep_per_question else None
    routes = None
    if keep_per_question:
        from .config import SOURCE_DOCS
        routes = [
            SOURCE_DOCS[d]["business_line"] if d in SOURCE_DOCS else "unknown"
            for d in predicted
        ]

    scores = score_run(
        graded,
        predicted_routes=routes,
        gold_routes=[q.route for q in questions] if routes else None,
    )

    return RunResult(
        config=config,
        scores=scores,
        by_route=score_by_facet(graded, [q.route for q in questions]),
        by_query_type=score_by_facet(graded, [q.query_type for q in questions]),
        n_chunks=len(chunks),
        build_seconds=dict(retriever.build_seconds),
        query_ms_mean=sum(query_times) / max(len(query_times), 1),
        rerank_ms_mean=sum(rerank_times) / max(len(rerank_times), 1),
        per_question=per_question,
    )


# ── Reporting helpers ────────────────────────────────────────────────────────
def markdown_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """Render result rows as a GitHub-flavoured markdown table."""
    if not rows:
        return "_(no rows)_"
    columns = columns or list(rows[0].keys())
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(
            "" if r.get(c) is None else
            (f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c)))
            for c in columns
        ) + " |"
        for r in rows
    ]
    return "\n".join([head, rule, *body])


def save_stage(stage: str, title: str, rows: list[dict], notes: str = "",
               columns: list[str] | None = None, extra: str = "") -> Path:
    """Write one stage's table to reports/results/ and mirror it as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = RESULTS_DIR / f"{stage}.md"
    body = [f"# {title}", "", markdown_table(rows, columns)]
    if extra:
        body += ["", extra]
    if notes:
        body += ["", "## Reading of these numbers", "", notes]
    md_path.write_text("\n".join(body) + "\n")
    (RESULTS_DIR / f"{stage}.json").write_text(json.dumps(rows, indent=2, default=str))
    return md_path


def config_row(result: RunResult, **prefix) -> dict:
    row = dict(prefix)
    row.update(result.scores.as_row())
    row.pop("Route acc", None)
    return row
