"""Stage 3 — embedding backends.

Three candidates spanning two serving styles: two local sentence-transformers
models and one served by Ollama.

ASYMMETRIC PREFIXES MATTER
--------------------------
``bge-small-en-v1.5`` and ``nomic-embed-text`` are trained with *asymmetric*
query/document instructions. Embedding a query the same way as a passage costs
them real accuracy, so a benchmark that skipped the prefixes would quietly rig
Stage 3 in MiniLM's favour. Each backend therefore declares its own query and
document prefixes and applies them automatically.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .config import OLLAMA_BASE_URL


@dataclass
class EmbeddingSpec:
    key: str
    model_id: str
    backend: str          # "sentence-transformers" | "ollama"
    dimensions: int
    query_prefix: str = ""
    doc_prefix: str = ""
    notes: str = ""


EMBEDDING_MODELS: dict[str, EmbeddingSpec] = {
    "minilm": EmbeddingSpec(
        key="minilm",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        backend="sentence-transformers",
        dimensions=384,
        notes="Symmetric model, no prefixes. The course default.",
    ),
    "bge-small": EmbeddingSpec(
        key="bge-small",
        model_id="BAAI/bge-small-en-v1.5",
        backend="sentence-transformers",
        dimensions=384,
        query_prefix="Represent this sentence for searching relevant passages: ",
        notes="Asymmetric: queries take the retrieval instruction, passages do not.",
    ),
    "nomic": EmbeddingSpec(
        key="nomic",
        model_id="nomic-embed-text:latest",
        backend="ollama",
        dimensions=768,
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
        notes="Asymmetric task prefixes are mandatory for this model.",
    ),
}


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    seconds: float
    per_item_ms: float = field(init=False)

    def __post_init__(self):
        n = max(len(self.vectors), 1)
        self.per_item_ms = self.seconds * 1000.0 / n


class Embedder:
    """Uniform embedding interface over both serving styles."""

    def __init__(self, spec: EmbeddingSpec):
        self.spec = spec

    # ── backends ────────────────────────────────────────────────────────────
    @staticmethod
    @lru_cache(maxsize=4)
    def _st_model(model_id: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_id)

    def _encode_st(self, texts: list[str]) -> np.ndarray:
        model = self._st_model(self.spec.model_id)
        return np.asarray(
            model.encode(texts, batch_size=32, normalize_embeddings=True,
                         show_progress_bar=False),
            dtype=np.float32,
        )

    def _encode_ollama(self, texts: list[str]) -> np.ndarray:
        import ollama

        client = ollama.Client(host=OLLAMA_BASE_URL)
        resp = client.embed(model=self.spec.model_id, input=texts)
        vectors = np.asarray(resp["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    # ── public API ──────────────────────────────────────────────────────────
    def encode(self, texts: list[str], is_query: bool = False) -> EmbeddingResult:
        prefix = self.spec.query_prefix if is_query else self.spec.doc_prefix
        prepared = [prefix + t for t in texts] if prefix else list(texts)

        start = time.perf_counter()
        if self.spec.backend == "sentence-transformers":
            vectors = self._encode_st(prepared)
        elif self.spec.backend == "ollama":
            vectors = self._encode_ollama(prepared)
        else:
            raise ValueError(f"unknown backend {self.spec.backend!r}")
        elapsed = time.perf_counter() - start

        if vectors.shape[1] != self.spec.dimensions:
            raise RuntimeError(
                f"{self.spec.key}: expected {self.spec.dimensions} dims, "
                f"got {vectors.shape[1]}"
            )
        return EmbeddingResult(vectors=vectors, seconds=elapsed)

    def encode_queries(self, texts: list[str]) -> EmbeddingResult:
        return self.encode(texts, is_query=True)

    def encode_documents(self, texts: list[str]) -> EmbeddingResult:
        return self.encode(texts, is_query=False)


def get_embedder(key: str) -> Embedder:
    if key not in EMBEDDING_MODELS:
        raise ValueError(f"unknown embedding model {key!r}; expected one of {sorted(EMBEDDING_MODELS)}")
    return Embedder(EMBEDDING_MODELS[key])
