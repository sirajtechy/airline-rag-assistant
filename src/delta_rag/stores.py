"""Stage 4 — vector stores.

FAISS and ChromaDB behind one interface. The methodology is explicit that
retrieval *quality* should be near-identical for the same embeddings, so R@3 is
reported as a parity check and the decision is argued on operational axes:
index build time, metadata filtering, and persistence.

Metadata filtering is not incidental here — it is how the final pipeline scopes
retrieval to a single business line once the router has picked one.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .chunking import Chunk
from .config import DATA_INDEX


@dataclass
class StoreCapabilities:
    metadata_filtering: bool
    persistence: bool
    notes: str


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    name: str
    capabilities: StoreCapabilities

    def build(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> float:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int,
               business_line: str | None = None) -> list[SearchHit]:
        raise NotImplementedError


class FaissStore(VectorStore):
    """Flat inner-product index. Vectors are L2-normalised, so IP == cosine.

    Exact search: with a 500-2000 chunk corpus an approximate index would add
    recall risk for no measurable speed benefit.
    """

    name = "faiss"
    capabilities = StoreCapabilities(
        metadata_filtering=False,   # not natively; emulated by over-fetch + filter
        persistence=True,           # via faiss.write_index
        notes="In-process, exact flat IP search. Filtering emulated by over-fetching.",
    )

    def __init__(self):
        self._index = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> float:
        import faiss

        start = time.perf_counter()
        self._chunks = list(chunks)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._index = index
        return time.perf_counter() - start

    def search(self, query_vector, k, business_line=None):
        # FAISS cannot filter, so over-fetch and post-filter to still return k.
        fetch = k if business_line is None else min(k * 12, len(self._chunks))
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        scores, idxs = self._index.search(query, fetch)

        hits = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            if business_line and chunk.business_line != business_line:
                continue
            hits.append(SearchHit(chunk=chunk, score=float(score)))
            if len(hits) == k:
                break
        return hits


class ChromaStore(VectorStore):
    """Persistent client with native `where` metadata filtering."""

    name = "chromadb"
    capabilities = StoreCapabilities(
        metadata_filtering=True,
        persistence=True,
        notes="Native where-clause filtering and on-disk persistence out of the box.",
    )

    def __init__(self, collection: str = "delta_rag", persist: bool = False):
        self._collection_name = collection
        self._persist = persist
        self._collection = None
        self._by_id: dict[str, Chunk] = {}

    def build(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> float:
        import chromadb

        start = time.perf_counter()
        if self._persist:
            path = DATA_INDEX / "chroma"
            shutil.rmtree(path, ignore_errors=True)
            client = chromadb.PersistentClient(path=str(path))
        else:
            client = chromadb.EphemeralClient()

        try:
            client.delete_collection(self._collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            self._collection_name, metadata={"hnsw:space": "cosine"}
        )

        chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in chunks}
        # Chroma rejects oversized batches; 2k is comfortably under the cap.
        for i in range(0, len(chunks), 2000):
            batch = chunks[i : i + 2000]
            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=[v.tolist() for v in vectors[i : i + 2000]],
                documents=[c.text for c in batch],
                metadatas=[
                    {"doc_id": c.doc_id, "business_line": c.business_line,
                     "locators": ",".join(c.locators)}
                    for c in batch
                ],
            )
        self._collection = collection
        return time.perf_counter() - start

    def search(self, query_vector, k, business_line=None):
        result = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(k, len(self._by_id)),
            where={"business_line": business_line} if business_line else None,
        )
        hits = []
        for cid, distance in zip(result["ids"][0], result["distances"][0]):
            # Chroma returns cosine *distance*; convert to similarity for parity
            # with the FAISS inner-product scores.
            hits.append(SearchHit(chunk=self._by_id[cid], score=1.0 - float(distance)))
        return hits


STORES = {"faiss": FaissStore, "chromadb": ChromaStore}


def get_store(name: str, **kwargs) -> VectorStore:
    if name not in STORES:
        raise ValueError(f"unknown store {name!r}; expected one of {sorted(STORES)}")
    return STORES[name](**kwargs)
