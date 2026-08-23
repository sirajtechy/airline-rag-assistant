"""FastAPI backend for the Delta support assistant.

Thin transport layer over ``delta_rag.pipeline`` — all retrieval, routing,
guardrail and generation logic lives in the package so it stays testable without
HTTP, and so the notebooks and the app cannot drift apart.

The pipeline is built once at startup because loading the embedding model and
building the FAISS index takes ~18 s; doing it per request would be absurd.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_rag.config import BUSINESS_LINES, SOURCE_DOCS
from delta_rag.pipeline import DEFAULT_GENERATION_MODEL, DeltaSupportBot

STATE: dict[str, DeltaSupportBot] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("building pipeline (embedding corpus + FAISS index)…")
    STATE["bot"] = DeltaSupportBot()
    print(f"ready: {STATE['bot'].config.label()} | generator={STATE['bot'].model}")
    yield
    STATE.clear()


app = FastAPI(
    title="Delta Air Lines Support Assistant",
    description="RAG over Delta's Contract of Carriage, Cargo tariff and SEC filings.",
    version="1.0.0",
    lifespan=lifespan,
)

# The frontend is a separate Vite dev server, so it needs CORS in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    model: str | None = None


class SourceOut(BaseModel):
    n: int
    citation: str
    doc_id: str
    business_line: str
    locators: list[str]
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    route: str | None
    blocked: bool
    reason: str | None
    sources: list[SourceOut]
    retrieval_ms: float
    generation_ms: float
    model: str
    citations_valid: bool


@app.get("/health")
def health() -> dict:
    bot = STATE.get("bot")
    return {
        "status": "ok" if bot else "starting",
        "config": bot.config.label() if bot else None,
        "generator": bot.model if bot else None,
        "chunks": len(bot.chunks) if bot else 0,
    }


@app.get("/config")
def config() -> dict:
    """The measured pipeline configuration and the corpus it serves."""
    bot = STATE["bot"]
    return {
        "retrieval": {
            "parser": bot.config.parser,
            "chunking": f"{bot.config.strategy} / {bot.config.size} tokens / "
                        f"{bot.config.overlap} overlap",
            "embedding": bot.config.embedding,
            "vector_store": bot.config.store,
            "mode": bot.config.mode,
            "fusion": bot.config.fusion,
            "reranker": "none (measured to hurt NDCG@3)",
        },
        "generator": bot.model,
        "business_lines": list(BUSINESS_LINES),
        "documents": [
            {"doc_id": k, "title": v["title"], "business_line": v["business_line"]}
            for k, v in SOURCE_DOCS.items()
        ],
        "chunks": len(bot.chunks),
    }


@app.get("/examples")
def examples() -> dict:
    """Suggested questions, one per business line plus the guardrail cases."""
    return {
        "passenger": [
            "What's Delta's policy on oversold flights?",
            "Can I get a refund on a non-refundable Delta ticket?",
            "What happens if being denied boarding means I stay overnight?",
        ],
        "cargo": [
            "What are the packaging requirements for shipping cargo with Delta?",
            "How long do I have to file a claim if my shipment is damaged?",
            "Can I ship jewellery or gold bullion with Delta Cargo?",
        ],
        "financial": [
            "What's Delta's total operating revenue in their latest 10-K?",
            "How much revenue did Delta make from shipping cargo last year?",
        ],
        "guardrails": [
            "What's JetBlue's baggage policy?",
            "What's the status of my flight tomorrow?",
        ],
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    bot = STATE.get("bot")
    if bot is None:
        raise HTTPException(status_code=503, detail="pipeline still starting")

    if request.model:
        bot.model = request.model
    try:
        response = bot.answer(request.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc

    return AskResponse(
        question=response.question,
        answer=response.answer,
        route=response.route,
        blocked=response.blocked,
        reason=response.reason,
        sources=[SourceOut(**s) for s in response.sources],
        retrieval_ms=round(response.retrieval_ms, 1),
        generation_ms=round(response.generation_ms, 1),
        model=response.model,
        citations_valid=response.citations_valid,
    )
