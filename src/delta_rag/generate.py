"""Grounded answer generation with citations.

The prompt is deliberately strict. This bot answers questions about refund
eligibility and denied-boarding compensation — a plausible-sounding invented
number is a materially worse failure than an admission of ignorance, so the
system prompt forbids outside knowledge and requires every factual claim to
carry a bracketed source marker.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from .config import OLLAMA_BASE_URL
from .corpus import citation_for
from .stores import SearchHit

# Ollama defaults to each model's *maximum* context window when it is not told
# otherwise — 131,072 tokens for Llama 3.1, which reserves ~22 GB of GPU memory
# and made generation take 30 s per answer. A grounded answer here needs the
# system prompt plus ~5 retrieved chunks, comfortably under 8k tokens, so the
# window is capped explicitly. This is set through Ollama's native API because
# its OpenAI-compatible /v1 surface exposes no equivalent option.
NUM_CTX = 8192
NUM_PREDICT = 700  # answers average ~520 chars; this is generous but bounded

SYSTEM_PROMPT = """You are Delta Air Lines' customer support assistant.

You answer ONLY from the numbered sources provided in each request. Those sources \
are Delta's own published documents: the Domestic Contract of Carriage (passenger \
ticket rules), the Delta Cargo U.S. domestic Shipping Rules & Tariffs, and Delta's \
SEC financial filings.

Rules you must follow:
1. Use ONLY the numbered sources given to you. Never use outside knowledge about \
Delta or any other airline.
2. Cite every factual claim with its source number in square brackets, like [1] or \
[2]. A sentence stating a rule, fee, deadline or dollar amount must carry a citation.
3. NEVER invent a policy, fee, deadline, dollar amount or rule. If the sources do \
not contain the answer, say plainly that Delta's published documents you have access \
to do not cover it.
4. Never answer about another airline. You only know Delta.
5. Quote exact figures, deadlines and dollar amounts from the sources rather than \
paraphrasing them loosely. Precision matters for these rules.
6. Be direct and helpful. Lead with the answer, then the relevant conditions or \
exceptions. Use plain language a customer understands, not tariff jargon.

These are U.S. DOMESTIC documents. If asked about international travel or shipping, \
say that your sources cover domestic rules only."""

USER_TEMPLATE = """Sources:
{sources}

Customer question: {question}

Answer using only the sources above, citing each claim with [n]."""


@dataclass
class GeneratedAnswer:
    text: str
    model: str
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sources: list[dict] = field(default_factory=list)


def format_sources(hits: Sequence[SearchHit], max_chars: int = 1800) -> tuple[str, list[dict]]:
    """Render retrieved chunks as a numbered, citable source block."""
    blocks, meta = [], []
    for i, hit in enumerate(hits, start=1):
        unit_like = type("U", (), {
            "locator": hit.chunk.locators[0],
            "title": hit.chunk.title,
            "meta": hit.chunk.meta,
        })
        citation = citation_for(unit_like)
        text = hit.chunk.text[:max_chars]
        blocks.append(f"[{i}] {citation}\n{text}")
        meta.append({
            "n": i,
            "citation": citation,
            "doc_id": hit.chunk.doc_id,
            "business_line": hit.chunk.business_line,
            "locators": list(hit.chunk.locators),
            "score": float(hit.score),
        })
    return "\n\n".join(blocks), meta


def generate(
    question: str,
    hits: Sequence[SearchHit],
    model: str,
    temperature: float = 0.0,
    timeout: int = 600,
) -> GeneratedAnswer:
    """Generate a grounded answer via Ollama's native chat API.

    The native API is used rather than the OpenAI-compatible one purely so the
    context window can be bounded; see NUM_CTX above.
    """
    import ollama

    sources_text, source_meta = format_sources(hits)
    client = ollama.Client(host=OLLAMA_BASE_URL, timeout=timeout)

    start = time.perf_counter()
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                sources=sources_text, question=question)},
        ],
        options={
            "temperature": temperature,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
        think=False,  # ignored by non-reasoning models; suppresses it on qwen3
    )
    elapsed = time.perf_counter() - start

    return GeneratedAnswer(
        text=(response["message"]["content"] or "").strip(),
        model=model,
        seconds=elapsed,
        prompt_tokens=response.get("prompt_eval_count", 0) or 0,
        completion_tokens=response.get("eval_count", 0) or 0,
        sources=source_meta,
    )
