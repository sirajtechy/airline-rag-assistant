"""RAGAS and DeepEval judging, wired to local Ollama models.

The two frameworks are deliberately given **different judge models from
different families** — qwen2.5:7b for RAGAS, llama3.1:8b for DeepEval. The
methodology's advice to treat them as two independent judges only means anything
if they are actually independent; running both on the same model would measure
one model's opinion twice and call the agreement corroboration.

Judge selection was measured, not assumed: the qwen3 family emits hidden
reasoning tokens even when constrained to JSON, costing 120-300 s per call
through Ollama's /v1 endpoint, which is unaffordable across hundreds of judge
calls. See AGENTS.md for the latency table.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .config import (
    GOOGLE_API_KEY,
    USE_GEMINI_JUDGE,
    OLLAMA_BASE_URL,
    DEEPEVAL_JUDGE_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_V1_URL,
    RAGAS_JUDGE_MODEL,
)

JUDGE_TIMEOUT = 900

# Gemini's free tier allows 15 generate_content requests per minute *per model*.
# Both judges are throttled below that in-process rather than relying on retries:
# a 429 costs a ~48 s server-mandated backoff, so pacing requests is far cheaper
# than provoking and then absorbing rate-limit errors.
#
# The two frameworks use different models (gemini-3.6-flash vs
# gemini-3.5-flash-lite), so they draw on separate quotas and can run
# concurrently without contending for the same budget.
FREE_TIER_RPM = 15
JUDGE_RPM = 12  # headroom for retries and for RAGAS's internal parallelism


class _RateLimiter:
    """Minimal thread-safe token bucket, one instance per judge model."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if wait:
            time.sleep(wait)


_LIMITERS: dict[str, _RateLimiter] = {}
_LIMITER_LOCK = threading.Lock()


def limiter_for(model: str, rpm: int = JUDGE_RPM) -> _RateLimiter:
    with _LIMITER_LOCK:
        if model not in _LIMITERS:
            _LIMITERS[model] = _RateLimiter(rpm)
        return _LIMITERS[model]


def _shim_vertexai() -> None:
    """Satisfy an unconditional Vertex AI import inside some RAGAS releases.

    RAGAS imports ``langchain_community.chat_models.vertexai`` while resolving its
    LLM wrappers, but current langchain-community no longer ships that module, so
    the import fails even though this project never touches Vertex AI. The
    course's own requirements.txt notes the same problem. A stub module is
    registered rather than pinning an older langchain-community, which would drag
    the rest of the stack backwards.
    """
    for name in ("langchain_community.chat_models.vertexai",
                 "langchain_community.llms.vertexai"):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)

        class _Unavailable:  # pragma: no cover - never instantiated
            def __init__(self, *a, **k):
                raise RuntimeError("Vertex AI is not used in this project")

        module.ChatVertexAI = _Unavailable
        module.VertexAI = _Unavailable
        module.VertexAIEmbeddings = _Unavailable
        sys.modules[name] = module


@dataclass
class EvalRecord:
    """One question's worth of material for the judges."""

    question: str
    answer: str
    contexts: list[str]
    ideal_answer: str


# ── RAGAS ────────────────────────────────────────────────────────────────────
def _ragas_llm(model: str):
    """Wrap the judge model for RAGAS, Gemini if configured else local Ollama."""
    _shim_vertexai()
    from ragas.llms import LangchainLLMWrapper

    if USE_GEMINI_JUDGE and model.startswith("gemini"):
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from langchain_google_genai import ChatGoogleGenerativeAI

        return LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=model, google_api_key=GOOGLE_API_KEY, temperature=0.0,
                max_retries=6,
                # Gemini 3.x Flash reasons by default and warns that temperature is
                # ignored. Judging is a short structured classification, not a
                # reasoning task, so the thinking budget is zeroed: it removes a
                # large latency tax and makes the judge deterministic, which a
                # measurement instrument needs to be.
                thinking_budget=0,
                # Paces requests under the free tier's per-model RPM ceiling.
                rate_limiter=InMemoryRateLimiter(
                    requests_per_second=JUDGE_RPM / 60.0,
                    check_every_n_seconds=0.25,
                    max_bucket_size=2,
                ),
            )
        )

    from langchain_openai import ChatOpenAI

    return LangchainLLMWrapper(
        ChatOpenAI(model=model, base_url=OLLAMA_V1_URL, api_key=OLLAMA_API_KEY,
                   temperature=0.0, timeout=JUDGE_TIMEOUT, max_retries=2)
    )


def _ragas_embeddings():
    """Local sentence-transformers embeddings for answer-relevancy similarity.

    RAGAS needs an embedding model as well as a judge LLM. Reusing the pipeline's
    own embedder would let a retrieval choice quietly influence the generation
    score, so a separate general-purpose model is used.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )


def run_ragas(records: list[EvalRecord], model: str = RAGAS_JUDGE_MODEL) -> dict:
    """Faithfulness, answer relevancy, context precision and context recall."""
    _shim_vertexai()  # must precede the ragas import, not just the LLM construction
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict({
        "question": [r.question for r in records],
        "answer": [r.answer for r in records],
        "contexts": [r.contexts for r in records],
        "ground_truth": [r.ideal_answer for r in records],
        "reference": [r.ideal_answer for r in records],
        "user_input": [r.question for r in records],
        "response": [r.answer for r in records],
        "retrieved_contexts": [r.contexts for r in records],
    })

    from ragas.run_config import RunConfig

    # RAGAS defaults to a 180 s per-job timeout, which a local 7B judge blows
    # through on faithfulness and context precision (both decompose the answer
    # into claims and verify each one). Those jobs silently became NaN. Raising
    # the timeout and widening the worker pool also lets Ollama overlap requests
    # instead of idling between them.
    run_config = RunConfig(timeout=1800, max_workers=2, max_retries=8)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=_ragas_llm(model),
        embeddings=_ragas_embeddings(),
        raise_exceptions=False,
        run_config=run_config,
    )

    scores: dict[str, float] = {}
    for key, values in result.to_pandas().items():
        if key in {"faithfulness", "answer_relevancy", "context_precision",
                   "context_recall"}:
            numeric = [v for v in values if isinstance(v, (int, float)) and v == v]
            scores[key] = sum(numeric) / len(numeric) if numeric else float("nan")
    scores["judge_model"] = model
    return scores


# ── DeepEval ─────────────────────────────────────────────────────────────────
def _deepeval_model(model: str):
    """Build the DeepEval judge, Gemini if configured else local Ollama.

    Note ``GPTModel`` cannot be used for the local path: it validates the model
    name against a hard-coded OpenAI whitelist and rejects anything else, so
    pointing it at Ollama's /v1 endpoint is not sufficient. ``OllamaModel`` talks
    to the native API and accepts any local tag.
    """
    if USE_GEMINI_JUDGE and model.startswith("gemini"):
        from deepeval.models import GeminiModel

        limiter = limiter_for(model)

        class ThrottledGeminiModel(GeminiModel):
            """GeminiModel with free-tier pacing and 429 retry.

            DeepEval exposes no rate-limit hook, so generate/a_generate are wrapped
            directly. On a 429 the API returns a required wait (~48 s); that is
            honoured rather than retried immediately, which would just burn quota.
            """

            def _throttled(self, fn, *args, **kwargs):
                for attempt in range(5):
                    limiter.acquire()
                    try:
                        return fn(*args, **kwargs)
                    except Exception as exc:
                        if "RESOURCE_EXHAUSTED" not in str(exc) and "429" not in str(exc):
                            raise
                        backoff = 20.0 * (attempt + 1)
                        print(f"    judge rate-limited, waiting {backoff:.0f}s",
                              flush=True)
                        time.sleep(backoff)
                raise RuntimeError("judge exhausted rate-limit retries")

            def generate(self, *args, **kwargs):
                return self._throttled(super().generate, *args, **kwargs)

            async def a_generate(self, *args, **kwargs):
                limiter.acquire()
                return await super().a_generate(*args, **kwargs)

        return ThrottledGeminiModel(
            model_name=model, api_key=GOOGLE_API_KEY, temperature=0.0
        )

    from deepeval.models import OllamaModel

    return OllamaModel(model=model, base_url=OLLAMA_BASE_URL, temperature=0.0)


def run_deepeval(records: list[EvalRecord], model: str = DEEPEVAL_JUDGE_MODEL,
                 max_workers: int = 2) -> dict:
    """Faithfulness, answer relevancy, hallucination and a custom G-Eval."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    judge = _deepeval_model(model)

    # The custom criterion encodes what actually matters for this bot: a support
    # answer that is fluent but drops the dollar amount or the deadline has
    # failed the customer, even though a generic relevancy metric would pass it.
    policy_accuracy = GEval(
        name="PolicyAccuracy",
        criteria=(
            "Assess whether the answer accurately reflects Delta's policy as stated in "
            "the retrieved context. Award high scores only when specific figures, "
            "deadlines, dollar amounts and conditions match the context exactly. "
            "Penalise heavily any invented fee, rule or deadline, and penalise omission "
            "of a materially important condition or exception."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
    )

    metrics = {
        "faithfulness": FaithfulnessMetric(model=judge, threshold=0.7),
        "answer_relevancy": AnswerRelevancyMetric(model=judge, threshold=0.7),
        "hallucination": HallucinationMetric(model=judge, threshold=0.3),
        "g_eval_policy_accuracy": policy_accuracy,
    }

    def score_one(record: EvalRecord) -> dict[str, float]:
        """Score one record. Metrics are stateful, so each thread builds its own."""
        judge_local = _deepeval_model(model)
        local_metrics = {
            "faithfulness": FaithfulnessMetric(model=judge_local, threshold=0.7),
            "answer_relevancy": AnswerRelevancyMetric(model=judge_local, threshold=0.7),
            "hallucination": HallucinationMetric(model=judge_local, threshold=0.3),
            "g_eval_policy_accuracy": GEval(
                name="PolicyAccuracy", criteria=policy_accuracy.criteria,
                evaluation_params=policy_accuracy.evaluation_params, model=judge_local,
            ),
        }
        # Faithfulness/AnswerRelevancy/G-Eval judge the answer against what was
        # actually retrieved, so they take `retrieval_context`.
        retrieval_case = LLMTestCase(
            input=record.question,
            actual_output=record.answer,
            expected_output=record.ideal_answer,
            retrieval_context=record.contexts,
            context=record.contexts,
        )
        # HallucinationMetric is different, and getting this wrong inflates the
        # score badly. It emits one verdict *per context document* and returns
        # hallucination_count / number_of_verdicts. Handed the 5 retrieved chunks,
        # a correct answer that legitimately draws on only one of them scores as
        # contradicting the other four — measuring retrieval breadth, not
        # hallucination. DeepEval intends `context` to be ground truth, so the
        # curated ideal answer is used instead.
        hallucination_case = LLMTestCase(
            input=record.question,
            actual_output=record.answer,
            expected_output=record.ideal_answer,
            context=[record.ideal_answer],
        )
        out: dict[str, float] = {}
        for name, metric in local_metrics.items():
            case = hallucination_case if name == "hallucination" else retrieval_case
            try:
                metric.measure(case)
                if metric.score is not None:
                    out[name] = float(metric.score)
            except Exception as exc:
                print(f"    deepeval {name} failed on {record.question[:40]!r}: "
                      f"{type(exc).__name__}", flush=True)
        return out

    # Records are independent, and a local judge spends nearly all its time waiting
    # on Ollama, so overlapping them is close to free. Sequentially this pass took
    # ~175 s per record.
    totals: dict[str, list[float]] = {name: [] for name in metrics}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(score_one, records):
            for name, value in result.items():
                totals[name].append(value)

    scores = {
        name: (sum(v) / len(v) if v else float("nan")) for name, v in totals.items()
    }
    scores["judge_model"] = model
    return scores
