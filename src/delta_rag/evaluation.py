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

from dataclasses import dataclass

from .config import (
    DEEPEVAL_JUDGE_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_V1_URL,
    RAGAS_JUDGE_MODEL,
)

JUDGE_TIMEOUT = 900


@dataclass
class EvalRecord:
    """One question's worth of material for the judges."""

    question: str
    answer: str
    contexts: list[str]
    ideal_answer: str


# ── RAGAS ────────────────────────────────────────────────────────────────────
def _ragas_llm(model: str):
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

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

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=_ragas_llm(model),
        embeddings=_ragas_embeddings(),
        raise_exceptions=False,
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
    """DeepEval speaks OpenAI; point it at Ollama's compatible endpoint."""
    from deepeval.models import GPTModel

    return GPTModel(model=model, base_url=OLLAMA_V1_URL, api_key=OLLAMA_API_KEY)


def run_deepeval(records: list[EvalRecord], model: str = DEEPEVAL_JUDGE_MODEL) -> dict:
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

    totals: dict[str, list[float]] = {name: [] for name in metrics}
    for record in records:
        test_case = LLMTestCase(
            input=record.question,
            actual_output=record.answer,
            expected_output=record.ideal_answer,
            retrieval_context=record.contexts,
            context=record.contexts,  # HallucinationMetric reads `context`
        )
        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                if metric.score is not None:
                    totals[name].append(float(metric.score))
            except Exception as exc:
                print(f"    deepeval {name} failed on {record.question[:40]!r}: "
                      f"{type(exc).__name__}")

    scores = {
        name: (sum(v) / len(v) if v else float("nan")) for name, v in totals.items()
    }
    scores["judge_model"] = model
    return scores
