# Stage 8 — LLM for generation

| LLM | Faithfulness (RAGAS) | Answer Relevancy (RAGAS) | Context Precision (RAGAS) | Context Recall (RAGAS) | Faithfulness (DeepEval) | Hallucination (DeepEval, lower=better) | G-Eval PolicyAccuracy (DeepEval) | Route acc | Valid citations | Latency (ms/answer) | Cost/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5 7B | 0.8986 | 0.7404 | 0.7676 | 0.9139 | 0.6644 | 0.5868 | 0.9867 | 1.0000 | 0.9333 | 16414 | $0.00 (local) |
| Llama 3.2 3B | 0.7829 | 0.6560 | 0.7676 | 0.9139 | 0.6359 | 0.5985 | 0.9000 | 1.0000 | 0.7333 | 8420 | $0.00 (local) |

## Reading of these numbers

Retrieval is byte-identical across every row — same parser, chunking, embeddings, store and fusion — so these differences are attributable to the generator alone. Context Precision and Context Recall are properties of the shared retrieval step and should be near-constant down the table; any spread in them is judge noise, which is itself a useful read on how much to trust the other columns.

**Two independent judges.** RAGAS is judged by `qwen2.5-7b-judge:latest` and DeepEval by `llama3.1-latest-judge:latest` — deliberately different model families. Where the two faithfulness columns agree, the score is trustworthy; where they diverge sharply, the honest conclusion is that neither number is reliable rather than that the flattering one is right.

**Cost is $0.00 for every candidate** because everything runs locally on Ollama, so cost cannot discriminate between models here. Latency and quality do the work instead. The G-Eval criterion is custom-written for this bot: it rewards exact figures, deadlines and dollar amounts and penalises invented fees, because a fluent answer that loses the $1,075 cap has failed the customer even though a generic relevancy metric would pass it.

Guardrail-blocked answers are excluded from the generation metrics: they have no retrieval context to be faithful to, so scoring them would penalise a model for the guardrail working.
