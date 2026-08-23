#!/usr/bin/env bash
# Create Ollama derivatives of the judge models with a bounded context window.
#
# Ollama allocates each model's *maximum* context when not told otherwise
# (131,072 tokens for Llama 3.1, reserving ~22 GB), which made RAGAS/DeepEval
# judging ~175 s per record. RAGAS reaches its judge via langchain-openai and
# DeepEval via its own OllamaModel; neither exposes num_ctx, so the cap has to
# live in the model definition to cover both paths.
set -euo pipefail
for base in qwen2.5:7b llama3.1:latest; do
  name="$(echo "$base" | tr ':' '-')-judge"
  printf 'FROM %s\nPARAMETER num_ctx 8192\nPARAMETER temperature 0\n' "$base" \
    > "/tmp/Modelfile.$name"
  ollama create "$name" -f "/tmp/Modelfile.$name"
  echo "created $name"
done
