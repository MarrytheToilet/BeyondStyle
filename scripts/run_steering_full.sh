#!/usr/bin/env bash
# Canonical resumable steering generation for all formal EN/ZH target models.
# Prerequisite: run_hidden_formal.sh has produced directions and held-out
# generator evaluations.
set -euo pipefail

cd "$(dirname "$0")/.."

SCOPE="${1:-all}"
if [[ "$SCOPE" != "all" && "$SCOPE" != "en" && "$SCOPE" != "zh" ]]; then
  echo "Usage: $0 [all|en|zh]" >&2
  exit 2
fi

MODEL_ROOT="${MODEL_ROOT:-../../models}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_SEEDS="${NUM_SEEDS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
read -r -a STRENGTHS <<< "${STEERING_STRENGTHS:--1 1}"

run_language() {
  local language="$1"
  local prompts="data/prompts/${language}/formal.json"
  local model_names=()
  if [[ "$language" == "en" ]]; then
    model_names=(
      "Llama-3.1-8B-Instruct"
      "Qwen3-4B"
      "gemma-4-E2B-it"
    )
  else
    model_names=(
      "Qwen3-4B"
      "Qwen3.5-4B"
      "gemma-4-E2B-it"
    )
  fi

  if [[ ! -f "$prompts" ]]; then
    echo "Missing formal prompts: $prompts" >&2
    echo "Run: python scripts/build_formal_prompts.py --language $language" >&2
    exit 1
  fi

  for model_name in "${model_names[@]}"; do
    local model_path="${MODEL_ROOT}/${model_name}"
    if [[ ! -d "$model_path" ]]; then
      echo "Missing model directory: $model_path" >&2
      exit 1
    fi
    echo "[steering] ${language} ${model_name}"
    python -u scripts/run_steering_generation.py \
      --model "$model_path" \
      --language "$language" \
      --prompts "$prompts" \
      --method nullspace \
      --strengths "${STRENGTHS[@]}" \
      --num-seeds "$NUM_SEEDS" \
      --batch-size "$BATCH_SIZE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --torch-dtype "$TORCH_DTYPE" \
      --attention-backend "$ATTENTION_BACKEND"
  done
}

if [[ "$SCOPE" == "all" || "$SCOPE" == "en" ]]; then
  run_language en
fi
if [[ "$SCOPE" == "all" || "$SCOPE" == "zh" ]]; then
  run_language zh
fi

echo "[complete] formal steering generation"
