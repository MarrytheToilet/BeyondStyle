#!/usr/bin/env bash
# Reproduce the canonical learned axes and eight-axis behavior null ensemble.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ROOT="${MODEL_ROOT:-../../models}"
PROMPTS="${PROMPTS:-data/prompts/en/formal.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
COMPONENTS=(
  lexical
  syntactic
  random
  random_01
  random_02
  random_03
  random_04
  random_05
  random_06
  random_07
)

run_model() {
  local model_id="$1"
  local model_path="${MODEL_ROOT}/${model_id}"
  local pullback_root="outputs/en/05_linguistic/causal_pullback/${model_id}/mean_pool"
  local generation_root="outputs/en/03_steering_vectors/${model_id}/proposition_aligned/mean_pool/causal_pullback_confirmatory_generation"

  python -u scripts/run_steering_generation.py \
    --model "$model_path" \
    --language en \
    --representations mean_pool \
    --dataset-tag proposition_aligned \
    --prompts "$PROMPTS" \
    --components "${COMPONENTS[@]}" \
    --component-family causal_pullback \
    --component-root "${pullback_root}/frozen_axes" \
    --strengths -1 1 \
    --prompt-start 34 \
    --max-prompts 166 \
    --num-seeds 1 \
    --base-seed 8236 \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --torch-dtype bf16 \
    --attention-backend sdpa \
    --output-subdir causal_pullback_confirmatory_generation

  python -u scripts/analyze_component_steering.py \
    --input-dir "$generation_root" \
    --observational-features \
      outputs/en/05_linguistic/features/paired_feature_differences.csv \
    --out-dir "${pullback_root}/confirmatory_analysis" \
    --components "${COMPONENTS[@]}" \
    --semantic-model "${MODEL_ROOT}/bge-base-en-v1.5" \
    --semantic-device cpu \
    --bootstrap 10000 \
    --permutations 20000
}

run_model Qwen3-4B
run_model Llama-3.1-8B-Instruct

echo "[complete] canonical learned axes and behavior null ensemble"
