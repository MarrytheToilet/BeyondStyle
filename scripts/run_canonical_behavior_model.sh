#!/usr/bin/env bash
# Run the canonical learned axes plus eight behavior-null axes for one model.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${1:?usage: run_canonical_behavior_model.sh MODEL_ID}"
MODEL_ROOT="${MODEL_ROOT:-../../models}"
PROMPTS="${PROMPTS:-data/prompts/en/formal.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PULLBACK_ROOT="outputs/en/05_linguistic/causal_pullback/${MODEL_ID}/mean_pool"
GENERATION_ROOT="outputs/en/03_steering_vectors/${MODEL_ID}/proposition_aligned/mean_pool/causal_pullback_confirmatory_generation"
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

python -u scripts/run_steering_generation.py \
  --model "${MODEL_ROOT}/${MODEL_ID}" \
  --language en \
  --representations mean_pool \
  --dataset-tag proposition_aligned \
  --prompts "$PROMPTS" \
  --components "${COMPONENTS[@]}" \
  --component-family causal_pullback \
  --component-root "${PULLBACK_ROOT}/frozen_axes" \
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
  --input-dir "$GENERATION_ROOT" \
  --observational-features \
    outputs/en/05_linguistic/features/paired_feature_differences.csv \
  --out-dir "${PULLBACK_ROOT}/confirmatory_analysis" \
  --components "${COMPONENTS[@]}" \
  --semantic-model "${MODEL_ROOT}/bge-base-en-v1.5" \
  --semantic-device cpu \
  --bootstrap 10000 \
  --permutations 20000

echo "[complete] ${MODEL_ID} canonical behavior ensemble"
