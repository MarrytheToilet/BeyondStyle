#!/usr/bin/env bash
# Canonical hidden-state extraction, formal separability evaluation, and
# information-dense geometry figures for one language.
set -euo pipefail

cd "$(dirname "$0")/.."

LANGUAGE="${1:-en}"
if [[ "$LANGUAGE" != "en" && "$LANGUAGE" != "zh" ]]; then
  echo "Usage: $0 [en|zh]" >&2
  exit 2
fi

DATASET="data/paired/${LANGUAGE}/proposition_aligned/pairs.jsonl"
DATASET_TAG="proposition_aligned"
MODEL_ROOT="${MODEL_ROOT:-../../models}"
EXTRACTION_BATCH_SIZE="${EXTRACTION_BATCH_SIZE:-4}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
if [[ "$LANGUAGE" == "zh" ]]; then
  MODELS=(
    "${MODEL_ROOT}/Qwen3-4B"
    "${MODEL_ROOT}/Qwen3.5-4B"
    "${MODEL_ROOT}/gemma-4-E2B-it"
  )
else
  MODELS=(
    "${MODEL_ROOT}/Llama-3.1-8B-Instruct"
    "${MODEL_ROOT}/Qwen3-4B"
    "${MODEL_ROOT}/gemma-4-E2B-it"
  )
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Missing canonical dataset: $DATASET" >&2
  exit 1
fi

for MODEL_PATH in "${MODELS[@]}"; do
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Missing model directory: $MODEL_PATH" >&2
    exit 1
  fi
  MODEL_ID="$(basename "$MODEL_PATH")"
  echo "[extract] ${LANGUAGE} ${MODEL_ID}"
  python -u scripts/steering_pipeline.py \
    --multi \
    --language "$LANGUAGE" \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --poolings last full_mean \
    --denoise nullspace raw \
    --skip_generation \
    --extraction_batch_size "$EXTRACTION_BATCH_SIZE" \
    --torch_dtype "$TORCH_DTYPE" \
    --attention_backend "$ATTENTION_BACKEND"
done

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_ID="$(basename "$MODEL_PATH")"
  for REPRESENTATION in last_token mean_pool; do
    STATES_GLOB="outputs/${LANGUAGE}/03_steering_vectors/${MODEL_ID}/${DATASET_TAG}/${REPRESENTATION}/states/paired_states_layer*.pt"
    EVAL_DIR="outputs/${LANGUAGE}/02_separability/${MODEL_ID}/${DATASET_TAG}/${REPRESENTATION}"
    mkdir -p "$EVAL_DIR"

    echo "[random holdout] ${LANGUAGE} ${MODEL_ID} ${REPRESENTATION}"
    python -u scripts/eval_heldout_separability.py \
      --states_glob "$STATES_GLOB" \
      --out "${EVAL_DIR}/random_holdout.json" \
      --n_runs 5 \
      --test_frac 0.2 \
      --null_k 10 \
      --seed 8236

    echo "[generator holdout] ${LANGUAGE} ${MODEL_ID} ${REPRESENTATION}"
    python -u scripts/eval_generator_holdout.py \
      --dataset "$DATASET" \
      --states_glob "$STATES_GLOB" \
      --out "${EVAL_DIR}/generator_holdout.json" \
      --null_k 10 \
      --n_boot 200 \
      --seed 8236
  done
done

echo "[geometry] ${LANGUAGE}"
MODEL_IDS=()
for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_IDS+=("$(basename "$MODEL_PATH")")
done
python -u scripts/analyze_hidden_geometry.py \
  --states-root "outputs/${LANGUAGE}/03_steering_vectors" \
  --eval-root "outputs/${LANGUAGE}/02_separability" \
  --dataset "$DATASET" \
  --dataset-tag "$DATASET_TAG" \
  --models "${MODEL_IDS[@]}" \
  --out-dir "outputs/${LANGUAGE}/02_separability/hidden_geometry" \
  --bootstrap 200 \
  --permutations 200 \
  --seed 20260724

echo "[complete] ${LANGUAGE} hidden-state pipeline"
