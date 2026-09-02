#!/usr/bin/env bash
# Extract mean-pooled hidden states for the matched prefix-only control.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {Qwen3-4B|Llama-3.1-8B-Instruct|gemma-4-E2B-it}" >&2
  exit 2
fi

MODEL_ID="$1"
case "$MODEL_ID" in
  Qwen3-4B|Llama-3.1-8B-Instruct|gemma-4-E2B-it)
    ;;
  *)
    echo "unsupported model: $MODEL_ID" >&2
    exit 2
    ;;
esac

MODEL_ROOT="${MODEL_ROOT:-../../models}"
EXTRACTION_BATCH_SIZE="${EXTRACTION_BATCH_SIZE:-4}"
DATASET="data/paired/en/prefix_only_control/pairs.jsonl"

if [[ ! -f "$DATASET" ]]; then
  echo "missing complete prefix-only control: $DATASET" >&2
  exit 1
fi

python -u scripts/steering_pipeline.py \
  --multi \
  --language en \
  --model "${MODEL_ROOT}/${MODEL_ID}" \
  --dataset "$DATASET" \
  --poolings full_mean \
  --denoise nullspace raw \
  --skip_generation \
  --extraction_batch_size "$EXTRACTION_BATCH_SIZE" \
  --torch_dtype bf16 \
  --attention_backend sdpa

echo "[complete] ${MODEL_ID} prefix-only hidden extraction"
