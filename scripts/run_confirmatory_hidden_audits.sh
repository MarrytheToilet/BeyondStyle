#!/usr/bin/env bash
# Run the bounded confirmatory hidden-state audit for one target model.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {Qwen3-4B|Llama-3.1-8B-Instruct|gemma-4-E2B-it}" >&2
  exit 2
fi

MODEL_ID="$1"
MODEL_ROOT="${MODEL_ROOT:-../../models}"
MAX_PROMPTS="${MAX_PROMPTS:-100}"
NULL_CONTROLS="${NULL_CONTROLS:-100}"
RANDOM_CONTROLS="${RANDOM_CONTROLS:-8}"

case "$MODEL_ID" in
  Qwen3-4B)
    INJECTION_LAYER=8
    ;;
  Llama-3.1-8B-Instruct)
    INJECTION_LAYER=7
    ;;
  gemma-4-E2B-it)
    INJECTION_LAYER=8
    ;;
  *)
    echo "unsupported model: $MODEL_ID" >&2
    exit 2
    ;;
esac

PULLBACK_ROOT="outputs/en/05_linguistic/causal_pullback/${MODEL_ID}/mean_pool"
GENERATION_ROOT="${GENERATION_ROOT:-outputs/en/03_steering_vectors/${MODEL_ID}/proposition_aligned/mean_pool/causal_pullback_confirmatory_generation}"
ANCHOR_COMPONENT="${ANCHOR_COMPONENT:-random}"
ANCHOR_STRENGTH="${ANCHOR_STRENGTH:--1}"
AUDIT_ROOT="${PULLBACK_ROOT}/confirmatory_hidden_audit"
LAYERWISE_OUT="${AUDIT_ROOT}/layerwise_transport"
MATCHED_NULL_OUT="${AUDIT_ROOT}/matched_nulls"
INJECTION_DIRECTIONS="${PULLBACK_ROOT}/frozen_axes/layer$(printf '%02d' "$INJECTION_LAYER")/directions"
MEAN_DIFFERENCE="outputs/en/03_steering_vectors/${MODEL_ID}/proposition_aligned/mean_pool/models/mean_diff_layer$(printf '%02d' "$INJECTION_LAYER").pt"

python -u scripts/audit_layerwise_concept_transport.py \
  --model "${MODEL_ROOT}/${MODEL_ID}" \
  --prompts data/prompts/en/formal.json \
  --generation-root "$GENERATION_ROOT" \
  --injection-directions-dir "$INJECTION_DIRECTIONS" \
  --layerwise-root "outputs/en/05_linguistic/concept_patterns_layerwise/${MODEL_ID}/mean_pool" \
  --mean-difference-checkpoint "$MEAN_DIFFERENCE" \
  --anchor-component "$ANCHOR_COMPONENT" \
  --anchor-strength "$ANCHOR_STRENGTH" \
  --max-prompts "$MAX_PROMPTS" \
  --random-controls "$RANDOM_CONTROLS" \
  --trajectory-direction-batch-size 4 \
  --torch-dtype bf16 \
  --attention-backend sdpa \
  --out-dir "$LAYERWISE_OUT"

python -u scripts/audit_pullback_matched_nulls.py \
  --model "${MODEL_ROOT}/${MODEL_ID}" \
  --model-id "$MODEL_ID" \
  --generation-root "$GENERATION_ROOT" \
  --canonical-root "${PULLBACK_ROOT}/frozen_axes" \
  --learned-transport "${LAYERWISE_OUT}/per_prompt_transport.csv" \
  --anchor-component "$ANCHOR_COMPONENT" \
  --anchor-strength "$ANCHOR_STRENGTH" \
  --null-controls "$NULL_CONTROLS" \
  --max-pairs 32 \
  --pullback-batch-size 2 \
  --readout-chunk-size 4 \
  --max-prompts "$MAX_PROMPTS" \
  --trajectory-direction-batch-size 4 \
  --torch-dtype bf16 \
  --attention-backend sdpa \
  --out-dir "$MATCHED_NULL_OUT"

echo "[complete] ${MODEL_ID} confirmatory hidden-state audit"
