#!/usr/bin/env bash
# Replicate the preregistered surface-to-hidden chain in a third target-model
# family. Existing proposition-aligned states are reused; prefix-only states
# are extracted only if absent.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="gemma-4-E2B-it"
MODEL_ROOT="${MODEL_ROOT:-../../models}"
MODEL="${MODEL_ROOT}/${MODEL_ID}"
INJECTION_LAYER=8
ANCHOR_LAYER=35
STATE_ROOT="outputs/en/03_steering_vectors/${MODEL_ID}/proposition_aligned/mean_pool"
PATTERN_ROOT="outputs/en/05_linguistic/concept_patterns_layerwise/${MODEL_ID}/mean_pool"
PULLBACK_ROOT="outputs/en/05_linguistic/causal_pullback/${MODEL_ID}/mean_pool"

if [[ ! -d "$MODEL" ]]; then
  echo "missing model directory: $MODEL" >&2
  exit 1
fi

if [[ ! -f "${STATE_ROOT}/states/paired_states_layer08.pt" ]]; then
  echo "missing existing proposition-aligned Gemma states" >&2
  exit 1
fi

if [[ ! -f "${PATTERN_ROOT}/layer08/directions/lexical.pt" ]]; then
  echo "missing Gemma layerwise concept patterns; run build_concept_patterns.py" >&2
  exit 1
fi

python -u scripts/build_causal_pullback_directions.py \
  --model "$MODEL" \
  --pairs data/paired/en/proposition_aligned/pairs.jsonl \
  --states "${STATE_ROOT}/states/paired_states_layer08.pt" \
  --input-directions-dir "${PATTERN_ROOT}/layer08/directions" \
  --downstream-directions-dir "${PATTERN_ROOT}/layer35/directions" \
  --input-layer "$INJECTION_LAYER" \
  --downstream-layer "$ANCHOR_LAYER" \
  --max-pairs 32 \
  --batch-size 2 \
  --null-k 10 \
  --torch-dtype bf16 \
  --attention-backend sdpa \
  --seed 8236 \
  --out-dir "$PULLBACK_ROOT"

python -u scripts/freeze_causal_pullback_axes.py \
  --source-checkpoint "${STATE_ROOT}/models/mean_diff_nullspace_layer08.pt" \
  --lexical-checkpoint "${PULLBACK_ROOT}/layer08/directions/lexical.pt" \
  --syntactic-checkpoint "${PULLBACK_ROOT}/layer08/directions/syntactic.pt" \
  --sealed-prompt-start 34 \
  --random-controls 8 \
  --seed 8236 \
  --out-dir "${PULLBACK_ROOT}/frozen_axes"

# One-seed random-axis generation provides a sealed trajectory for the
# fixed-token audit and a matched behavior-null calibration.
MODEL_ROOT="$MODEL_ROOT" bash scripts/run_canonical_behavior_model.sh "$MODEL_ID"

# The confirmatory generated-behavior estimand averages three paired seeds.
MODEL_ROOT="$MODEL_ROOT" bash scripts/run_seed_robustness_audit.sh "$MODEL_ID"

# Fixed-token intermediate propagation and 100 pipeline-matched nulls.
MODEL_ROOT="$MODEL_ROOT" bash scripts/run_confirmatory_hidden_audits.sh "$MODEL_ID"

# Complete the interface-transfer test on the third model.
if [[ ! -f "outputs/en/03_steering_vectors/${MODEL_ID}/prefix_only_control/mean_pool/states/paired_states_layer35.pt" ]]; then
  MODEL_ROOT="$MODEL_ROOT" bash scripts/run_prefix_only_hidden_control.sh "$MODEL_ID"
fi
python -u scripts/evaluate_prefix_only_transfer.py \
  --model-specs \
    Llama-3.1-8B-Instruct:7:32 \
    Qwen3-4B:8:36 \
    gemma-4-E2B-it:8:35

echo "[complete] Gemma third-family causal replication"
