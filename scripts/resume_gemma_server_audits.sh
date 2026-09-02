#!/usr/bin/env bash
# Resume only the two independent Gemma audits after behavior generation.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="gemma-4-E2B-it"
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
RUNTIME_BIN="${RUNTIME_BIN:-/root/autodl-tmp/envs/beyondstyle-runtime/bin}"
LOG_DIR="${LOG_DIR:-outputs/en/05_linguistic/revision_bundle_logs}"
FIXED_GENERATION_ROOT="outputs/en/03_steering_vectors/${MODEL_ID}/proposition_aligned/mean_pool/causal_pullback_fixed_trajectory_generation"
mkdir -p "$LOG_DIR"

# Several strong Gemma directions terminate immediately. The fixed-token
# estimand therefore uses an independently generated, unsteered (alpha=0)
# continuation as its common token trajectory.
if [[ ! -f "${FIXED_GENERATION_ROOT}/generation_component_lexical.json" ]]; then
  echo "missing unsteered Gemma fixed trajectories" >&2
  exit 1
fi

export PATH="${RUNTIME_BIN}:${PATH}"

CUDA_VISIBLE_DEVICES=0 \
  MODEL_ROOT="$MODEL_ROOT" \
  GENERATION_ROOT="$FIXED_GENERATION_ROOT" \
  ANCHOR_COMPONENT=lexical \
  ANCHOR_STRENGTH=0 \
  MAX_PROMPTS=100 \
  NULL_CONTROLS=100 \
  RANDOM_CONTROLS=8 \
  bash scripts/run_confirmatory_hidden_audits.sh "$MODEL_ID" \
  >"${LOG_DIR}/gemma_hidden_recovery.log" 2>&1 &
hidden_pid=$!

CUDA_VISIBLE_DEVICES=1 \
  MODEL_ROOT="$MODEL_ROOT" \
  EXTRACTION_BATCH_SIZE=12 \
  bash scripts/run_prefix_only_hidden_control.sh "$MODEL_ID" \
  >"${LOG_DIR}/gemma_prefix_only.log" 2>&1 &
prefix_pid=$!

status=0
wait "$hidden_pid" || status=$?
wait "$prefix_pid" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "Gemma recovery audit failed with status ${status}" >&2
  exit "$status"
fi

python -u scripts/evaluate_prefix_only_transfer.py \
  --model-specs \
    Llama-3.1-8B-Instruct:7:32 \
    Qwen3-4B:8:36 \
    gemma-4-E2B-it:8:35
python -u scripts/summarize_compute_cost.py
python scripts/build_acl_result_tables.py
python -u scripts/record_server_snapshot.py

echo "[complete] Gemma hidden, prefix-only, and final table recovery"
