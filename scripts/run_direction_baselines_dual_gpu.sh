#!/usr/bin/env bash
# Use one GPU per target model, then analyze both matched-protocol comparator
# runs after generation succeeds.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
LOG_DIR="${LOG_DIR:-outputs/en/05_linguistic/causal_pullback/comparator_logs}"
BATCH_SIZE="${BATCH_SIZE:-4}"
mkdir -p "$LOG_DIR"

run_model() {
  local gpu="$1"
  local model="$2"
  local log="$LOG_DIR/${model}.log"
  echo "[start] GPU ${gpu}: ${model}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    MODEL_ROOT="$MODEL_ROOT" \
    BATCH_SIZE="$BATCH_SIZE" \
    RUN_ANALYSIS=0 \
    bash scripts/run_direction_baseline_audit.sh "$model" \
    >"$log" 2>&1
  echo "[generated] ${model}"
}

run_model 0 Llama-3.1-8B-Instruct &
llama_pid=$!
run_model 1 Qwen3-4B &
qwen_pid=$!

wait "$llama_pid"
wait "$qwen_pid"

for model in Llama-3.1-8B-Instruct Qwen3-4B; do
  generation_root="outputs/en/03_steering_vectors/${model}/proposition_aligned/mean_pool/encoding_source_seed_robustness_generation"
  analysis_root="outputs/en/05_linguistic/causal_pullback/${model}/mean_pool/comparator_seed_robustness"
  python -u scripts/analyze_seed_robustness.py \
    --input-dir "$generation_root" \
    --out-dir "$analysis_root" \
    --model-id "$model" \
    --direction-family encoding_and_source \
    --observational-features \
      outputs/en/05_linguistic/features/paired_feature_differences.csv \
    --components lexical syntactic source \
    --bootstrap 10000 \
    --permutations 20000
done

python scripts/build_acl_result_tables.py
echo "[complete] dual-GPU direction baselines and local result table"
