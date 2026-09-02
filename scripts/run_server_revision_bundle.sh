#!/usr/bin/env bash
# Run the bounded revision experiment set on two GPUs. GPU 0 continues from
# Llama to the third-family Gemma replication while GPU 1 handles Qwen.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
LOG_DIR="${LOG_DIR:-outputs/en/05_linguistic/revision_bundle_logs}"
# The comparator run may already contain resumable batch-4 shards. Keep that
# protocol frozen while using the much larger 5090 memory headroom for new
# decoding and Gemma runs. Generation automatically bisects an individual
# batch on CUDA OOM, so these are aggressive but safe requested sizes.
COMPARATOR_BATCH_SIZE="${COMPARATOR_BATCH_SIZE:-4}"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-8}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-16}"
GEMMA_BATCH_SIZE="${GEMMA_BATCH_SIZE:-12}"
mkdir -p "$LOG_DIR"

python -u scripts/record_server_snapshot.py

run_primary_chain() {
  CUDA_VISIBLE_DEVICES=0 \
    MODEL_ROOT="$MODEL_ROOT" \
    BATCH_SIZE="$COMPARATOR_BATCH_SIZE" \
    RUN_ANALYSIS=0 \
    bash scripts/run_direction_baseline_audit.sh \
      Llama-3.1-8B-Instruct

  CUDA_VISIBLE_DEVICES=0 \
    MODEL_ROOT="$MODEL_ROOT" \
    BATCH_SIZE="$LLAMA_BATCH_SIZE" \
    RUN_ANALYSIS=0 \
    bash scripts/run_decoding_regime_audit.sh \
      Llama-3.1-8B-Instruct

  CUDA_VISIBLE_DEVICES=0 python -u scripts/smoke_layer_alignment.py \
    --model "${MODEL_ROOT}/gemma-4-E2B-it" \
    --layers 8 35 \
    --dtype bfloat16
  CUDA_VISIBLE_DEVICES=0 python -u scripts/smoke_steering_runtime.py \
    --model "${MODEL_ROOT}/gemma-4-E2B-it" \
    --layer 8 \
    --batch-size 2 \
    --max-new-tokens 4 \
    --torch-dtype bf16 \
    --attention-backend sdpa \
    --json-out "${LOG_DIR}/gemma_runtime_smoke.json"

  CUDA_VISIBLE_DEVICES=0 \
    MODEL_ROOT="$MODEL_ROOT" \
    BATCH_SIZE="$GEMMA_BATCH_SIZE" \
    bash scripts/run_gemma_causal_replication.sh
}

run_secondary_chain() {
  CUDA_VISIBLE_DEVICES=1 \
    MODEL_ROOT="$MODEL_ROOT" \
    BATCH_SIZE="$COMPARATOR_BATCH_SIZE" \
    RUN_ANALYSIS=0 \
    bash scripts/run_direction_baseline_audit.sh Qwen3-4B

  CUDA_VISIBLE_DEVICES=1 \
    MODEL_ROOT="$MODEL_ROOT" \
    BATCH_SIZE="$QWEN_BATCH_SIZE" \
    RUN_ANALYSIS=0 \
    bash scripts/run_decoding_regime_audit.sh Qwen3-4B
}

run_primary_chain >"${LOG_DIR}/gpu0_llama_gemma.log" 2>&1 &
primary_pid=$!
run_secondary_chain >"${LOG_DIR}/gpu1_qwen.log" 2>&1 &
secondary_pid=$!

wait "$primary_pid"
wait "$secondary_pid"

analysis_pids=()
for model in Llama-3.1-8B-Instruct Qwen3-4B; do
  python -u scripts/analyze_seed_robustness.py \
    --input-dir \
      "outputs/en/03_steering_vectors/${model}/proposition_aligned/mean_pool/encoding_source_seed_robustness_generation" \
    --out-dir \
      "outputs/en/05_linguistic/causal_pullback/${model}/mean_pool/comparator_seed_robustness" \
    --model-id "$model" \
    --direction-family encoding_and_source \
    --observational-features \
      outputs/en/05_linguistic/features/paired_feature_differences.csv \
    --components lexical syntactic source \
    --n-process 4 \
    --bootstrap 10000 \
    --permutations 20000 &
  analysis_pids+=("$!")

  for label in t0 t1; do
    python -u scripts/analyze_seed_robustness.py \
      --input-dir \
        "outputs/en/03_steering_vectors/${model}/proposition_aligned/mean_pool/causal_pullback_decode_${label}_generation" \
      --out-dir \
        "outputs/en/05_linguistic/causal_pullback/${model}/mean_pool/decode_${label}_robustness" \
      --model-id "$model" \
      --direction-family "canonical_pullback_${label}" \
      --observational-features \
      outputs/en/05_linguistic/features/paired_feature_differences.csv \
      --components lexical syntactic \
      --n-process 4 \
      --bootstrap 10000 \
      --permutations 20000 &
    analysis_pids+=("$!")
  done
done

for analysis_pid in "${analysis_pids[@]}"; do
  wait "$analysis_pid"
done

python -u scripts/summarize_decoding_regimes.py
python -u scripts/summarize_compute_cost.py
python scripts/build_acl_result_tables.py
python -u scripts/record_server_snapshot.py

echo "[complete] revision bundle"
