#!/usr/bin/env bash
# Use one GPU per target model for the bounded decoding-regime audit.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
LOG_DIR="${LOG_DIR:-outputs/en/05_linguistic/causal_pullback/decoding_regime_logs}"
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
    bash scripts/run_decoding_regime_audit.sh "$model" \
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
  for label in t0 t1; do
    generation_root="outputs/en/03_steering_vectors/${model}/proposition_aligned/mean_pool/causal_pullback_decode_${label}_generation"
    analysis_root="outputs/en/05_linguistic/causal_pullback/${model}/mean_pool/decode_${label}_robustness"
    python -u scripts/analyze_seed_robustness.py \
      --input-dir "$generation_root" \
      --out-dir "$analysis_root" \
      --model-id "$model" \
      --direction-family "canonical_pullback_${label}" \
      --observational-features \
        outputs/en/05_linguistic/features/paired_feature_differences.csv \
      --components lexical syntactic \
      --bootstrap 10000 \
      --permutations 20000
  done
done

python -u scripts/summarize_decoding_regimes.py
python scripts/build_acl_result_tables.py
echo "[complete] dual-GPU decoding-regime audit"
