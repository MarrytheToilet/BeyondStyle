#!/usr/bin/env bash
# Evaluate the frozen pullback axes under greedy and higher-temperature
# decoding. The published temperature-0.7, three-seed run remains untouched.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${1:?usage: run_decoding_regime_audit.sh MODEL_ID}"
MODEL_ROOT="${MODEL_ROOT:-../../models}"
PROMPTS="${PROMPTS:-data/prompts/en/formal.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

case "$MODEL_ID" in
  Llama-3.1-8B-Instruct)
    LAYER=7
    ;;
  Qwen3-4B|gemma-4-E2B-it)
    LAYER=8
    ;;
  *)
    echo "unsupported model: $MODEL_ID" >&2
    exit 2
    ;;
esac

PULLBACK_ROOT="outputs/en/05_linguistic/causal_pullback/${MODEL_ID}/mean_pool"
FROZEN_ROOT="${PULLBACK_ROOT}/frozen_axes"
if [[ ! -f "${FROZEN_ROOT}/layer$(printf '%02d' "$LAYER")/directions/lexical.pt" ]]; then
  echo "missing frozen lexical axis under ${FROZEN_ROOT}" >&2
  exit 1
fi

run_regime() {
  local label="$1"
  local temperature="$2"
  local top_p="$3"
  local seeds="$4"
  local output_subdir="causal_pullback_decode_${label}_generation"
  local generation_root="outputs/en/03_steering_vectors/${MODEL_ID}/proposition_aligned/mean_pool/${output_subdir}"
  local analysis_root="${PULLBACK_ROOT}/decode_${label}_robustness"

  python -u scripts/run_steering_generation.py \
    --model "${MODEL_ROOT}/${MODEL_ID}" \
    --language en \
    --representations mean_pool \
    --dataset-tag proposition_aligned \
    --prompts "$PROMPTS" \
    --components lexical syntactic \
    --component-family causal_pullback \
    --component-root "$FROZEN_ROOT" \
    --layer "$LAYER" \
    --strengths -1 1 \
    --prompt-start 34 \
    --max-prompts 166 \
    --num-seeds "$seeds" \
    --base-seed 8236 \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$temperature" \
    --top-p "$top_p" \
    --torch-dtype bf16 \
    --attention-backend sdpa \
    --output-subdir "$output_subdir"

  if [[ "$RUN_ANALYSIS" == "1" ]]; then
    python -u scripts/analyze_seed_robustness.py \
      --input-dir "$generation_root" \
      --out-dir "$analysis_root" \
      --model-id "$MODEL_ID" \
      --direction-family "canonical_pullback_${label}" \
      --observational-features \
        outputs/en/05_linguistic/features/paired_feature_differences.csv \
      --components lexical syntactic \
      --bootstrap 10000 \
      --permutations 20000
  fi
}

run_regime t0 0.0 1.0 1
run_regime t1 1.0 0.95 3

echo "[complete] ${MODEL_ID} decoding-regime audit"
