#!/usr/bin/env bash
# Run three paired decode seeds for the two learned axes of one target model.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${1:?usage: run_seed_robustness_audit.sh MODEL_ID}"
MODEL_ROOT="${MODEL_ROOT:-../../models}"
PROMPTS="${PROMPTS:-data/prompts/en/formal.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_SEEDS="${NUM_SEEDS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

PULLBACK_ROOT="outputs/en/05_linguistic/causal_pullback/${MODEL_ID}/mean_pool"
OUTPUT_SUBDIR="causal_pullback_seed_robustness_generation"
GENERATION_ROOT="outputs/en/03_steering_vectors/${MODEL_ID}/proposition_aligned/mean_pool/${OUTPUT_SUBDIR}"
ANALYSIS_ROOT="${PULLBACK_ROOT}/confirmatory_seed_robustness"

# A second GPU may precompute this independent stage while the primary chain
# is still producing the random-axis ensemble. Serialize duplicate invocations
# so the main bundle can safely resume without two writers sharing work shards.
mkdir -p "$GENERATION_ROOT"
exec 9>"${GENERATION_ROOT}/.audit.lock"
flock 9

python -u scripts/run_steering_generation.py \
  --model "${MODEL_ROOT}/${MODEL_ID}" \
  --language en \
  --representations mean_pool \
  --dataset-tag proposition_aligned \
  --prompts "$PROMPTS" \
  --components lexical syntactic \
  --component-family causal_pullback \
  --component-root "${PULLBACK_ROOT}/frozen_axes" \
  --strengths -1 1 \
  --prompt-start 34 \
  --max-prompts 166 \
  --num-seeds "$NUM_SEEDS" \
  --base-seed 8236 \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --torch-dtype bf16 \
  --attention-backend sdpa \
  --output-subdir "$OUTPUT_SUBDIR"

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  python -u scripts/analyze_seed_robustness.py \
    --input-dir "$GENERATION_ROOT" \
    --out-dir "$ANALYSIS_ROOT" \
    --model-id "$MODEL_ID" \
    --observational-features \
      outputs/en/05_linguistic/features/paired_feature_differences.csv \
    --components lexical syntactic \
    --bootstrap 10000 \
    --permutations 20000
else
  echo "[skip] local linguistic analysis requested"
fi

echo "[complete] ${MODEL_ID} paired decode-seed robustness"
