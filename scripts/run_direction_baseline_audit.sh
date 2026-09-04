#!/usr/bin/env bash
# Evaluate encoding-pattern and nullspace source-difference directions under
# exactly the same paired 3-seed free-generation protocol as the pullbacks.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${1:?usage: run_direction_baseline_audit.sh MODEL_ID}"
MODEL_ROOT="${MODEL_ROOT:-../../models}"
PROMPTS="${PROMPTS:-data/prompts/en/formal.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_SEEDS="${NUM_SEEDS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
ENDPOINT_STRENGTH="${ENDPOINT_STRENGTH:-1}"

if ! python - "$ENDPOINT_STRENGTH" <<'PY'
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if value > 0 else 1)
PY
then
  echo "ENDPOINT_STRENGTH must be a positive number" >&2
  exit 2
fi

case "$MODEL_ID" in
  Llama-3.1-8B-Instruct)
    LAYER=7
    ;;
  Qwen3-4B)
    LAYER=8
    ;;
  *)
    echo "Unsupported frozen comparator model: $MODEL_ID" >&2
    exit 2
    ;;
esac

PATTERN_ROOT="outputs/en/05_linguistic/concept_patterns_layerwise/${MODEL_ID}/mean_pool"
strength_tag="${ENDPOINT_STRENGTH//./p}"
if [[ "$ENDPOINT_STRENGTH" == "1" || "$ENDPOINT_STRENGTH" == "1.0" ]]; then
  default_output_subdir="encoding_source_seed_robustness_generation"
  default_analysis_subdir="comparator_seed_robustness"
else
  default_output_subdir="encoding_source_strength${strength_tag}_seed_robustness_generation"
  default_analysis_subdir="comparator_strength${strength_tag}_seed_robustness"
fi
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-$default_output_subdir}"
ANALYSIS_SUBDIR="${ANALYSIS_SUBDIR:-$default_analysis_subdir}"
GENERATION_ROOT="outputs/en/03_steering_vectors/${MODEL_ID}/proposition_aligned/mean_pool/${OUTPUT_SUBDIR}"
ANALYSIS_ROOT="outputs/en/05_linguistic/causal_pullback/${MODEL_ID}/mean_pool/${ANALYSIS_SUBDIR}"

python -u scripts/run_steering_generation.py \
  --model "${MODEL_ROOT}/${MODEL_ID}" \
  --language en \
  --representations mean_pool \
  --dataset-tag proposition_aligned \
  --prompts "$PROMPTS" \
  --components lexical syntactic source \
  --component-family concept_patterns_layerwise \
  --component-root "$PATTERN_ROOT" \
  --layer "$LAYER" \
  --strengths "-${ENDPOINT_STRENGTH}" "$ENDPOINT_STRENGTH" \
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
    --direction-family encoding_and_source \
    --observational-features \
      outputs/en/05_linguistic/features/paired_feature_differences.csv \
    --components lexical syntactic source \
    --endpoint-strength "$ENDPOINT_STRENGTH" \
    --bootstrap 10000 \
    --permutations 20000
else
  echo "[skip] local linguistic analysis requested"
fi

echo "[complete] ${MODEL_ID} encoding/source direction audit"
