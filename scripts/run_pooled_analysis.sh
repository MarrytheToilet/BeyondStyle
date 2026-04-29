#!/usr/bin/env bash
# Re-run the non-generation, non-eval analysis pipelines using mean-pooled
# hidden states (instead of the default last-token pooling) and write
# everything to a sibling output root: outputs_pool/.
#
# Layout produced (mirrors outputs/):
#   outputs_pool/{lang}/02_separability/<model>/<dataset>/
#   outputs_pool/{lang}/03_steering_vectors/<model>/<dataset>/   (states + v_src models, no generation)
#   outputs_pool/{lang}/06_interpretability/<model>/<dataset>/
#
# Usage:
#   bash scripts/run_pooled_analysis.sh en   # English  (Llama-3.1-8B-Instruct + paired_mixed_500)
#   bash scripts/run_pooled_analysis.sh zh   # Chinese  (Qwen3-4B + paired_mixed_500)

set -euo pipefail

LANG_ARG="${1:-en}"
POOLING="${POOLING:-full_mean}"
export BEYONDSTYLE_OUTPUT_DIR="outputs_pool"

case "$LANG_ARG" in
  en)
    MODEL="${MODEL_EN:-../../models/Llama-3.1-8B-Instruct}"
    DATASET="data/paired/en/paired_mixed_500.json"
    ;;
  zh)
    MODEL="${MODEL_ZH:-../../models/Qwen3-4B}"
    DATASET="data/paired/zh/paired_mixed_500.json"
    ;;
  *)
    echo "unknown lang: $LANG_ARG (use en|zh)" ; exit 1 ;;
esac

echo "=== POOLED ANALYSIS  (lang=$LANG_ARG  model=$MODEL  pool=$POOLING) ==="

# 1) extraction + v_src training + per-layer metrics  (NO generation)
python scripts/steering_pipeline.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --multi \
    --denoise raw nullspace fisher \
    --pooling "$POOLING" \
    --skip_generation \
    --language "$LANG_ARG"

# 2) held-out separability over the same cached states
MODEL_ID=$(basename "$MODEL")
DS_TAG=$(basename "$DATASET" .json | sed 's/paired_//')
STEER_DIR="outputs_pool/$LANG_ARG/03_steering_vectors/$MODEL_ID/$DS_TAG"
SEP_DIR="outputs_pool/$LANG_ARG/02_separability/$MODEL_ID/$DS_TAG"
mkdir -p "$SEP_DIR"

python scripts/eval_heldout_separability.py \
    --states_glob "$STEER_DIR/states/paired_states_layer*.pt" \
    --out "$SEP_DIR/heldout_auc.json" \
    --test_frac 0.2 --n_runs 5

# 3) re-render separability figures from new JSON
python scripts/plot_separability_figs.py --in-dir "$SEP_DIR"

# 4) interpretability over the new v_src models
python scripts/run_interpretability.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --language "$LANG_ARG"

echo "=== DONE.  See: outputs_pool/$LANG_ARG/ ==="
