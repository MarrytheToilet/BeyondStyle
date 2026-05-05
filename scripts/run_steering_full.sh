#!/bin/bash
# Steering generation (top-3 layers auto-picked from cached metrics) + lm-eval.
# Pre-req: extract+train already done (states/ and models/ on disk).
set -e
cd "$(dirname "$0")/.."

EN_MODEL="${EN_MODEL:-${LOCAL_MODEL_PATH:-../../models/Llama-3.1-8B-Instruct}}"
ZH_MODEL="${ZH_MODEL:-../../models/Qwen3-4B}"
EN_DATA="${EN_DATA:-data/paired/en/paired_mixed_500.json}"
ZH_DATA="${ZH_DATA:-data/paired/zh/paired_mixed_500.json}"
EN_PROMPTS="${EN_PROMPTS:-data/prompts/en/wikipedia_200.json}"
ZH_PROMPTS="${ZH_PROMPTS:-data/prompts/zh/chinese_news_alt_200.json}"
ALPHAS="-2.0 -1.0 0.0 1.0 2.0"

run() { echo; echo "═══ $* ═══"; date; eval "$@"; }

# ─── EN steering generation (top-3 layers from cached metrics) ───
run python scripts/steering_pipeline.py --multi --language en \
  --model $EN_MODEL --dataset $EN_DATA \
  --denoise raw nullspace fisher --alphas $ALPHAS \
  --test_prompts_file $EN_PROMPTS --skip_extraction

# ─── ZH steering generation ──────────────────────────────────────
run python scripts/steering_pipeline.py --multi --language zh \
  --model $ZH_MODEL --dataset $ZH_DATA \
  --denoise raw nullspace fisher --alphas $ALPHAS \
  --test_prompts_file $ZH_PROMPTS --skip_extraction

# ─── lm-eval on Llama-3.1-8B (top-3 layers × {raw,nullspace,fisher} × alphas) ───
# Top-3 EN layers by hull_centroid_dist_norm: 15, 20, 11
run python scripts/eval_lm_benchmark.py --mode grid \
  --model $EN_MODEL \
  --layers 15 20 11 \
  --alpha -2.0 -1.0 0.0 1.0 2.0 \
  --denoise raw nullspace fisher \
  --tasks wikitext hellaswag arc_easy \
  --batch-size 8 --resume

echo; echo "═══ ALL DONE ═══"; date
