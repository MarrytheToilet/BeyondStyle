#!/bin/bash
# BeyondStyle: Complete all remaining training tasks
# Waits for GPU to be free, then runs eval grid + interpretability + summary
set -e
cd /root/code/BeyondStyle

echo ""
echo "============================================================"
echo "=== BeyondStyle: Waiting for GPU to be free...           ==="
echo "============================================================"

# Wait until GPU memory usage drops below 2GB (meaning no large models loaded)
while true; do
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$used_mb" ]; then
        echo "[$(date +%H:%M:%S)] nvidia-smi failed, retrying..."
        sleep 30
        continue
    fi
    if [ "$used_mb" -lt 2000 ]; then
        echo "[$(date +%H:%M:%S)] GPU free (${used_mb}MB used). Starting BeyondStyle tasks."
        break
    fi
    echo "[$(date +%H:%M:%S)] GPU busy (${used_mb}MB used). Waiting 60s..."
    sleep 60
done

echo ""
echo "============================================================"
echo "=== Task 1/3: Eval Grid Completion (resume mode)         ==="
echo "============================================================"
echo "Start: $(date)"

python scripts/eval_lm_benchmark.py --mode grid \
    --layers 10 14 16 20 \
    --denoise raw nullspace fisher \
    --resume

echo ""
echo "=== Task 1 Complete: $(date) ==="

echo ""
echo "============================================================"
echo "=== Task 2/3: Interpretability Analysis                  ==="
echo "============================================================"
echo "Start: $(date)"

python scripts/run_interpretability.py \
    --layers 10 14 16 20 \
    --max_samples 100 \
    --device cuda

echo ""
echo "=== Task 2 Complete: $(date) ==="

echo ""
echo "============================================================"
echo "=== Task 3/3: Regenerate Analysis Figures                ==="
echo "============================================================"
echo "Start: $(date)"

python scripts/run_analysis.py

echo ""
echo "=== Task 3 Complete: $(date) ==="

echo ""
echo "============================================================"
echo "=== ALL BEYONDSTYLE TASKS COMPLETE: $(date) ==="
echo "============================================================"
