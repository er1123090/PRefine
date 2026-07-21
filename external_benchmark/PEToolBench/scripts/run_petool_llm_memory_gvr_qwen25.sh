#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python run_petool_llm_memory_gvr.py \
  --run-name petool_llm_memory_gvr_qwen25_full \
  --batch-size "${BATCH_SIZE:-64}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.82}" \
  --max-input-tokens "${MAX_INPUT_TOKENS:-5000}" \
  --memory-override-history-types p r \
  --inference-strength strong \
  --memory-content-style full \
  --prompt-layout custom \
  --post-refine-policy p_r_memory_c_latest_namespace
