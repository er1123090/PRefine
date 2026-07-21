#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python run_petool_memory_gvr.py \
  --generator-cache results/qwen25_7b_instruct_vllm_full \
  --run-name petool_memory_gvr_qwen25_full_provider \
  --refine-policy p_r_memory_c_generator

