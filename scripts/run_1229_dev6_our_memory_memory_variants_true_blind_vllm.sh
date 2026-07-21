#!/usr/bin/env bash

# Build the true-blind memory ablation variants:
# current session dialogue + previous memory only, with no API calls in prompts.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data/minseo/experiments5}"
RUN_ID="${RUN_ID:-1229_dev6_memory_variants_true_blind_20260528}"
ONLY_MEMORY_MODES="${ONLY_MEMORY_MODES:-generation_only generation_only_accum blind_refine_1 blind_refine_2 blind_refine_3}"
ONLY_MEMORY_MODELS="${ONLY_MEMORY_MODELS:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B deepseek-ai/DeepSeek-R1-0528-Qwen3-8B google/gemma-3-12b-it}"
TRUE_BLIND=1

export ROOT_DIR RUN_ID ONLY_MEMORY_MODES ONLY_MEMORY_MODELS TRUE_BLIND

exec "$ROOT_DIR/scripts/run_1229_dev6_our_memory_memory_variants_vllm.sh" "$@"
