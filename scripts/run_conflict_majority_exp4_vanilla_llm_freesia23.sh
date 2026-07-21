#!/usr/bin/env bash

# Run conflict-majority BASE PROMPTING through the experiment-4 vanillaLLM
# prompt/call path while freesia serves vLLM on GPUs 2,3.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data/minseo/experiments5}"
RUN_ID="${RUN_ID:-conflict_majority_exp4_vanilla_llm_freesia23_$(date +%Y%m%d_%H%M%S)}"

export ROOT_DIR
export RUN_ID
export METHOD_DIR="${METHOD_DIR:-vanilla_llm_exp4}"
export METHOD_NAME="${METHOD_NAME:-vanilla_llm_exp4}"
export INFER_SCRIPT="${INFER_SCRIPT:-$ROOT_DIR/scripts/run_conflict_majority_exp4_vanilla_llm_inference.py}"
export EVAL_SCRIPT="${EVAL_SCRIPT:-$ROOT_DIR/scripts/build_conflict_majority_exp4_vanilla_llm_results.py}"

exec bash "$ROOT_DIR/scripts/run_conflict_majority_vanilla_llm_freesia23.sh"
