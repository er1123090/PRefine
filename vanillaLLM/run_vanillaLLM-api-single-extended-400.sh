#!/bin/bash

set -euo pipefail

# ==============================================================================
# 1. Environment & Paths
# ==============================================================================
PYTHON_SCRIPT="/data/minseo/experiments4/vanillaLLM/vanillaLLM_inference-api-single.py"

INPUT_PATH="/data/minseo/experiments4/data/1229_dev_6.json"
QUERY_PATH="/data/minseo/experiments4/query_new_single.json"
PREF_LIST_PATH="/data/minseo/experiments4/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments4/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments4/schema_easy_extended.json"

BASE_OUTPUT_DIR="/data/minseo/experiments4/vanillaLLM/inference/extended/outputs/singleturn/api"
BASE_LOG_DIR="/data/minseo/experiments4/vanillaLLM/inference/extended/logs/singleturn/api"

DATE_TAG="$(date +%m%d)"
TEST_TAG="${TEST_TAG:-extended_400}"
CONCURRENCY="${CONCURRENCY:-10}"
MAX_QUERIES="${MAX_QUERIES:-400}"

# ==============================================================================
# 2. Experiment Variables
# ==============================================================================
MODELS=(
    "gemini-3-flash-preview"
    "gpt-5"
)

PROMPT_TYPES=("imp-zs")
CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("hard")

# ==============================================================================
# 3. Batch Execution
# ==============================================================================
echo "========================================================"
echo "VanillaLLM Extended API Single-turn Started at $(date)"
echo "Query File : $QUERY_PATH"
echo "Max Queries: $MAX_QUERIES"
echo "========================================================"

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_LOG_DIR"

for model in "${MODELS[@]}"; do
    MODEL_SAFE_NAME="${model//\//_}"

    EFFORT_LEVELS=("default")
    if [[ "$model" == *"gpt-5"* ]] || \
       [[ "$model" == *"gemini-3"* ]] || \
       [[ "$model" == *"o1"* ]] || \
       [[ "$model" == *"o3"* ]] || \
       [[ "$model" == *"claude"* ]]; then
        EFFORT_LEVELS=("minimal")
    fi

    if [[ "$model" == *"gemini"* ]] && [[ -z "${GOOGLE_API_KEY:-}" ]]; then
        echo "[ERROR] GOOGLE_API_KEY is not set for model: $model"
        exit 1
    fi

    for prompt_type in "${PROMPT_TYPES[@]}"; do
        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do
                for effort in "${EFFORT_LEVELS[@]}"; do
                    echo ""
                    echo "--------------------------------------------------------------------------------"
                    echo "[RUNNING] Model: $model | Prompt: $prompt_type | Pref: $pref | Effort: $effort"
                    echo "--------------------------------------------------------------------------------"

                    CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/singleturn-query/$MODEL_SAFE_NAME-$effort/$prompt_type"
                    CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/singleturn-query/$MODEL_SAFE_NAME-$effort/$prompt_type"
                    mkdir -p "$CURRENT_OUT_DIR"
                    mkdir -p "$CURRENT_LOG_DIR"

                    OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_${TEST_TAG}.json"
                    LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_${TEST_TAG}.jsonl"

                    CMD=(
                        python "$PYTHON_SCRIPT"
                        --input_path "$INPUT_PATH"
                        --query_path "$QUERY_PATH"
                        --pref_list_path "$PREF_LIST_PATH"
                        --pref_group_path "$PREF_GROUP_PATH"
                        --tools_schema_path "$TOOLS_SCHEMA_PATH"
                        --context_type "$context"
                        --pref_type "$pref"
                        --prompt_type "$prompt_type"
                        --model_name "$model"
                        --output_path "$OUTPUT_FILE"
                        --log_path "$LOG_FILE"
                        --concurrency "$CONCURRENCY"
                        --max_queries "$MAX_QUERIES"
                    )

                    if [ "$effort" != "default" ]; then
                        CMD+=(--reasoning_effort "$effort")
                    fi

                    "${CMD[@]}"
                done
            done
        done
    done
done

echo "========================================================"
echo "VanillaLLM Extended API Single-turn Finished at $(date)"
echo "========================================================"
