#!/bin/bash

set -euo pipefail

PYTHON_SCRIPT="/data/minseo/experiments5/extended_schema/ours_memory/inference_api_multiturn_fixed_pairs.py"

INPUT_PATH="/data/minseo/experiments5/data/1229_dev_6.json"
MULTITURN_PATH="/data/minseo/experiments5/config/query_new_multi.json"
FIXED_PAIRS_PATH="/data/minseo/experiments5/data/fixed_multiturn_example_query_pairs_400.json"
PREF_LIST_PATH="/data/minseo/experiments5/config/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments5/config/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments5/config/schema_all_extended_complete.json"

RUN_ROOT="/data/minseo/experiments5/extended_schema/ours_memory/output"
MEMORY_ROOT="${MEMORY_ROOT:-/data/minseo/experiments5/outputs/our_memory/memories}"
DATE_TAG="$(date +%m%d)"
TEST_TAG="${TEST_TAG:-fixed_400_rerun}"
CONCURRENCY="${CONCURRENCY:-10}"
MAX_QUERIES="${MAX_QUERIES:-400}"

MODELS=(
    "gemini-3-flash-preview"
    "gpt-5"
)

MEMORY_FOLDERS=(
    #"google_gemma-3-12b-it"
    "gpt-4o-mini"
)

CONTEXT_TYPES=("memory_api")
PREF_TYPES=("hard")
PROMPT_NAME="implicit_zs"

echo "========================================================"
echo "Starting Cloud API Inference (Multi-turn Fixed-400 Hard)"
echo "Fixed Pairs : $FIXED_PAIRS_PATH"
echo "========================================================"

for model in "${MODELS[@]}"; do
    MODEL_SAFE_NAME="${model//\//_}"

    echo "####################################################################"
    echo "[Processing Model]: $model"
    echo "####################################################################"

    for MEM_FOLDER in "${MEMORY_FOLDERS[@]}"; do
        MEMORY_PATH="$MEMORY_ROOT/${MEM_FOLDER}/memory_result.jsonl"
        BASE_OUTPUT_DIR="$RUN_ROOT/multiturn/api/${MEM_FOLDER}"
        BASE_LOG_DIR="$RUN_ROOT/logs/multiturn/api/${MEM_FOLDER}"

        mkdir -p "$BASE_OUTPUT_DIR" "$BASE_LOG_DIR"

        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do
                CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/${MODEL_SAFE_NAME}_high/$PROMPT_NAME"
                CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/${MODEL_SAFE_NAME}_high/$PROMPT_NAME"
                mkdir -p "$CURRENT_OUT_DIR" "$CURRENT_LOG_DIR"

                OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_${TEST_TAG}.json"
                LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_${TEST_TAG}.log"

                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --memory_path "$MEMORY_PATH" \
                    --output_path "$OUTPUT_FILE" \
                    --log_path "$LOG_FILE" \
                    --multiturn_path "$MULTITURN_PATH" \
                    --fixed_pairs_path "$FIXED_PAIRS_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --model_name "$model" \
                    --base_url "" \
                    --api_key "ENV" \
                    --concurrency "$CONCURRENCY" \
                    --max_queries "$MAX_QUERIES" \
                    --reasoning_effort "high"
            done
        done

        echo "--------------------------------------------------------"
        sleep 2
    done
done

echo "========================================================"
echo "All Cloud API Jobs Finished at $(date)"
echo "========================================================"
