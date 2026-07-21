#!/bin/bash

set -euo pipefail

PYTHON_SCRIPT="/data/minseo/experiments4/extended_schema/ours_memory/Preference_Memory_step2_ACTION_singleturn_api_fixed_pairs.py"

INPUT_PATH="/data/minseo/experiments4/data/1229_dev_6.json"
QUERY_PATH="/data/minseo/experiments4/query_new_single.json"
FIXED_PAIRS_PATH="/data/minseo/experiments4/data/fixed_singleturn_example_query_pairs_400.json"
PREF_LIST_PATH="/data/minseo/experiments4/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments4/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments4/schema_easy_extended.json"

RUN_ROOT="/data/minseo/experiments4/extended_schema/ours_memory/output"
DATE_TAG="$(date +%m%d)"
TEST_TAG="${TEST_TAG:-fixed_400_rerun}"
CONCURRENCY="${CONCURRENCY:-10}"
MAX_QUERIES="${MAX_QUERIES:-400}"

MODELS=(
    #"gemini-3-flash-preview"
    "gpt-5"
)

MEMORY_FOLDERS=(
    #"google_gemma-3-12b-it"
    "gpt-4o-mini"
)

STOP_AFTER_MEMORY_FOLDER="${STOP_AFTER_MEMORY_FOLDER:-gpt-4o-mini}"
STOP_RUN_AFTER_TARGET_MEMORY="${STOP_RUN_AFTER_TARGET_MEMORY:-1}"

CONTEXT_TYPES=("memory_api")
PREF_TYPES=("hard")
PROMPT_NAME="implicit_zs"

echo "####################################################################"
echo "Starting Fixed-400 Inference using OpenAI/Gemini API (Single-turn Hard)"
echo "Fixed Pairs : $FIXED_PAIRS_PATH"
echo "####################################################################"

STOP_REQUESTED=0

for model in "${MODELS[@]}"; do
    MODEL_SAFE_NAME="${model//\//_}"
    echo ">> [Model] Current Inference Model: $model"

    if [[ "$model" == *"gemini"* ]] && [[ -z "${GOOGLE_API_KEY:-}" ]]; then
        echo "[ERROR] GOOGLE_API_KEY is not set. Gemini inference cannot start."
        exit 1
    fi

    for MEM_FOLDER in "${MEMORY_FOLDERS[@]}"; do
        MEMORY_PATH="/data/minseo/experiments4/ours_memory/inference/1231_MEMORY3/${MEM_FOLDER}/_memory1.jsonl"
        BASE_OUTPUT_DIR="$RUN_ROOT/singleturn/api/${MEM_FOLDER}"
        BASE_LOG_DIR="$RUN_ROOT/logs/singleturn/api/${MEM_FOLDER}"

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
                    --query_path "$QUERY_PATH" \
                    --fixed_pairs_path "$FIXED_PAIRS_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --model_name "$model" \
                    --concurrency "$CONCURRENCY" \
                    --reasoning_effort "high" \
                    --max_queries "$MAX_QUERIES"
            done
        done

        if [ "${STOP_RUN_AFTER_TARGET_MEMORY}" = "1" ] && [ "$MEM_FOLDER" = "$STOP_AFTER_MEMORY_FOLDER" ]; then
            echo ">> Stop requested after memory source: $MEM_FOLDER"
            STOP_REQUESTED=1
            break
        fi
    done

    if [ "${STOP_REQUESTED}" = "1" ]; then
        echo ">> Exiting after requested memory boundary: ${STOP_AFTER_MEMORY_FOLDER}"
        break
    fi

    echo "--------------------------------------------------------"
done

echo "========================================================"
echo "All Fixed-400 API Jobs Finished Successfully"
echo "========================================================"
