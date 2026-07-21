#!/bin/bash

set -euo pipefail

PYTHON_SCRIPT="/data/minseo/experiments4/extended_schema/mem0/mem0_evaluate_multiturn_fixed_pairs.py"

INPUT_PATH="/data/minseo/experiments4/data/1229_dev_6.json"
MULTITURN_PATH="/data/minseo/experiments4/query_new_multi.json"
FIXED_PAIRS_PATH="/data/minseo/experiments4/data/fixed_multiturn_example_query_pairs_400.json"
PREF_LIST_PATH="/data/minseo/experiments4/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments4/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments4/schema_all_extended_complete.json"

RUN_ROOT="/data/minseo/experiments4/extended_schema/mem0/output"
DATE_TAG="$(date +%m%d)"
TEST_TAG="${TEST_TAG:-fixed_400_rerun}"
CONCURRENCY="${CONCURRENCY:-10}"
MAX_QUERIES="${MAX_QUERIES:-400}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
BASE_URL="${BASE_URL:-}"
API_KEY="${API_KEY:-EMPTY}"

MODELS=(
    "gemini-3-flash-preview"
    "gpt-5"
)

CONTEXT_TYPES=("memory_api")
PREF_TYPES=("hard")
PROMPT_NAME="implicit_zs"

if [ -z "${MEM0_API_KEY:-}" ]; then
    echo "[ERROR] MEM0_API_KEY is not set."
    exit 1
fi

if [ -n "${TARGET_MODELS:-}" ]; then
    IFS='|' read -r -a MODELS <<< "${TARGET_MODELS}"
fi

echo "========================================================"
echo "Mem0 Fixed-400 Multi-turn Started at $(date)"
echo "Fixed Pairs: $FIXED_PAIRS_PATH"
echo "========================================================"

for model in "${MODELS[@]}"; do
    MODEL_SAFE_NAME="${model//\//_}"
    EFFORT_TAG="${REASONING_EFFORT:-default}"
    BACKEND_TAG="api"
    if [ -n "$BASE_URL" ]; then
        BACKEND_TAG="vllm"
    fi

    if [ -z "$BASE_URL" ] && [[ "$model" == gemini* ]] && [[ -z "${GOOGLE_API_KEY:-}" ]]; then
        echo "[ERROR] GOOGLE_API_KEY is not set for model: $model"
        exit 1
    fi

    if [ -z "$BASE_URL" ] && [[ "$model" == gpt-* || "$model" == o1* || "$model" == o3* || "$model" == o4* ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "[ERROR] OPENAI_API_KEY is not set for model: $model"
        exit 1
    fi

    BASE_OUTPUT_DIR="$RUN_ROOT/multiturn/$BACKEND_TAG"
    BASE_LOG_DIR="$RUN_ROOT/logs/multiturn/$BACKEND_TAG"
    mkdir -p "$BASE_OUTPUT_DIR" "$BASE_LOG_DIR"

    for context in "${CONTEXT_TYPES[@]}"; do
        for pref in "${PREF_TYPES[@]}"; do
            CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/${MODEL_SAFE_NAME}_${EFFORT_TAG}/$PROMPT_NAME"
            CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/${MODEL_SAFE_NAME}_${EFFORT_TAG}/$PROMPT_NAME"
            mkdir -p "$CURRENT_OUT_DIR" "$CURRENT_LOG_DIR"

            OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_${TEST_TAG}.json"
            LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_${TEST_TAG}.jsonl"

            CMD=(
                python "$PYTHON_SCRIPT"
                --input_path "$INPUT_PATH"
                --output_path "$OUTPUT_FILE"
                --log_path "$LOG_FILE"
                --multiturn_path "$MULTITURN_PATH"
                --fixed_pairs_path "$FIXED_PAIRS_PATH"
                --pref_list_path "$PREF_LIST_PATH"
                --pref_group_path "$PREF_GROUP_PATH"
                --tools_schema_path "$TOOLS_SCHEMA_PATH"
                --pref_type "$pref"
                --context_type "$context"
                --model_name "$model"
                --concurrency "$CONCURRENCY"
                --max_queries "$MAX_QUERIES"
            )

            if [ -n "$REASONING_EFFORT" ]; then
                CMD+=(--reasoning_effort "$REASONING_EFFORT")
            fi
            if [ -n "$BASE_URL" ]; then
                CMD+=(--base_url "$BASE_URL" --api_key "$API_KEY")
            fi

            "${CMD[@]}"
        done
    done
done

echo "========================================================"
echo "Mem0 Fixed-400 Multi-turn Finished at $(date)"
echo "========================================================"
