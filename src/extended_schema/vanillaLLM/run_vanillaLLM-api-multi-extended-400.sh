#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHON_SCRIPT="$RELEASE_ROOT/extended_schema/vanillaLLM/vanillaLLM_inference-api-multi_fixed_pairs.py"

INPUT_PATH="$RELEASE_ROOT/data/1229_dev_6.json"
MULTITURN_QUERY_PATH="$RELEASE_ROOT/query_new_multi.json"
FIXED_PAIRS_PATH="$RELEASE_ROOT/data/fixed_multiturn_example_query_pairs_400.json"
PREF_LIST_PATH="$RELEASE_ROOT/pref_list_extended.json"
PREF_GROUP_PATH="$RELEASE_ROOT/pref_group_extended.json"
TOOLS_SCHEMA_PATH="$RELEASE_ROOT/schema_all_extended_complete.json"

RUN_ROOT="$RELEASE_ROOT/extended_schema/vanillaLLM/output"
BASE_OUTPUT_DIR="$RUN_ROOT/multiturn/api"
BASE_LOG_DIR="$RUN_ROOT/logs/multiturn/api"

DATE_TAG="$(date +%m%d)"
TEST_TAG="${TEST_TAG:-fixed_400_rerun}"
CONCURRENCY="${CONCURRENCY:-10}"
MAX_QUERIES="${MAX_QUERIES:-400}"

MODELS=(
    "gemini-3-flash-preview"
    "gpt-5"
)

PROMPT_TYPES=("imp-zs")
CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("hard")

echo "========================================================"
echo "VanillaLLM Fixed-400 API Multi-turn Started at $(date)"
echo "Fixed Pairs: $FIXED_PAIRS_PATH"
echo "========================================================"

mkdir -p "$BASE_OUTPUT_DIR" "$BASE_LOG_DIR"

for model in "${MODELS[@]}"; do
    MODEL_SAFE_NAME="${model//\//_}"

    EFFORT_LEVELS=("default")
    if [[ "$model" == *"gpt-5"* ]] || [[ "$model" == *"gemini-3"* ]] || [[ "$model" == *"o1"* ]] || [[ "$model" == *"o3"* ]]; then
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
                    CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/multiturn-query/$MODEL_SAFE_NAME-$effort/$prompt_type"
                    CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/multiturn-query/$MODEL_SAFE_NAME-$effort/$prompt_type"
                    mkdir -p "$CURRENT_OUT_DIR" "$CURRENT_LOG_DIR"

                    OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_${TEST_TAG}.json"
                    LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_${TEST_TAG}.jsonl"

                    CMD=(
                        python "$PYTHON_SCRIPT"
                        --input_path "$INPUT_PATH"
                        --multiturn_path "$MULTITURN_QUERY_PATH"
                        --fixed_pairs_path "$FIXED_PAIRS_PATH"
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
echo "VanillaLLM Fixed-400 API Multi-turn Finished at $(date)"
echo "========================================================"
