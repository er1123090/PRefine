#!/bin/bash

set -euo pipefail

PYTHON_SCRIPT="/data/minseo/experiments4/extended_schema/RAG/RAG_inference_multi_fixed_pairs.py"

INPUT_PATH="/data/minseo/experiments4/data/1229_dev_6.json"
MULTITURN_PATH="/data/minseo/experiments4/query_new_multi.json"
FIXED_PAIRS_PATH="/data/minseo/experiments4/data/fixed_multiturn_example_query_pairs_400.json"
PREF_LIST_PATH="/data/minseo/experiments4/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments4/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments4/schema_all_extended_complete.json"
DB_PATH="${DB_PATH:-/data/minseo/experiments4/RAG/chroma_db_rag}"
COLLECTION_NAME="${COLLECTION_NAME:-user_memories}"

RUN_ROOT="/data/minseo/experiments4/extended_schema/RAG/output"
DATE_TAG="$(date +%m%d)"
TEST_TAG="${TEST_TAG:-fixed_400_rerun}"
CONCURRENCY="${CONCURRENCY:-10}"
MAX_QUERIES="${MAX_QUERIES:-400}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
RAG_TOP_K="${RAG_TOP_K:-5}"
BASE_URL="${BASE_URL:-}"
API_KEY="${API_KEY:-EMPTY}"

MODELS=(
    "gemini-3-flash-preview"
    "gpt-5"
)

CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("hard")
PROMPT_NAME="implicit_zs"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] OPENAI_API_KEY is not set. RAG retrieval needs it for embeddings."
    exit 1
fi

if [ -n "${TARGET_MODELS:-}" ]; then
    IFS='|' read -r -a MODELS <<< "${TARGET_MODELS}"
fi

echo "========================================================"
echo "RAG Fixed-400 Multi-turn Started at $(date)"
echo "Fixed Pairs: $FIXED_PAIRS_PATH"
echo "DB Path: $DB_PATH"
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

    BASE_OUTPUT_DIR="$RUN_ROOT/multiturn/$BACKEND_TAG"
    BASE_LOG_DIR="$RUN_ROOT/logs/multiturn/$BACKEND_TAG"
    mkdir -p "$BASE_OUTPUT_DIR" "$BASE_LOG_DIR"

    for context in "${CONTEXT_TYPES[@]}"; do
        for pref in "${PREF_TYPES[@]}"; do
            CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/multiturn-query/${MODEL_SAFE_NAME}_${EFFORT_TAG}/$PROMPT_NAME"
            CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/multiturn-query/${MODEL_SAFE_NAME}_${EFFORT_TAG}/$PROMPT_NAME"
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
                --context_type "$context"
                --pref_type "$pref"
                --prompt_type "imp-zs"
                --model_name "$model"
                --concurrency "$CONCURRENCY"
                --max_queries "$MAX_QUERIES"
                --use_rag
                --db_path "$DB_PATH"
                --collection_name "$COLLECTION_NAME"
                --rag_top_k "$RAG_TOP_K"
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
echo "RAG Fixed-400 Multi-turn Finished at $(date)"
echo "========================================================"
