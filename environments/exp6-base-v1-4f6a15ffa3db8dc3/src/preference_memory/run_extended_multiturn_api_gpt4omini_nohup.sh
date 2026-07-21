#!/bin/bash

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[ERROR] OPENAI_API_KEY is not set."
    exit 1
fi

DATE_TAG="${DATE_TAG:-$(date +%m%d)}"
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
CONCURRENCY="${CONCURRENCY:-5}"
MAX_QUERIES="${MAX_QUERIES:-400}"
RUN_LABEL="${RUN_LABEL:-nohup}"

INPUT_PATH="/data/minseo/experiments6/data/1229_dev_6.json"
QUERY_PATH="/data/minseo/experiments6/query_new_multi.json"
PREF_LIST_PATH="/data/minseo/experiments6/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments6/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments6/schema_all_extended_complete.json"
PYTHON_SCRIPT="/data/minseo/experiments6/ours_memory/Preference_Memory_step2_ACTION_multiturn_api.py"
BASE_ROOT="/data/minseo/experiments6/ours_memory/inference/extended"
PROMPT_NAME="implicit_zs"
CONTEXT_TYPE="memory_api"
PREF_TYPE="hard"

MEMORY_FOLDERS=(
    "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B"
    "google_gemma-3-12b-it"
    "gpt-4o-mini"
    "Qwen_Qwen3-8B"
)

echo "[RUN] Multi-turn extended hard with ${MODEL_NAME}"

for MEM_FOLDER in "${MEMORY_FOLDERS[@]}"; do
    echo "[MULTI] ${MEM_FOLDER}"

    MEMORY_PATH="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3/${MEM_FOLDER}/_memory1.jsonl"
    OUT_DIR="${BASE_ROOT}/outputs/multiturn/api/${MEM_FOLDER}/${CONTEXT_TYPE}/${PREF_TYPE}/${MODEL_NAME}_high/${PROMPT_NAME}"
    LOG_DIR="${BASE_ROOT}/logs/multiturn/api/${MEM_FOLDER}/${CONTEXT_TYPE}/${PREF_TYPE}/${MODEL_NAME}_high/${PROMPT_NAME}"
    OUTPUT_FILE="${OUT_DIR}/${DATE_TAG}_test1.json"
    LOG_FILE="${LOG_DIR}/${DATE_TAG}_test1_${RUN_LABEL}.log"

    mkdir -p "${OUT_DIR}" "${LOG_DIR}"

    if [[ -s "${OUTPUT_FILE}" ]]; then
        echo "[SKIP] Output already exists: ${OUTPUT_FILE}"
        continue
    fi

    python "${PYTHON_SCRIPT}" \
        --input_path "${INPUT_PATH}" \
        --memory_path "${MEMORY_PATH}" \
        --output_path "${OUTPUT_FILE}" \
        --log_path "${LOG_FILE}" \
        --multiturn_path "${QUERY_PATH}" \
        --pref_list_path "${PREF_LIST_PATH}" \
        --pref_group_path "${PREF_GROUP_PATH}" \
        --tools_schema_path "${TOOLS_SCHEMA_PATH}" \
        --context_type "${CONTEXT_TYPE}" \
        --pref_type "${PREF_TYPE}" \
        --model_name "${MODEL_NAME}" \
        --base_url "" \
        --api_key "ENV" \
        --concurrency "${CONCURRENCY}" \
        --max_queries "${MAX_QUERIES}"
done

echo "[DONE] Multi-turn extended hard run finished."
