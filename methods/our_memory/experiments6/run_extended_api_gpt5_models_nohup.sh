#!/bin/bash

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[ERROR] OPENAI_API_KEY is not set."
    exit 1
fi

DATE_TAG="${DATE_TAG:-$(date +%m%d)}"
RUN_LABEL="${RUN_LABEL:-nohup}"
PROMPT_NAME="implicit_zs"
BASE_ROOT="/data/minseo/experiments6/ours_memory/inference/extended"

INPUT_PATH="/data/minseo/experiments6/data/1229_dev_6.json"
SINGLE_QUERY_PATH="/data/minseo/experiments6/query_new_single.json"
MULTI_QUERY_PATH="/data/minseo/experiments6/query_new_multi.json"
PREF_LIST_PATH="/data/minseo/experiments6/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments6/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments6/schema_all_extended_complete.json"
SINGLE_PYTHON_SCRIPT="/data/minseo/experiments6/ours_memory/Preference_Memory_step2_ACTION_singleturn_api.py"
MULTI_PYTHON_SCRIPT="/data/minseo/experiments6/ours_memory/Preference_Memory_step2_ACTION_multiturn_api.py"

SINGLE_CONTEXT="memory_only"
MULTI_CONTEXT="memory_api"
PREF_TYPE="hard"
SINGLE_CONCURRENCY="${SINGLE_CONCURRENCY:-20}"
MULTI_CONCURRENCY="${MULTI_CONCURRENCY:-5}"
MAX_QUERIES="${MAX_QUERIES:-400}"

MEMORY_FOLDERS=(
    "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B"
    "google_gemma-3-12b-it"
    "gpt-4o-mini"
    "Qwen_Qwen3-8B"
)

# model_name:reasoning_effort:effort_label
MODEL_SPECS=(
    "gpt-5-mini:high:high"
    "gpt-5::default"
)

run_singleturn_for_model() {
    local model_name="$1"
    local reasoning_effort="$2"
    local effort_label="$3"
    local model_safe_name="${model_name//\//_}"
    local model_dir_name="${model_safe_name}_${effort_label}"

    echo "[RUN][SINGLE] ${model_name} (${effort_label})"

    for mem_folder in "${MEMORY_FOLDERS[@]}"; do
        echo "[SINGLE][${model_name}] ${mem_folder}"

        local memory_path="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3/${mem_folder}/_memory1.jsonl"
        local out_dir="${BASE_ROOT}/outputs/singleturn/api/${mem_folder}/${SINGLE_CONTEXT}/${PREF_TYPE}/${model_dir_name}/${PROMPT_NAME}"
        local log_dir="${BASE_ROOT}/logs/singleturn/api/${mem_folder}/${SINGLE_CONTEXT}/${PREF_TYPE}/${model_dir_name}/${PROMPT_NAME}"
        local output_file="${out_dir}/${DATE_TAG}_test1.json"
        local log_file="${log_dir}/${DATE_TAG}_test1_${RUN_LABEL}.log"

        mkdir -p "${out_dir}" "${log_dir}"

        if [[ -s "${output_file}" ]]; then
            echo "[SKIP][SINGLE] Output already exists: ${output_file}"
            continue
        fi

        local reasoning_args=()
        if [[ -n "${reasoning_effort}" ]]; then
            reasoning_args+=(--reasoning_effort "${reasoning_effort}")
        fi

        python "${SINGLE_PYTHON_SCRIPT}" \
            --input_path "${INPUT_PATH}" \
            --memory_path "${memory_path}" \
            --output_path "${output_file}" \
            --log_path "${log_file}" \
            --query_path "${SINGLE_QUERY_PATH}" \
            --pref_list_path "${PREF_LIST_PATH}" \
            --pref_group_path "${PREF_GROUP_PATH}" \
            --tools_schema_path "${TOOLS_SCHEMA_PATH}" \
            --context_type "${SINGLE_CONTEXT}" \
            --pref_type "${PREF_TYPE}" \
            --model_name "${model_name}" \
            --concurrency "${SINGLE_CONCURRENCY}" \
            --max_queries "${MAX_QUERIES}" \
            "${reasoning_args[@]}"
    done
}

run_multiturn_for_model() {
    local model_name="$1"
    local reasoning_effort="$2"
    local effort_label="$3"
    local model_safe_name="${model_name//\//_}"
    local model_dir_name="${model_safe_name}_${effort_label}"

    echo "[RUN][MULTI] ${model_name} (${effort_label})"

    for mem_folder in "${MEMORY_FOLDERS[@]}"; do
        echo "[MULTI][${model_name}] ${mem_folder}"

        local memory_path="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3/${mem_folder}/_memory1.jsonl"
        local out_dir="${BASE_ROOT}/outputs/multiturn/api/${mem_folder}/${MULTI_CONTEXT}/${PREF_TYPE}/${model_dir_name}/${PROMPT_NAME}"
        local log_dir="${BASE_ROOT}/logs/multiturn/api/${mem_folder}/${MULTI_CONTEXT}/${PREF_TYPE}/${model_dir_name}/${PROMPT_NAME}"
        local output_file="${out_dir}/${DATE_TAG}_test1.json"
        local log_file="${log_dir}/${DATE_TAG}_test1_${RUN_LABEL}.log"

        mkdir -p "${out_dir}" "${log_dir}"

        if [[ -s "${output_file}" ]]; then
            echo "[SKIP][MULTI] Output already exists: ${output_file}"
            continue
        fi

        local reasoning_args=()
        if [[ -n "${reasoning_effort}" ]]; then
            reasoning_args+=(--reasoning_effort "${reasoning_effort}")
        fi

        python "${MULTI_PYTHON_SCRIPT}" \
            --input_path "${INPUT_PATH}" \
            --memory_path "${memory_path}" \
            --output_path "${output_file}" \
            --log_path "${log_file}" \
            --multiturn_path "${MULTI_QUERY_PATH}" \
            --pref_list_path "${PREF_LIST_PATH}" \
            --pref_group_path "${PREF_GROUP_PATH}" \
            --tools_schema_path "${TOOLS_SCHEMA_PATH}" \
            --context_type "${MULTI_CONTEXT}" \
            --pref_type "${PREF_TYPE}" \
            --model_name "${model_name}" \
            --base_url "" \
            --api_key "ENV" \
            --concurrency "${MULTI_CONCURRENCY}" \
            --max_queries "${MAX_QUERIES}" \
            "${reasoning_args[@]}"
    done
}

echo "[RUN] Extended hard API inference with gpt-5-mini(high) and gpt-5(default)"

for spec in "${MODEL_SPECS[@]}"; do
    IFS=":" read -r model_name reasoning_effort effort_label <<< "${spec}"
    run_singleturn_for_model "${model_name}" "${reasoning_effort}" "${effort_label}"
    run_multiturn_for_model "${model_name}" "${reasoning_effort}" "${effort_label}"
done

echo "[DONE] Extended hard API inference finished."
