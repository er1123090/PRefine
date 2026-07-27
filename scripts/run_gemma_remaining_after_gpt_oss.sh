#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPT_OSS_PID_FILE="${GPT_OSS_PID_FILE:-${REPO_ROOT}/outputs/mpt_v2_0725_hint_no_easy_conflict/runtime/gpt_oss_dp4/gpt_oss_memory_self_dp4_after_gemma_20260726.pid}"

while [[ -f "${GPT_OSS_PID_FILE}" ]]; do
    gpt_oss_pid=""
    read -r gpt_oss_pid <"${GPT_OSS_PID_FILE}" || true
    if [[ -z "${gpt_oss_pid}" ]] || ! kill -0 "${gpt_oss_pid}" 2>/dev/null; then
        break
    fi
    printf '[%(%Y-%m-%d %H:%M:%S)T] Waiting for GPU GPT-OSS queue pid=%s\n' -1 "${gpt_oss_pid}"
    sleep 10
done

printf '[%(%Y-%m-%d %H:%M:%S)T] Resuming GPT-OSS memory -> Gemma on GPUs 0,1,2,3\n' -1
exec env \
    CUDA_DEVICES=0,1,2,3 \
    PORT=8002 \
    BACKEND_PORT_BASE=8200 \
    MEMORY_ROOT="${REPO_ROOT}/outputs/mpt_v2_0725_hint_no_easy_conflict/memory" \
    MEMORY_FILENAME=_memory.jsonl \
    OUTPUT_ROOT="${REPO_ROOT}/outputs/mpt_v2_0725_hint_no_easy_conflict/ours_memory" \
    INPUT_PATH="${REPO_ROOT}/data/MPT_v2_0725.json" \
    EXPECTED_COUNT=4695 \
    EXPECTED_MEMORY_COUNT=459 \
    EXCLUDE_EASY_CONFLICT=1 \
    QUERY=hint \
    LOG_DIR="${REPO_ROOT}/outputs/mpt_v2_0725_hint_no_easy_conflict/runtime/gemma4_dp4" \
    RUN_TAG=gpt_oss_memory_to_gemma_resume_after_gpt_oss_20260726 \
    MAX_RESUME_PASSES=20 \
    INITIAL_MAX_TOKENS=32768 \
    MAX_RETRY_TOKENS=32768 \
    THERMAL_GUARD_ENABLED=1 \
    THERMAL_PAUSE_TEMP_C=85 \
    THERMAL_RESUME_TEMP_C=78 \
    SKIP_COMBINATIONS=qwen3_8b:gemma4_12b_it,gemma4_12b_it:gemma4_12b_it \
    bash "${SCRIPT_DIR}/run_cross_target_memory_dp4.sh" gemma4-12b
