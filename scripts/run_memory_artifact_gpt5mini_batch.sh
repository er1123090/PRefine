#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data/minseo/.venvs/experiment8/bin/python}"
OPENAI_ENV_FILE="${OPENAI_ENV_FILE:-${REPO_ROOT}/.secrets/openai.env}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PREPARED_ROOT="${PREPARED_ROOT:-${REPO_ROOT}/outputs/mpt_v2_0725_hint_no_easy_conflict}"
AMEM_DIR="${AMEM_DIR:-${PREPARED_ROOT}/amem/openai_gpt-5-mini_minimal_batch}"
LANGMEM_DIR="${LANGMEM_DIR:-${PREPARED_ROOT}/langmem/openai_gpt-5-mini_minimal_batch}"
RAG_DIR="${RAG_DIR:-${PREPARED_ROOT}/rag/openai_gpt-5-mini_minimal_batch}"
LANGMEM_ARTIFACT="${LANGMEM_ARTIFACT:-${REPO_ROOT}/outputs/langmem/MPT_v2_0725_gpt-5-mini_batch/memory.jsonl}"
RAG_DB="${RAG_DB:-${REPO_ROOT}/outputs/rag/MPT_v2_0725_openai_chroma}"
RUN_TAG="${RUN_TAG:-memory_artifact_gpt5mini_batch_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/runtime/memory_artifact_gpt5mini_batch/${RUN_TAG}}"
RUN_LOG="${LOG_DIR}/run.log"
RUNNER="${SCRIPT_DIR}/run_memory_artifact_inference.py"

mkdir -p "${LOG_DIR}"
touch "${RUN_LOG}"

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" \
        | tee -a "${RUN_LOG}"
}

key_args=()
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    status "Using OPENAI_API_KEY from the process environment"
elif [[ -f "${OPENAI_ENV_FILE}" ]]; then
    key_args+=(--api-key-env-file "${OPENAI_ENV_FILE}")
    status "Using OPENAI_API_KEY from ${OPENAI_ENV_FILE}"
else
    printf 'A valid OPENAI_API_KEY is required in the environment or %s\n' \
        "${OPENAI_ENV_FILE}" >&2
    exit 2
fi

prepare_if_needed() {
    local method="$1"
    local output_dir="$2"
    shift 2
    if [[ -s "${output_dir}/manifest.jsonl" \
        && -s "${output_dir}/requests.jsonl" \
        && -s "${output_dir}/prepare_summary.json" ]]; then
        status "${method} preparation already complete: ${output_dir}"
        return
    fi
    status "Preparing ${method} retrieval and byte-stable prompts"
    "${PYTHON_BIN}" "${RUNNER}" prepare \
        --method "${method}" \
        --input-path "${REPO_ROOT}/data/MPT_v2_0725.json" \
        --output-dir "${output_dir}" \
        --model gpt-5-mini \
        --reasoning-effort minimal \
        --max-completion-tokens 2048 \
        --query hint \
        --exclude-easy-conflict \
        --expected-count 4695 \
        --top-k 5 \
        --embedding-model text-embedding-3-small \
        "${key_args[@]}" \
        "$@" \
        2>&1 | tee -a "${RUN_LOG}"
}

if [[ ! -s "${AMEM_DIR}/manifest.jsonl" \
    || ! -s "${AMEM_DIR}/requests.jsonl" \
    || ! -s "${AMEM_DIR}/prepare_summary.json" ]]; then
    printf 'A-MEM prepared inputs are missing from %s\n' "${AMEM_DIR}" >&2
    exit 2
fi

prepare_if_needed langmem "${LANGMEM_DIR}" \
    --memory-path "${LANGMEM_ARTIFACT}"
prepare_if_needed rag "${RAG_DIR}" \
    --db-path "${RAG_DB}" \
    --collection-name user_memories

methods=(amem langmem rag)
directories=("${AMEM_DIR}" "${LANGMEM_DIR}" "${RAG_DIR}")

for index in "${!methods[@]}"; do
    method="${methods[$index]}"
    directory="${directories[$index]}"
    if [[ -f "${directory}/batch_state.json" ]]; then
        status "${method} Batch state already exists; duplicate submission skipped"
    else
        status "Submitting ${method} GPT-5-mini minimal Batch"
        "${PYTHON_BIN}" "${RUNNER}" submit \
            --method "${method}" \
            --prepared-dir "${directory}" \
            "${key_args[@]}" \
            2>&1 | tee -a "${RUN_LOG}"
    fi
done

while true; do
    all_terminal=1
    any_failed=0
    for index in "${!methods[@]}"; do
        method="${methods[$index]}"
        directory="${directories[$index]}"
        "${PYTHON_BIN}" "${RUNNER}" status \
            --method "${method}" \
            --prepared-dir "${directory}" \
            "${key_args[@]}" \
            >>"${RUN_LOG}" 2>&1
        batch_status="$("${PYTHON_BIN}" - "${directory}/batch_state.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "unknown"))
PY
)"
        status "${method} Batch status=${batch_status}"
        case "${batch_status}" in
            completed)
                ;;
            failed|expired|cancelled)
                any_failed=1
                ;;
            *)
                all_terminal=0
                ;;
        esac
    done
    if (( any_failed == 1 )); then
        status "At least one Batch ended unsuccessfully"
        exit 1
    fi
    if (( all_terminal == 1 )); then
        break
    fi
    sleep "${POLL_SECONDS}"
done

for index in "${!methods[@]}"; do
    method="${methods[$index]}"
    directory="${directories[$index]}"
    if [[ -s "${directory}/evaluation.json" ]]; then
        status "${method} results already collected"
        continue
    fi
    status "Collecting and evaluating ${method} Batch results"
    "${PYTHON_BIN}" "${RUNNER}" collect \
        --method "${method}" \
        --prepared-dir "${directory}" \
        "${key_args[@]}" \
        2>&1 | tee -a "${RUN_LOG}"
done

status "All GPT-5-mini minimal Batch inference runs completed"
