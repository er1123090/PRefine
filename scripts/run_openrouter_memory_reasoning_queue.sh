#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/minseo/.venvs/experiment8/bin/python}"
ENV_FILE="${ENV_FILE:-/data/minseo/.config/experiment8/openrouter.env}"
API_BASE="${API_BASE:-https://openrouter.ai/api/v1}"
INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/data/MPT_v2_0725.json}"
TARGET_ROOT="${TARGET_ROOT:-${REPO_ROOT}/outputs/mpt_v2_0725_hint_no_easy_conflict}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TARGET_ROOT}/ours_memory}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${TARGET_ROOT}/runtime/openrouter_memory_reasoning}"
CONCURRENCY="${CONCURRENCY:-256}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-64}"
QWEN_REQUESTS_PER_SECOND="${QWEN_REQUESTS_PER_SECOND:-4}"
GPT_OSS_CONCURRENCY="${GPT_OSS_CONCURRENCY:-256}"
GPT_OSS_REQUESTS_PER_SECOND="${GPT_OSS_REQUESTS_PER_SECOND:-0}"
INITIAL_MAX_TOKENS="${INITIAL_MAX_TOKENS:-8192}"
MAX_RETRY_TOKENS="${MAX_RETRY_TOKENS:-32768}"
RETRY_ROUNDS="${RETRY_ROUNDS:-3}"
MAX_RESUME_PASSES="${MAX_RESUME_PASSES:-5}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1800}"
EXPECTED_COUNT="${EXPECTED_COUNT:-4695}"
SKIP_COMBINATIONS="${SKIP_COMBINATIONS:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'Python environment is missing: %s\n' "${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f "${ENV_FILE}" ]]; then
    printf 'OpenRouter environment file is missing: %s\n' "${ENV_FILE}" >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    printf 'OPENROUTER_API_KEY is not set after loading %s\n' "${ENV_FILE}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${RUNTIME_ROOT}"
RUN_TAG="${RUN_TAG:-openrouter_memory_reasoning_$(date '+%Y%m%d_%H%M%S')}"
RUN_LOG="${RUN_LOG:-${RUNTIME_ROOT}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${RUNTIME_ROOT}/${RUN_TAG}.pid}"
touch "${RUN_LOG}"
printf '%s\n' "$$" >"${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN_LOG}"
}

summary_complete() {
    "${PYTHON_BIN}" - "$1" "${EXPECTED_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]) / "run_summary.json"
expected = int(sys.argv[2])
if not summary_path.exists():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
complete = (
    summary.get("expected") == expected
    and summary.get("checkpoint_unique") == expected
    and summary.get("prediction_count") == expected
    and summary.get("status_counts") == {"OK": expected}
)
raise SystemExit(0 if complete else 1)
PY
}

checkpoint_progress() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "inference.jsonl"
latest = {}
if path.exists():
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[str(row.get("sample_id"))] = row
ok = sum(
    row.get("status") == "OK"
    and row.get("finish_reason") != "length"
    and not str(row.get("llm_output", "")).lstrip().startswith("<think>")
    for row in latest.values()
)
print(f"{ok}/{len(latest)}")
PY
}

memory_count() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

ids = set()
with Path(sys.argv[1]).open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            ids.add(str(json.loads(line).get("example_id", "")))
ids.discard("")
print(len(ids))
PY
}

combination_forced_skip() {
    local combination="$1:$2"
    local candidate
    IFS=',' read -r -a forced_skips <<<"${SKIP_COMBINATIONS}"
    for candidate in "${forced_skips[@]}"; do
        if [[ "${candidate}" == "${combination}" ]]; then
            return 0
        fi
    done
    return 1
}

run_combination() {
    local memory_slug="$1"
    local model_id="$2"
    local target_slug="$3"
    local combination_concurrency="$4"
    local requests_per_second="$5"
    local memory_path="${TARGET_ROOT}/memory/${memory_slug}/_memory.jsonl"
    local output_dir="${OUTPUT_ROOT}/${memory_slug}_memory__to__openrouter_${target_slug}_high_reasoning"

    if combination_forced_skip "${memory_slug}" "${target_slug}"; then
        status "Skipping combination by operator request: ${memory_slug} -> ${model_id}"
        return 0
    fi
    if [[ ! -f "${memory_path}" ]]; then
        status "Missing memory file: ${memory_path}"
        return 1
    fi
    if [[ "$(memory_count "${memory_path}")" != "459" ]]; then
        status "Memory completeness check failed: ${memory_slug}, expected 459 users"
        return 1
    fi
    if summary_complete "${output_dir}"; then
        status "Already complete: ${memory_slug} -> ${model_id} (${EXPECTED_COUNT}/${EXPECTED_COUNT})"
        return 0
    fi

    mkdir -p "${output_dir}"
    local combination_log="${output_dir}/run.log"
    touch "${combination_log}"
    status "Starting: ${memory_slug} -> ${model_id}, cases=${EXPECTED_COUNT}, concurrency=${combination_concurrency}, requests_per_second=${requests_per_second}, progress=$(checkpoint_progress "${output_dir}")"

    local pass=1
    while (( pass <= MAX_RESUME_PASSES )); do
        status "Pass ${pass}/${MAX_RESUME_PASSES}: ${memory_slug} -> ${model_id}, progress=$(checkpoint_progress "${output_dir}")"
        set +e
        PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/run_local_full_inference.py" \
            --method ours_memory \
            --model "${model_id}" \
            --record-model-name "openrouter/${target_slug}-high-reasoning" \
            --api-base "${API_BASE}" \
            --output-dir "${output_dir}" \
            --memory-path "${memory_path}" \
            --input-path "${INPUT_PATH}" \
            --query hint \
            --exclude-easy-conflict \
            --expected-count "${EXPECTED_COUNT}" \
            --concurrency "${combination_concurrency}" \
            --requests-per-second "${requests_per_second}" \
            --max-tokens "${INITIAL_MAX_TOKENS}" \
            --max-retry-tokens "${MAX_RETRY_TOKENS}" \
            --temperature 0.0 \
            --top-p 1.0 \
            --reasoning-effort high \
            --thinking \
            --request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}" \
            --client-max-retries 0 \
            --retry-rounds "${RETRY_ROUNDS}" \
            --retry-delay-seconds 30 \
            --resume \
            2>&1 | tee -a "${combination_log}" "${RUN_LOG}"
        local inference_status="${PIPESTATUS[0]}"
        set -e

        if summary_complete "${output_dir}"; then
            status "Complete: ${memory_slug} -> ${model_id} (${EXPECTED_COUNT}/${EXPECTED_COUNT})"
            return 0
        fi
        status "Incomplete after pass ${pass}: ${memory_slug} -> ${model_id}, exit=${inference_status}, progress=$(checkpoint_progress "${output_dir}")"
        pass=$((pass + 1))
    done

    status "Failed to complete after ${MAX_RESUME_PASSES} passes: ${memory_slug} -> ${model_id}"
    return 1
}

status "OpenRouter memory-reasoning queue started"
status "cases_per_combination=${EXPECTED_COUNT}, qwen_concurrency=${QWEN_CONCURRENCY}, qwen_requests_per_second=${QWEN_REQUESTS_PER_SECOND}, gpt_oss_concurrency=${GPT_OSS_CONCURRENCY}, max_tokens=${INITIAL_MAX_TOKENS}, max_retry_tokens=${MAX_RETRY_TOKENS}"

memory_slugs=(qwen3_8b gemma4_12b_it gpt_oss_20b)

for memory_slug in "${memory_slugs[@]}"; do
    run_combination \
        "${memory_slug}" \
        "qwen/qwen3-8b" \
        "qwen3_8b" \
        "${QWEN_CONCURRENCY}" \
        "${QWEN_REQUESTS_PER_SECOND}"
done

for memory_slug in "${memory_slugs[@]}"; do
    run_combination \
        "${memory_slug}" \
        "openai/gpt-oss-20b" \
        "gpt_oss_20b" \
        "${GPT_OSS_CONCURRENCY}" \
        "${GPT_OSS_REQUESTS_PER_SECOND}"
done

status "OpenRouter memory-reasoning queue finished"
