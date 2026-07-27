#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if (( $# != 1 )); then
    printf 'Usage: %s {qwen3-8b|gemma4-12b|gpt-oss-20b}\n' "$0" >&2
    exit 2
fi

PROFILE="$1"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
DP_SIZE="${DP_SIZE:-4}"
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-8002}"
BACKEND_PORT_BASE="${BACKEND_PORT_BASE:-8100}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-7200}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-0}"
INITIAL_MAX_TOKENS="${INITIAL_MAX_TOKENS:-8192}"
MAX_RETRY_TOKENS="${MAX_RETRY_TOKENS:-32768}"
INITIAL_RETRY_ROUNDS="${INITIAL_RETRY_ROUNDS:-3}"
FINAL_RETRY_ROUNDS="${FINAL_RETRY_ROUNDS:-2}"
MAX_RESUME_PASSES="${MAX_RESUME_PASSES:-5}"
SKIP_COMBINATIONS="${SKIP_COMBINATIONS:-}"
THERMAL_GUARD_ENABLED="${THERMAL_GUARD_ENABLED:-0}"
THERMAL_PAUSE_TEMP_C="${THERMAL_PAUSE_TEMP_C:-85}"
THERMAL_RESUME_TEMP_C="${THERMAL_RESUME_TEMP_C:-78}"
THERMAL_POLL_SECONDS="${THERMAL_POLL_SECONDS:-5}"
DTYPE="${DTYPE:-bfloat16}"
INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/data/MPT_v2_mix600.json}"
QUERY="${QUERY:-hint}"
EXCLUDE_EASY_CONFLICT="${EXCLUDE_EASY_CONFLICT:-0}"
EXPECTED_COUNT="${EXPECTED_COUNT:-6508}"
EXPECTED_MEMORY_COUNT="${EXPECTED_MEMORY_COUNT:-600}"
MEMORY_FILENAME="${MEMORY_FILENAME:-memory.jsonl}"

case "${PROFILE}" in
    qwen3-8b)
        PYTHON_BIN="${PYTHON_BIN:-/data/minseo/.venvs/vllm/bin/python}"
        MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
        SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"
        TARGET_SLUG="qwen3_8b"
        REASONING_PARSER="qwen3"
        REASONING_EFFORT=""
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
        CONCURRENCY="${CONCURRENCY:-256}"
        ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
        LANGUAGE_MODEL_ONLY=0
        ENFORCE_EAGER=0
        VLLM_USE_FLASHINFER=1
        ;;
    gemma4-12b)
        PYTHON_BIN="${PYTHON_BIN:-/tmp/minseo_gemma4_vllm_exact/bin/python}"
        MODEL_REVISION="${MODEL_REVISION:-12ace6d648d72bd41519e140f1185f34d38c7e3d}"
        MODEL_PATH="${MODEL_PATH:-/tmp/minseo_hf_cache/models--google--gemma-4-12B-it/snapshots/${MODEL_REVISION}}"
        SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-google/gemma-4-12B-it}"
        TARGET_SLUG="gemma4_12b_it"
        REASONING_PARSER="gemma4"
        REASONING_EFFORT="high"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
        CONCURRENCY="${CONCURRENCY:-256}"
        ASYNC_SCHEDULING=0
        LANGUAGE_MODEL_ONLY=1
        ENFORCE_EAGER=1
        VLLM_USE_FLASHINFER=0
        ;;
    gpt-oss-20b)
        PYTHON_BIN="${PYTHON_BIN:-/data/minseo/.venvs/vllm/bin/python}"
        MODEL_REVISION="${MODEL_REVISION:-6cee5e81ee83917806bbde320786a8fb61efebee}"
        MODEL_PATH="${MODEL_PATH:-/data/minseo/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/${MODEL_REVISION}}"
        SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openai/gpt-oss-20b}"
        TARGET_SLUG="gpt_oss_20b"
        REASONING_PARSER="openai_gptoss"
        REASONING_EFFORT="high"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
        CONCURRENCY="${CONCURRENCY:-256}"
        ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
        LANGUAGE_MODEL_ONLY=0
        ENFORCE_EAGER=0
        VLLM_USE_FLASHINFER=1
        ;;
    *)
        printf 'Unknown profile: %s\n' "${PROFILE}" >&2
        exit 2
        ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'Python environment is missing: %s\n' "${PYTHON_BIN}" >&2
    exit 2
fi
if [[ "${MODEL_PATH}" == /* && ! -f "${MODEL_PATH}/config.json" ]]; then
    printf 'Model snapshot is incomplete or missing: %s\n' "${MODEL_PATH}" >&2
    exit 1
fi

MEMORY_ROOT="${MEMORY_ROOT:-${REPO_ROOT}/outputs/ours_memory/cross_model_mix600_high_reasoning_20260721}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/ours_memory/full_6508}"
PROXY_SCRIPT="${SCRIPT_DIR}/least_loaded_openai_proxy.py"
INFERENCE_SCRIPT="${SCRIPT_DIR}/run_local_full_inference.py"
API_BASE="http://127.0.0.1:${PORT}/v1"

RUN_TAG="${RUN_TAG:-cross_memory_to_${TARGET_SLUG}_dp4_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/runtime/cross_memory_3x3/${TARGET_SLUG}}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_TAG}.run.log}"
PROXY_LOG="${PROXY_LOG:-${LOG_DIR}/${RUN_TAG}.proxy.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
VLLM_CACHE_BASE="${VLLM_CACHE_BASE:-/tmp/experiment8_vllm_cache/cross_${TARGET_SLUG}}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${VLLM_CACHE_BASE}"
touch "${RUN_LOG}" "${PROXY_LOG}"
printf '%s\n' "$$" >"${PID_FILE}"

IFS=',' read -r -a GPU_IDS <<<"${CUDA_DEVICES}"
if (( ${#GPU_IDS[@]} != DP_SIZE )); then
    printf 'Expected %d GPU IDs for DP_SIZE=%d, got: %s\n' \
        "${DP_SIZE}" "${DP_SIZE}" "${CUDA_DEVICES}" >&2
    exit 2
fi

declare -a SERVER_PIDS=()
declare -a SERVER_PORTS=()
declare -a SERVER_LOGS=()
PROXY_PID=""
THERMAL_GUARD_PID=""

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN_LOG}"
}

stop_runtime() {
    if [[ -n "${THERMAL_GUARD_PID}" ]] && kill -0 "${THERMAL_GUARD_PID}" 2>/dev/null; then
        status "Stopping thermal guard pid=${THERMAL_GUARD_PID}"
        kill "${THERMAL_GUARD_PID}" 2>/dev/null || true
        wait "${THERMAL_GUARD_PID}" 2>/dev/null || true
    fi
    if [[ -n "${PROXY_PID}" ]] && kill -0 "${PROXY_PID}" 2>/dev/null; then
        status "Stopping load-balancing proxy pid=${PROXY_PID}"
        kill "${PROXY_PID}" 2>/dev/null || true
        wait "${PROXY_PID}" 2>/dev/null || true
    fi
    local server_pid
    for server_pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "${server_pid}" 2>/dev/null; then
            status "Stopping ${PROFILE} replica pid=${server_pid}"
            kill "${server_pid}" 2>/dev/null || true
        fi
    done
    for server_pid in "${SERVER_PIDS[@]}"; do
        wait "${server_pid}" 2>/dev/null || true
    done
}

cleanup() {
    local exit_code=$?
    stop_runtime
    rm -f "${PID_FILE}"
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

wait_for_backends() {
    local replica_index
    for replica_index in "${!SERVER_PIDS[@]}"; do
        local waited=0
        local server_pid="${SERVER_PIDS[$replica_index]}"
        local server_port="${SERVER_PORTS[$replica_index]}"
        local server_log="${SERVER_LOGS[$replica_index]}"
        while (( waited < 3600 )); do
            if curl -fsS "http://127.0.0.1:${server_port}/health" >/dev/null 2>&1; then
                status "Replica ${replica_index} ready: GPU=${GPU_IDS[$replica_index]}, port=${server_port}"
                break
            fi
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                status "Replica ${replica_index} exited during startup"
                tail -n 160 "${server_log}" | tee -a "${RUN_LOG}"
                return 1
            fi
            sleep 5
            waited=$((waited + 5))
        done
        if (( waited >= 3600 )); then
            status "Timed out waiting for replica ${replica_index}"
            tail -n 160 "${server_log}" | tee -a "${RUN_LOG}"
            return 1
        fi
    done
}

start_thermal_guard() {
    if (( THERMAL_GUARD_ENABLED == 0 )); then
        return 0
    fi

    local waited=0
    local server_pid
    local engine_pid
    local -a engine_pids=()
    while (( waited < 60 )); do
        engine_pids=()
        for server_pid in "${SERVER_PIDS[@]}"; do
            engine_pid="$(
                pgrep -P "${server_pid}" -f '^VLLM::EngineCore' 2>/dev/null \
                    | head -n 1 \
                    || true
            )"
            if [[ -n "${engine_pid}" ]]; then
                engine_pids+=("${engine_pid}")
            fi
        done
        if (( ${#engine_pids[@]} == DP_SIZE )); then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    if (( ${#engine_pids[@]} != DP_SIZE )); then
        status "Thermal guard could not resolve all ${DP_SIZE} EngineCore processes"
        return 1
    fi

    local target_pids
    target_pids="$(IFS=,; printf '%s' "${engine_pids[*]}")"
    local thermal_log="${LOG_DIR}/${RUN_TAG}.thermal.log"
    GPU_IDS="${CUDA_DEVICES}" \
    TARGET_PIDS="${target_pids}" \
    PAUSE_TEMP_C="${THERMAL_PAUSE_TEMP_C}" \
    RESUME_TEMP_C="${THERMAL_RESUME_TEMP_C}" \
    POLL_SECONDS="${THERMAL_POLL_SECONDS}" \
    LOG_PATH="${thermal_log}" \
        bash "${SCRIPT_DIR}/gpu_thermal_guard.sh" >>"${RUN_LOG}" 2>&1 &
    THERMAL_GUARD_PID=$!
    status "Started thermal guard pid=${THERMAL_GUARD_PID}, pause>=${THERMAL_PAUSE_TEMP_C}C, resume<=${THERMAL_RESUME_TEMP_C}C"
}

wait_for_proxy() {
    local waited=0
    while (( waited < 60 )); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            status "Load-balancing proxy ready: port=${PORT}, replicas=${DP_SIZE}"
            return 0
        fi
        if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
            status "Load-balancing proxy exited during startup"
            tail -n 80 "${PROXY_LOG}" | tee -a "${RUN_LOG}"
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    status "Timed out waiting for load-balancing proxy"
    return 1
}

memory_count() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
ids = set()
if path.exists():
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("example_id", ""))
            if not example_id:
                raise SystemExit(f"missing example_id at line {line_number}")
            ids.add(example_id)
print(len(ids))
PY
}

combination_complete() {
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

status "Starting cross-memory target queue: target=${TARGET_SLUG}"
status "model=${MODEL_PATH}, served_model=${SERVED_MODEL_NAME}"
status "GPUs=${CUDA_DEVICES}, TP=${TP_SIZE}, DP=${DP_SIZE}, concurrency=${CONCURRENCY}"
status "max_model_len=${MAX_MODEL_LEN}, max_num_seqs_per_replica=${MAX_NUM_SEQS}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"

if (( VLLM_USE_FLASHINFER == 0 )); then
    export VLLM_USE_FLASHINFER_SAMPLER=0
else
    unset VLLM_USE_FLASHINFER_SAMPLER
fi

for replica_index in "${!GPU_IDS[@]}"; do
    server_port=$((BACKEND_PORT_BASE + replica_index))
    replica_log="${LOG_DIR}/${RUN_TAG}.gpu${GPU_IDS[$replica_index]}.server.log"
    replica_cache="${VLLM_CACHE_BASE}/gpu${GPU_IDS[$replica_index]}"
    SERVER_PORTS+=("${server_port}")
    SERVER_LOGS+=("${replica_log}")
    mkdir -p "${replica_cache}"
    touch "${replica_log}"

    extra_server_args=()
    if (( ASYNC_SCHEDULING == 1 )); then
        extra_server_args+=(--async-scheduling)
    fi
    if (( LANGUAGE_MODEL_ONLY == 1 )); then
        extra_server_args+=(--model-impl vllm --language-model-only)
    fi
    if (( ENFORCE_EAGER == 1 )); then
        extra_server_args+=(--enforce-eager)
    fi

    CUDA_VISIBLE_DEVICES="${GPU_IDS[$replica_index]}" \
    VLLM_CACHE_ROOT="${replica_cache}" \
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --dtype "${DTYPE}" \
        --host 127.0.0.1 \
        --port "${server_port}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --reasoning-parser "${REASONING_PARSER}" \
        --trust-remote-code \
        "${extra_server_args[@]}" \
        >>"${replica_log}" 2>&1 &
    SERVER_PIDS+=("$!")
    status "Started replica ${replica_index}: GPU=${GPU_IDS[$replica_index]}, port=${server_port}, pid=${SERVER_PIDS[$replica_index]}"
done

wait_for_backends
start_thermal_guard

proxy_args=()
for server_port in "${SERVER_PORTS[@]}"; do
    proxy_args+=(--backend "http://127.0.0.1:${server_port}")
done
"${PYTHON_BIN}" "${PROXY_SCRIPT}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    "${proxy_args[@]}" \
    >>"${PROXY_LOG}" 2>&1 &
PROXY_PID=$!
status "Started load-balancing proxy pid=${PROXY_PID}"
wait_for_proxy

"${PYTHON_BIN}" - \
    "${API_BASE}" \
    "${SERVED_MODEL_NAME}" \
    "${REASONING_EFFORT}" \
    "${REQUEST_TIMEOUT_SECONDS}" <<'PY' | tee -a "${RUN_LOG}"
import sys
from openai import OpenAI

api_base, model, reasoning_effort, timeout_seconds = sys.argv[1:]
client = OpenAI(
    api_key="EMPTY",
    base_url=api_base,
    timeout=float(timeout_seconds),
    max_retries=0,
)
request = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": 'Return exactly this function call: GetWeather(location="Seoul")',
        }
    ],
    "temperature": 0.0,
    "max_tokens": 2048,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
}
if reasoning_effort:
    request["reasoning_effort"] = reasoning_effort
response = client.chat.completions.create(**request)
content = (response.choices[0].message.content or "").strip()
if not content:
    raise SystemExit("High-reasoning preflight returned no final content")
print(
    f"high-reasoning preflight passed: finish_reason={response.choices[0].finish_reason}"
)
PY

memory_slugs=(qwen3_8b gemma4_12b_it gpt_oss_20b)
population_args=(
    --input-path "${INPUT_PATH}"
    --query "${QUERY}"
)
if (( EXCLUDE_EASY_CONFLICT == 1 )); then
    population_args+=(--exclude-easy-conflict)
fi

for memory_slug in "${memory_slugs[@]}"; do
    memory_path="${MEMORY_ROOT}/${memory_slug}/${MEMORY_FILENAME}"
    output_dir="${OUTPUT_ROOT}/${memory_slug}_memory__to__${TARGET_SLUG}_high_reasoning_optimized"

    if combination_forced_skip "${memory_slug}" "${TARGET_SLUG}"; then
        status "Skipping combination by operator request: ${memory_slug} -> ${TARGET_SLUG}"
        continue
    fi
    if [[ "$(memory_count "${memory_path}")" != "${EXPECTED_MEMORY_COUNT}" ]]; then
        status "Memory completeness check failed: source=${memory_slug}, expected=${EXPECTED_MEMORY_COUNT}"
        exit 1
    fi
    if combination_complete "${output_dir}"; then
        status "Skipping completed combination: ${memory_slug} -> ${TARGET_SLUG} (${EXPECTED_COUNT}/${EXPECTED_COUNT} OK)"
        continue
    fi

    status "Starting combination: ${memory_slug} -> ${TARGET_SLUG}, cases=${EXPECTED_COUNT}"
    reasoning_args=()
    if [[ -n "${REASONING_EFFORT}" ]]; then
        reasoning_args+=(--reasoning-effort "${REASONING_EFFORT}")
    fi

    pass=1
    while (( pass <= MAX_RESUME_PASSES )); do
        pass_max_tokens="${INITIAL_MAX_TOKENS}"
        pass_retry_rounds="${INITIAL_RETRY_ROUNDS}"
        if (( pass > 1 )); then
            pass_max_tokens="${MAX_RETRY_TOKENS}"
            pass_retry_rounds="${FINAL_RETRY_ROUNDS}"
        fi

        status "Inference pass ${pass}/${MAX_RESUME_PASSES}: ${memory_slug} -> ${TARGET_SLUG}, initial_max_tokens=${pass_max_tokens}"
        set +e
        PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
            --method ours_memory \
            --model "${SERVED_MODEL_NAME}" \
            --api-base "${API_BASE}" \
            --api-key EMPTY \
            --output-dir "${output_dir}" \
            --memory-path "${memory_path}" \
            --expected-count "${EXPECTED_COUNT}" \
            --concurrency "${CONCURRENCY}" \
            --max-tokens "${pass_max_tokens}" \
            --max-retry-tokens "${MAX_RETRY_TOKENS}" \
            --temperature 0.0 \
            --top-p 1.0 \
            --top-k -1 \
            --min-p 0.0 \
            --thinking \
            --request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}" \
            --client-max-retries "${CLIENT_MAX_RETRIES}" \
            --retry-rounds "${pass_retry_rounds}" \
            --resume \
            "${population_args[@]}" \
            "${reasoning_args[@]}" \
            2>&1 | tee -a "${RUN_LOG}"
        inference_status=${PIPESTATUS[0]}
        set -e

        if combination_complete "${output_dir}"; then
            status "Combination complete: ${memory_slug} -> ${TARGET_SLUG} (${EXPECTED_COUNT}/${EXPECTED_COUNT} OK)"
            break
        fi
        status "Combination incomplete after pass ${pass}, inference_exit=${inference_status}; resuming"
        pass=$((pass + 1))
    done

    if ! combination_complete "${output_dir}"; then
        status "Combination failed to reach ${EXPECTED_COUNT}/${EXPECTED_COUNT} after ${MAX_RESUME_PASSES} passes: ${memory_slug} -> ${TARGET_SLUG}"
        exit 1
    fi
done

status "Cross-memory target queue complete: target=${TARGET_SLUG}"
