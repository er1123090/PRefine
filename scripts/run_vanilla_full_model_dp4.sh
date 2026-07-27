#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if (( $# != 1 )); then
    printf 'Usage: %s {gemma4-12b|gpt-oss-20b}\n' "$0" >&2
    exit 2
fi

PROFILE="$1"
PYTHON_BIN="${PYTHON_BIN:-}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
DP_SIZE="${DP_SIZE:-4}"
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-8002}"
BACKEND_PORT_BASE="${BACKEND_PORT_BASE:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
CONCURRENCY="${CONCURRENCY:-256}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-7200}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_RETRY_TOKENS="${MAX_RETRY_TOKENS:-2048}"
RETRY_ROUNDS="${RETRY_ROUNDS:-4}"
DTYPE="${DTYPE:-bfloat16}"
INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/data/MPT_v2_mix600.json}"
EXPECTED_COUNT="${EXPECTED_COUNT:-6508}"
QUERY="${QUERY:-hint}"
EXCLUDE_EASY_CONFLICT="${EXCLUDE_EASY_CONFLICT:-0}"
THERMAL_GUARD_ENABLED="${THERMAL_GUARD_ENABLED:-1}"
THERMAL_PAUSE_TEMP_C="${THERMAL_PAUSE_TEMP_C:-85}"
THERMAL_RESUME_TEMP_C="${THERMAL_RESUME_TEMP_C:-78}"
THERMAL_POLL_SECONDS="${THERMAL_POLL_SECONDS:-5}"

case "${PROFILE}" in
    gemma4-12b)
        PYTHON_BIN="${PYTHON_BIN:-/tmp/minseo_gemma4_vllm_exact/bin/python}"
        MODEL_REPOSITORY="google/gemma-4-12B-it"
        MODEL_REVISION="12ace6d648d72bd41519e140f1185f34d38c7e3d"
        MODEL_PATH="${MODEL_PATH:-/tmp/minseo_hf_cache/models--google--gemma-4-12B-it/snapshots/${MODEL_REVISION}}"
        SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-google/gemma-4-12B-it}"
        REASONING_PARSER="gemma4"
        REASONING_EFFORT=""
        LANGUAGE_MODEL_ONLY=1
        OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vanilla_llm/full_6508/google_gemma-4-12b-it_thinking_off_optimized}"
        ;;
    gpt-oss-20b)
        PYTHON_BIN="${PYTHON_BIN:-/data/minseo/.venvs/vllm/bin/python}"
        MODEL_REPOSITORY="openai/gpt-oss-20b"
        MODEL_REVISION="6cee5e81ee83917806bbde320786a8fb61efebee"
        MODEL_PATH="${MODEL_PATH:-/data/minseo/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/${MODEL_REVISION}}"
        SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openai/gpt-oss-20b}"
        REASONING_PARSER="openai_gptoss"
        REASONING_EFFORT="${REASONING_EFFORT:-low}"
        LANGUAGE_MODEL_ONLY=0
        OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vanilla_llm/full_6508/openai_gpt-oss-20b_thinking_off_reasoning_low_optimized}"
        ;;
    *)
        printf 'Unknown profile: %s\n' "${PROFILE}" >&2
        exit 2
        ;;
esac

PYTHON_BIN_DIR="$(dirname "${PYTHON_BIN}")"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

RUN_TAG="${RUN_TAG:-${PROFILE}_thinking_off_dp4_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/runtime/vanilla_full_dp4/${PROFILE}}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_TAG}.run.log}"
PROXY_LOG="${PROXY_LOG:-${LOG_DIR}/${RUN_TAG}.proxy.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
VLLM_CACHE_BASE="${VLLM_CACHE_BASE:-/tmp/experiment8_vllm_cache/${PROFILE}}"
API_BASE="http://127.0.0.1:${PORT}/v1"

PROXY_SCRIPT="${SCRIPT_DIR}/least_loaded_openai_proxy.py"
INFERENCE_SCRIPT="${SCRIPT_DIR}/run_local_full_inference.py"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    printf 'Model snapshot is incomplete or missing: %s\n' "${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${VLLM_CACHE_BASE}"
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

cleanup() {
    local exit_code=$?
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
            status "Stopping vLLM replica pid=${server_pid}"
            kill "${server_pid}" 2>/dev/null || true
        fi
    done
    for server_pid in "${SERVER_PIDS[@]}"; do
        wait "${server_pid}" 2>/dev/null || true
    done

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
                status "vLLM replica ${replica_index} ready: GPU=${GPU_IDS[$replica_index]}, port=${server_port}"
                break
            fi
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                status "vLLM replica ${replica_index} exited during startup"
                tail -n 160 "${server_log}" | tee -a "${RUN_LOG}"
                return 1
            fi
            sleep 5
            waited=$((waited + 5))
        done
        if (( waited >= 3600 )); then
            status "Timed out waiting for vLLM replica ${replica_index}"
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
    tail -n 80 "${PROXY_LOG}" | tee -a "${RUN_LOG}"
    return 1
}

preflight() {
    "${PYTHON_BIN}" - \
        "${API_BASE}" \
        "${SERVED_MODEL_NAME}" \
        "${REASONING_EFFORT}" \
        "${REQUEST_TIMEOUT_SECONDS}" <<'PY'
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
            "content": 'Return exactly: GetWeather(location="Seoul")',
        }
    ],
    "temperature": 0.0,
    "max_tokens": 256,
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": False},
    },
}
if reasoning_effort:
    request["reasoning_effort"] = reasoning_effort
response = client.chat.completions.create(**request)
content = (response.choices[0].message.content or "").strip()
if not content:
    raise SystemExit("Preflight returned no final content")
print(
    f"preflight passed: model={model}, reasoning_effort={reasoning_effort or 'none'}, "
    f"finish_reason={response.choices[0].finish_reason}"
)
PY
}

status "Starting full vanilla inference profile=${PROFILE}"
status "repository=${MODEL_REPOSITORY}, revision=${MODEL_REVISION}"
status "model_path=${MODEL_PATH}, served_model=${SERVED_MODEL_NAME}"
status "thinking=false, reasoning_effort=${REASONING_EFFORT:-none}, temperature=0.0"
status "GPUs=${CUDA_DEVICES}, TP=${TP_SIZE}, DP=${DP_SIZE}, concurrency=${CONCURRENCY}"
status "input_path=${INPUT_PATH}, query=${QUERY}, exclude_easy_conflict=${EXCLUDE_EASY_CONFLICT}, expected=${EXPECTED_COUNT}"

for replica_index in "${!GPU_IDS[@]}"; do
    server_port=$((BACKEND_PORT_BASE + replica_index))
    replica_log="${LOG_DIR}/${RUN_TAG}.gpu${GPU_IDS[$replica_index]}.server.log"
    replica_cache="${VLLM_CACHE_BASE}/gpu${GPU_IDS[$replica_index]}"
    SERVER_PORTS+=("${server_port}")
    SERVER_LOGS+=("${replica_log}")
    mkdir -p "${replica_cache}"
    touch "${replica_log}"

    extra_server_args=()
    if (( LANGUAGE_MODEL_ONLY == 1 )); then
        extra_server_args+=(--model-impl vllm --language-model-only)
    fi

    CUDA_VISIBLE_DEVICES="${GPU_IDS[$replica_index]}" \
    VLLM_CACHE_ROOT="${replica_cache}" \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
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
        --enforce-eager \
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
preflight | tee -a "${RUN_LOG}"

reasoning_args=()
if [[ -n "${REASONING_EFFORT}" ]]; then
    reasoning_args+=(--reasoning-effort "${REASONING_EFFORT}")
fi

dataset_args=(--input-path "${INPUT_PATH}" --query "${QUERY}")
if (( EXCLUDE_EASY_CONFLICT == 1 )); then
    dataset_args+=(--exclude-easy-conflict)
fi

status "Launching vanilla_llm inference over ${EXPECTED_COUNT} cases"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
    --method vanilla_llm \
    --model "${SERVED_MODEL_NAME}" \
    --api-base "${API_BASE}" \
    --api-key EMPTY \
    --output-dir "${OUTPUT_DIR}" \
    --expected-count "${EXPECTED_COUNT}" \
    --concurrency "${CONCURRENCY}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-retry-tokens "${MAX_RETRY_TOKENS}" \
    --temperature 0.0 \
    --top-p 1.0 \
    --top-k -1 \
    --min-p 0.0 \
    --no-thinking \
    --request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}" \
    --client-max-retries "${CLIENT_MAX_RETRIES}" \
    --retry-rounds "${RETRY_ROUNDS}" \
    --resume \
    "${dataset_args[@]}" \
    "${reasoning_args[@]}" \
    2>&1 | tee -a "${RUN_LOG}"

status "Full vanilla inference complete: ${OUTPUT_DIR}"
