#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data/minseo/.venvs/vllm/bin/python}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-oss-20b}"
DTYPE="${DTYPE:-bfloat16}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-4}"
PORT="${PORT:-8002}"
BACKEND_PORT_BASE="${BACKEND_PORT_BASE:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
CONCURRENCY="${CONCURRENCY:-256}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-7200}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-0}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"

RUN_TAG="${RUN_TAG:-gpt_oss_20b_high_replica_dp4_inference_$(date '+%Y%m%d_%H%M%S')}"
CACHE_TAG="${CACHE_TAG:-gpt_oss_20b_high_replica_dp4}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/runtime/gpt_oss_20b_high_dp4}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_TAG}.run.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
PROXY_LOG="${PROXY_LOG:-${LOG_DIR}/${RUN_TAG}.proxy.log}"

PROXY_SCRIPT="${PROXY_SCRIPT:-${SCRIPT_DIR}/least_loaded_openai_proxy.py}"
INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-${SCRIPT_DIR}/run_inference.py}"
API_BASE="http://127.0.0.1:${PORT}/v1"

if (( $# == 0 )); then
    printf 'Usage: %s --method {vanilla_llm,ours_memory} --turn {single,multi} --pref_type TYPE [run_inference.py options]\n' "$0" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
touch "${RUN_LOG}" "${PROXY_LOG}"
echo "$$" >"${PID_FILE}"

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

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN_LOG}"
}

cleanup() {
    local exit_code=$?
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

preflight_reasoning_effort() {
    "${PYTHON_BIN}" - \
        "${API_BASE}" \
        "${SERVED_MODEL_NAME}" \
        "${REASONING_EFFORT}" \
        "${REQUEST_TIMEOUT_SECONDS}" \
        "${CLIENT_MAX_RETRIES}" <<'PY'
import sys
from openai import OpenAI

api_base, model, effort, timeout_seconds, max_retries = sys.argv[1:]
client = OpenAI(
    api_key="EMPTY",
    base_url=api_base,
    timeout=float(timeout_seconds),
    max_retries=int(max_retries),
)
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": 'Return exactly this function call: GetWeather(location="Seoul")',
        }
    ],
    reasoning_effort=effort,
)
content = (response.choices[0].message.content or "").strip()
if not content:
    raise SystemExit("Preflight returned no final content")
print(
    f"reasoning preflight passed: effort={effort}, "
    f"finish_reason={response.choices[0].finish_reason}"
)
PY
}

status "Starting optimized gpt-oss-20b inference runtime"
status "model=${MODEL_NAME}, dtype=${DTYPE}, reasoning_effort=${REASONING_EFFORT}, GPUs=${CUDA_DEVICES}, TP=${TP_SIZE}, DP=${DP_SIZE}"
status "concurrency=${CONCURRENCY}, max_num_seqs_per_replica=${MAX_NUM_SEQS}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}, async_scheduling=${ASYNC_SCHEDULING}, structured_output=off"

for replica_index in "${!GPU_IDS[@]}"; do
    server_port=$((BACKEND_PORT_BASE + replica_index))
    replica_log="${LOG_DIR}/${RUN_TAG}.gpu${GPU_IDS[$replica_index]}.server.log"
    replica_cache="${LOG_DIR}/${CACHE_TAG}.gpu${GPU_IDS[$replica_index]}.cache"
    SERVER_PORTS+=("${server_port}")
    SERVER_LOGS+=("${replica_log}")
    mkdir -p "${replica_cache}"
    touch "${replica_log}"

    scheduler_args=()
    if [[ "${ASYNC_SCHEDULING}" == "1" ]]; then
        scheduler_args+=(--async-scheduling)
    fi

    CUDA_VISIBLE_DEVICES="${GPU_IDS[$replica_index]}" \
    VLLM_CACHE_ROOT="${replica_cache}" \
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_NAME}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --dtype "${DTYPE}" \
        --host 127.0.0.1 \
        --port "${server_port}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --reasoning-parser openai_gptoss \
        --trust-remote-code \
        "${scheduler_args[@]}" \
        >>"${replica_log}" 2>&1 &
    SERVER_PIDS+=("$!")
    status "Started vLLM replica ${replica_index}: GPU=${GPU_IDS[$replica_index]}, port=${server_port}, pid=${SERVER_PIDS[$replica_index]}"
done

wait_for_backends

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
preflight_reasoning_effort | tee -a "${RUN_LOG}"

status "Launching inference with optimized client settings"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
    "$@" \
    --model "${SERVED_MODEL_NAME}" \
    --base_url "${API_BASE}" \
    --api_key EMPTY \
    --concurrency "${CONCURRENCY}" \
    --reasoning_effort "${REASONING_EFFORT}" \
    --request_timeout_seconds "${REQUEST_TIMEOUT_SECONDS}" \
    --client_max_retries "${CLIENT_MAX_RETRIES}" \
    --python "${PYTHON_BIN}" \
    2>&1 | tee -a "${RUN_LOG}"

status "Inference complete"
