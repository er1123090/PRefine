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
USE_RESPONSE_FORMAT="${USE_RESPONSE_FORMAT:-0}"
GENERATION_MAX_TOKENS="${GENERATION_MAX_TOKENS:-8192}"
VERIFICATION_MAX_TOKENS="${VERIFICATION_MAX_TOKENS:-4096}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-7200}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-0}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"

RUN_TAG="${RUN_TAG:-20260724_gpt_oss_20b_high_replica_dp4_c256_b8192_async_prompt_json_resume}"
CACHE_TAG="${CACHE_TAG:-20260723_gpt_oss_20b_high_replica_dp4_resume}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/ours_memory/cross_model_mix600_high_reasoning_20260721/gpt_oss_20b}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/${RUN_TAG}.server.log}"
CONSTRUCTION_LOG="${CONSTRUCTION_LOG:-${LOG_DIR}/${RUN_TAG}.construction.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"

INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/data/MPT_v2_mix600.json}"
MEMORY_OUTPUT="${MEMORY_OUTPUT:-${OUTPUT_DIR}/memory.jsonl}"
VERIFIER_OUTPUT="${VERIFIER_OUTPUT:-${OUTPUT_DIR}/verifier_logs.jsonl}"
REFINEMENT_OUTPUT="${REFINEMENT_OUTPUT:-${OUTPUT_DIR}/refinement_logs.jsonl}"
# Keep the established cross-model memory schema used by the existing 31 rows.
STEP1_SCRIPT="${STEP1_SCRIPT:-/data/minseo/experiment4/ours_memory/Preference_Memory_step1_LATENTPREF_VLLM.py}"
PROXY_SCRIPT="${PROXY_SCRIPT:-${SCRIPT_DIR}/least_loaded_openai_proxy.py}"
API_BASE="http://127.0.0.1:${PORT}/v1"

mkdir -p "${LOG_DIR}"
touch "${SERVER_LOG}" "${CONSTRUCTION_LOG}"
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
PROXY_LOG="${LOG_DIR}/${RUN_TAG}.proxy.log"

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${CONSTRUCTION_LOG}"
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
                tail -n 160 "${server_log}" | tee -a "${CONSTRUCTION_LOG}"
                return 1
            fi
            sleep 5
            waited=$((waited + 5))
        done
        if (( waited >= 3600 )); then
            status "Timed out waiting for vLLM replica ${replica_index}"
            tail -n 160 "${server_log}" | tee -a "${CONSTRUCTION_LOG}"
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
            tail -n 80 "${PROXY_LOG}" | tee -a "${CONSTRUCTION_LOG}"
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    status "Timed out waiting for load-balancing proxy"
    tail -n 80 "${PROXY_LOG}" | tee -a "${CONSTRUCTION_LOG}"
    return 1
}

preflight_reasoning_effort() {
    "${PYTHON_BIN}" - "${API_BASE}" "${SERVED_MODEL_NAME}" "${REASONING_EFFORT}" "${USE_RESPONSE_FORMAT}" <<'PY'
import json
import sys
from openai import OpenAI

api_base, model, effort, use_response_format = sys.argv[1:]
client = OpenAI(api_key="EMPTY", base_url=api_base)
request_kwargs = dict(
    model=model,
    messages=[{"role": "user", "content": "Return a JSON object with key ok and value true."}],
    reasoning_effort=effort,
    max_tokens=4096,
)
if use_response_format == "1":
    request_kwargs["response_format"] = {"type": "json_object"}
response = client.chat.completions.create(**request_kwargs)
message = response.choices[0].message
if not (message.content or "").strip():
    raise SystemExit("Preflight returned no final content")
parsed = json.loads(message.content)
if parsed.get("ok") is not True:
    raise SystemExit(f"Preflight returned unexpected JSON: {parsed}")
print(f"reasoning preflight passed: effort={effort}, finish_reason={response.choices[0].finish_reason}")
PY
}

status "Starting ours_memory Step 1 for mix600"
status "model=${MODEL_NAME}, dtype=${DTYPE}, reasoning_effort=${REASONING_EFFORT}, GPUs=${CUDA_DEVICES}, TP=${TP_SIZE}, DP=${DP_SIZE}"
status "concurrency=${CONCURRENCY}, max_num_seqs_per_replica=${MAX_NUM_SEQS}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}, async_scheduling=${ASYNC_SCHEDULING}, response_format=${USE_RESPONSE_FORMAT}, CUDA Graph=enabled"
status "output=${MEMORY_OUTPUT}"

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
preflight_reasoning_effort | tee -a "${CONSTRUCTION_LOG}"

status "Launching 600-dialogue memory construction with concurrency=${CONCURRENCY}"
client_args=()
if [[ "${USE_RESPONSE_FORMAT}" == "0" ]]; then
    client_args+=(--disable_response_format)
fi
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${STEP1_SCRIPT}" \
    --input "${INPUT_PATH}" \
    --output "${MEMORY_OUTPUT}" \
    --verifier_output "${VERIFIER_OUTPUT}" \
    --refinement_output "${REFINEMENT_OUTPUT}" \
    --model "${SERVED_MODEL_NAME}" \
    --api_base "${API_BASE}" \
    --api_key EMPTY \
    --concurrency "${CONCURRENCY}" \
    --reasoning_effort "${REASONING_EFFORT}" \
    --generation_max_tokens "${GENERATION_MAX_TOKENS}" \
    --verification_max_tokens "${VERIFICATION_MAX_TOKENS}" \
    --request_timeout_seconds "${REQUEST_TIMEOUT_SECONDS}" \
    --client_max_retries "${CLIENT_MAX_RETRIES}" \
    "${client_args[@]}" \
    --resume \
    2>&1 | tee -a "${CONSTRUCTION_LOG}"

status "Memory construction complete"
