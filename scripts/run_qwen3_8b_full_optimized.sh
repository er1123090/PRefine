#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data/minseo/.venvs/vllm/bin/python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"
DTYPE="${DTYPE:-bfloat16}"
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
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-7200}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-0}"
VANILLA_MAX_TOKENS="${VANILLA_MAX_TOKENS:-256}"
VANILLA_MAX_RETRY_TOKENS="${VANILLA_MAX_RETRY_TOKENS:-2048}"
VANILLA_RETRY_ROUNDS="${VANILLA_RETRY_ROUNDS:-4}"
VANILLA_TEMPERATURE="${VANILLA_TEMPERATURE:-0.0}"
VANILLA_TOP_P="${VANILLA_TOP_P:-1.0}"
VANILLA_TOP_K="${VANILLA_TOP_K:--1}"
VANILLA_MIN_P="${VANILLA_MIN_P:-0.0}"
OURS_MAX_TOKENS="${OURS_MAX_TOKENS:-8192}"
OURS_MAX_RETRY_TOKENS="${OURS_MAX_RETRY_TOKENS:-32768}"
OURS_RETRY_ROUNDS="${OURS_RETRY_ROUNDS:-3}"
OURS_TEMPERATURE="${OURS_TEMPERATURE:-0.0}"
OURS_TOP_P="${OURS_TOP_P:-1.0}"
OURS_TOP_K="${OURS_TOP_K:--1}"
OURS_MIN_P="${OURS_MIN_P:-0.0}"

RUN_TAG="${RUN_TAG:-qwen3_8b_full_optimized_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/runtime/qwen3_8b_full_optimized}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_TAG}.run.log}"
PROXY_LOG="${PROXY_LOG:-${LOG_DIR}/${RUN_TAG}.proxy.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
CACHE_TAG="${CACHE_TAG:-qwen3_8b_replica_dp4}"

VANILLA_OUTPUT_DIR="${VANILLA_OUTPUT_DIR:-${REPO_ROOT}/outputs/vanilla_llm/full_6508/qwen_qwen3-8b_thinking_off_optimized}"
MEMORY_ROOT="${MEMORY_ROOT:-${REPO_ROOT}/outputs/ours_memory/cross_model_mix600_high_reasoning_20260721/qwen3_8b}"
MEMORY_OUTPUT="${MEMORY_OUTPUT:-${MEMORY_ROOT}/memory.jsonl}"
OURS_OUTPUT_DIR="${OURS_OUTPUT_DIR:-${REPO_ROOT}/outputs/ours_memory/full_6508/qwen3_8b_memory__to__qwen3_8b_high_reasoning_optimized}"

PROXY_SCRIPT="${PROXY_SCRIPT:-${SCRIPT_DIR}/least_loaded_openai_proxy.py}"
FULL_INFERENCE_SCRIPT="${FULL_INFERENCE_SCRIPT:-${SCRIPT_DIR}/run_local_full_inference.py}"
API_BASE="http://127.0.0.1:${PORT}/v1"

mkdir -p "${LOG_DIR}" "${MEMORY_ROOT}"
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

preflight() {
    "${PYTHON_BIN}" - "${API_BASE}" "${SERVED_MODEL_NAME}" <<'PY'
import sys
from openai import OpenAI

api_base, model = sys.argv[1:]
client = OpenAI(api_key="EMPTY", base_url=api_base, timeout=7200, max_retries=0)
for thinking in (False, True):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": 'Return exactly: GetWeather(location="Seoul")',
            }
        ],
        temperature=0.0,
        max_tokens=1024 if thinking else 256,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise SystemExit(f"Preflight returned no final content: thinking={thinking}")
    print(
        f"preflight passed: thinking={thinking}, "
        f"finish_reason={response.choices[0].finish_reason}"
    )
PY
}

memory_count() {
    "${PYTHON_BIN}" - "${MEMORY_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
ids = set()
if path.exists():
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
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

status "Starting Qwen3-8B full optimized queue"
status "stages=vanilla_6508 -> reuse_qwen3_8b_memory_600 -> ours_memory_6508"
status "model=${MODEL_NAME}, GPUs=${CUDA_DEVICES}, TP=${TP_SIZE}, DP=${DP_SIZE}, concurrency=${CONCURRENCY}"
status "max_num_seqs_per_replica=${MAX_NUM_SEQS}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}, async_scheduling=${ASYNC_SCHEDULING}"

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
        --reasoning-parser qwen3 \
        --trust-remote-code \
        "${scheduler_args[@]}" \
        >>"${replica_log}" 2>&1 &
    SERVER_PIDS+=("$!")
    status "Started replica ${replica_index}: GPU=${GPU_IDS[$replica_index]}, port=${server_port}, pid=${SERVER_PIDS[$replica_index]}"
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
preflight | tee -a "${RUN_LOG}"

status "Stage 1/3: Qwen3-8B vanilla inference over all 6,508 cases"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${FULL_INFERENCE_SCRIPT}" \
    --method vanilla_llm \
    --model "${SERVED_MODEL_NAME}" \
    --api-base "${API_BASE}" \
    --api-key EMPTY \
    --output-dir "${VANILLA_OUTPUT_DIR}" \
    --expected-count 6508 \
    --concurrency "${CONCURRENCY}" \
    --max-tokens "${VANILLA_MAX_TOKENS}" \
    --max-retry-tokens "${VANILLA_MAX_RETRY_TOKENS}" \
    --temperature "${VANILLA_TEMPERATURE}" \
    --top-p "${VANILLA_TOP_P}" \
    --top-k "${VANILLA_TOP_K}" \
    --min-p "${VANILLA_MIN_P}" \
    --no-thinking \
    --request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}" \
    --client-max-retries "${CLIENT_MAX_RETRIES}" \
    --retry-rounds "${VANILLA_RETRY_ROUNDS}" \
    --resume \
    2>&1 | tee -a "${RUN_LOG}"

status "Stage 2/3: Reuse existing Qwen3-8B high-reasoning memory"
if [[ "$(memory_count)" != "600" ]]; then
    status "Memory completeness check failed: completed=$(memory_count), expected=600"
    exit 1
fi
status "Existing Qwen3-8B memory verified: 600/600"

status "Stage 3/3: Qwen3-8B memory -> Qwen3-8B high-reasoning inference over all 6,508 cases"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${FULL_INFERENCE_SCRIPT}" \
    --method ours_memory \
    --model "${SERVED_MODEL_NAME}" \
    --api-base "${API_BASE}" \
    --api-key EMPTY \
    --output-dir "${OURS_OUTPUT_DIR}" \
    --memory-path "${MEMORY_OUTPUT}" \
    --expected-count 6508 \
    --concurrency "${CONCURRENCY}" \
    --max-tokens "${OURS_MAX_TOKENS}" \
    --max-retry-tokens "${OURS_MAX_RETRY_TOKENS}" \
    --temperature "${OURS_TEMPERATURE}" \
    --top-p "${OURS_TOP_P}" \
    --top-k "${OURS_TOP_K}" \
    --min-p "${OURS_MIN_P}" \
    --thinking \
    --request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}" \
    --client-max-retries "${CLIENT_MAX_RETRIES}" \
    --retry-rounds "${OURS_RETRY_ROUNDS}" \
    --resume \
    2>&1 | tee -a "${RUN_LOG}"

status "Qwen3-8B full optimized queue complete"
