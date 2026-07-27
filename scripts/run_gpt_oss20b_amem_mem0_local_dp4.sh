#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-full}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    printf 'Usage: %s [smoke|full]\n' "$0" >&2
    exit 2
fi

VLLM_PYTHON="${VLLM_PYTHON:-/data/minseo/.venvs/vllm/bin/python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-/data/minseo/.venvs/experiment8/bin/python}"
MODEL_REVISION="${MODEL_REVISION:-6cee5e81ee83917806bbde320786a8fb61efebee}"
MODEL_PATH="${MODEL_PATH:-/data/minseo/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/${MODEL_REVISION}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-oss-20b}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
DP_SIZE="${DP_SIZE:-4}"
PROXY_PORT="${PROXY_PORT:-8003}"
BACKEND_PORT_BASE="${BACKEND_PORT_BASE:-8200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_TEMP_LIMIT_C="${GPU_TEMP_LIMIT_C:-85}"
GPU_TEMP_RESUME_C="${GPU_TEMP_RESUME_C:-70}"
GPU_TEMP_POLL_SECONDS="${GPU_TEMP_POLL_SECONDS:-5}"

AMEM_CONCURRENCY="${AMEM_CONCURRENCY:-256}"
AMEM_EMBEDDING_CONCURRENCY="${AMEM_EMBEDDING_CONCURRENCY:-8}"
MEM0_CONCURRENCY="${MEM0_CONCURRENCY:-32}"
AMEM_MAX_COMPLETION_TOKENS="${AMEM_MAX_COMPLETION_TOKENS:-1536}"
MEM0_MAX_COMPLETION_TOKENS="${MEM0_MAX_COMPLETION_TOKENS:-4096}"
AMEM_RESPONSE_FORMAT="${AMEM_RESPONSE_FORMAT:-none}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-7200}"
RETRY_COUNT="${RETRY_COUNT:-3}"

SMOKE_MAX_EXAMPLES="${SMOKE_MAX_EXAMPLES:-16}"
SMOKE_MAX_SESSIONS="${SMOKE_MAX_SESSIONS:-1}"

RUN_TAG="${RUN_TAG:-gpt_oss20b_amem_mem0_local_dp4_${MODE}_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/runtime/gpt_oss20b_amem_mem0_local/${RUN_TAG}}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/experiment8_gpt_oss20b_amem_mem0_cache}"
RUN_LOG="${LOG_DIR}/run.log"
PROXY_LOG="${LOG_DIR}/proxy.log"
GPU_LOG="${LOG_DIR}/gpu_dmon.log"
PID_FILE="${LOG_DIR}/runner.pid"

if [[ "${MODE}" == "smoke" ]]; then
    AMEM_OUTPUT_DIR="${AMEM_OUTPUT_DIR:-${REPO_ROOT}/outputs/amem/smoke_gpt-oss-20b_low_local_dp4}"
    MEM0_OUTPUT_ROOT="${MEM0_OUTPUT_ROOT:-${REPO_ROOT}/outputs/mem0_local/smoke_gpt-oss-20b_low_local_dp4}"
else
    AMEM_OUTPUT_DIR="${AMEM_OUTPUT_DIR:-${REPO_ROOT}/outputs/amem/MPT_v2_0725_gpt-oss-20b_low_local_dp4}"
    MEM0_OUTPUT_ROOT="${MEM0_OUTPUT_ROOT:-${REPO_ROOT}/outputs/mem0_local/MPT_v2_0725_gpt-oss-20b_low_local_dp4}"
fi

MEM0_METRICS="${MEM0_METRICS:-${MEM0_OUTPUT_ROOT}/construction.jsonl}"
MEM0_VECTOR_STORE="${MEM0_VECTOR_STORE:-${MEM0_OUTPUT_ROOT}/qdrant}"
MEM0_HISTORY_DB="${MEM0_HISTORY_DB:-${MEM0_OUTPUT_ROOT}/history.sqlite}"
MEM0_COLLECTION="${MEM0_COLLECTION:-experiment8_mem0_local_0725_gpt_oss20b_low}"
API_BASE="http://127.0.0.1:${PROXY_PORT}/v1"
PROXY_SCRIPT="${SCRIPT_DIR}/least_loaded_openai_proxy.py"

mkdir -p "${LOG_DIR}" "${CACHE_ROOT}" "${AMEM_OUTPUT_DIR}" "${MEM0_OUTPUT_ROOT}"
touch "${RUN_LOG}" "${PROXY_LOG}" "${GPU_LOG}"
printf '%s\n' "$$" >"${PID_FILE}"

IFS=',' read -r -a GPU_IDS <<<"${CUDA_DEVICES}"
if (( ${#GPU_IDS[@]} != DP_SIZE )); then
    printf 'Expected %d GPU IDs for DP_SIZE=%d, got %s\n' \
        "${DP_SIZE}" "${DP_SIZE}" "${CUDA_DEVICES}" >&2
    exit 2
fi
if [[ ! "${GPU_TEMP_LIMIT_C}" =~ ^[0-9]+$ ]] \
    || [[ ! "${GPU_TEMP_RESUME_C}" =~ ^[0-9]+$ ]] \
    || [[ ! "${GPU_TEMP_POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Thermal guard values must be non-negative integer temperatures and a positive poll interval\n' >&2
    exit 2
fi
if (( GPU_TEMP_RESUME_C >= GPU_TEMP_LIMIT_C )); then
    printf 'GPU_TEMP_RESUME_C must be lower than GPU_TEMP_LIMIT_C\n' >&2
    exit 2
fi

declare -a SERVER_PIDS=()
declare -a SERVER_PORTS=()
declare -a SERVER_LOGS=()
PROXY_PID=""
GPU_MONITOR_PID=""
THERMAL_GUARD_PID=""
AMEM_PID=""
MEM0_PID=""

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN_LOG}"
}

stop_pid() {
    local pid="$1"
    local label="$2"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        status "Stopping ${label} pid=${pid}"
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}

stop_process_group() {
    local pid="$1"
    local label="$2"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        status "Stopping ${label} process group pgid=${pid}"
        kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}

signal_pid_if_running() {
    local signal_name="$1"
    local pid="$2"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill "-${signal_name}" "${pid}" 2>/dev/null || true
    fi
}

signal_server_groups() {
    local signal_name="$1"
    local server_pid
    for server_pid in "${SERVER_PIDS[@]}"; do
        if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
            kill "-${signal_name}" -- "-${server_pid}" 2>/dev/null \
                || kill "-${signal_name}" "${server_pid}" 2>/dev/null \
                || true
        fi
    done
}

pause_workload() {
    signal_pid_if_running STOP "${AMEM_PID}"
    signal_pid_if_running STOP "${MEM0_PID}"
    signal_server_groups STOP
}

resume_workload() {
    signal_server_groups CONT
    signal_pid_if_running CONT "${AMEM_PID}"
    signal_pid_if_running CONT "${MEM0_PID}"
}

thermal_guard() {
    local paused=0
    local max_temperature=0
    local temperature
    local readings
    while true; do
        if ! readings="$(nvidia-smi \
            --query-gpu=temperature.gpu \
            --format=csv,noheader,nounits \
            -i "${CUDA_DEVICES}" 2>/dev/null)"; then
            status "Thermal guard could not read GPU temperatures; retrying"
            sleep "${GPU_TEMP_POLL_SECONDS}"
            continue
        fi

        max_temperature=0
        while IFS= read -r temperature; do
            temperature="${temperature//[[:space:]]/}"
            if [[ "${temperature}" =~ ^[0-9]+$ ]] \
                && (( temperature > max_temperature )); then
                max_temperature="${temperature}"
            fi
        done <<<"${readings}"

        if (( paused == 0 && max_temperature >= GPU_TEMP_LIMIT_C )); then
            status "Thermal guard pausing workload: max_gpu_temp=${max_temperature}C >= limit=${GPU_TEMP_LIMIT_C}C"
            pause_workload
            paused=1
        elif (( paused == 1 && max_temperature <= GPU_TEMP_RESUME_C )); then
            resume_workload
            status "Thermal guard resumed workload: max_gpu_temp=${max_temperature}C <= resume=${GPU_TEMP_RESUME_C}C"
            paused=0
        fi
        sleep "${GPU_TEMP_POLL_SECONDS}"
    done
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    stop_pid "${THERMAL_GUARD_PID}" "thermal guard"
    THERMAL_GUARD_PID=""
    resume_workload
    stop_pid "${AMEM_PID}" "A-MEM client"
    stop_pid "${MEM0_PID}" "Mem0 client"
    stop_pid "${GPU_MONITOR_PID}" "GPU monitor"
    stop_pid "${PROXY_PID}" "load-balancing proxy"
    local server_pid
    for server_pid in "${SERVER_PIDS[@]}"; do
        stop_process_group "${server_pid}" "vLLM replica"
    done
    status "Runner exiting with code=${exit_code}"
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

wait_for_proxy() {
    local waited=0
    while (( waited < 60 )); do
        if curl -fsS "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
            status "Proxy ready: port=${PROXY_PORT}, replicas=${DP_SIZE}"
            return 0
        fi
        if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
            status "Proxy exited during startup"
            tail -n 100 "${PROXY_LOG}" | tee -a "${RUN_LOG}"
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    status "Timed out waiting for proxy"
    return 1
}

preflight_structured_outputs() {
    local ports_csv
    ports_csv="$(IFS=,; printf '%s' "${SERVER_PORTS[*]}")"
    "${CLIENT_PYTHON}" - \
        "${ports_csv}" \
        "${SERVED_MODEL_NAME}" \
        "${REASONING_EFFORT}" \
        "${AMEM_RESPONSE_FORMAT}" <<'PY'
import asyncio
import json
import sys

from openai import AsyncOpenAI

from methods.amem.build_memory_batch import build_metadata_request
from methods.amem.build_memory_local import (
    _chat_body_from_batch_request,
    parse_local_message_json,
)

ports = [int(value) for value in sys.argv[1].split(",")]
model = sys.argv[2]
effort = sys.argv[3]
response_format = sys.argv[4]

async def check(port: int) -> None:
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://127.0.0.1:{port}/v1",
        timeout=600.0,
        max_retries=0,
    )
    request = build_metadata_request(
        custom_id=f"preflight-{port}",
        model=model,
        reasoning_effort=effort,
        max_completion_tokens=1024,
        content="User: I prefer a quiet hotel room.",
    )
    request_body = _chat_body_from_batch_request(
        request,
        response_format=response_format,
    )
    response = await client.chat.completions.create(**request_body)
    body = response.model_dump()
    try:
        parsed = parse_local_message_json(body)
    except Exception:
        content = body.get("choices", [{}])[0].get("message", {}).get("content")
        print(
            json.dumps(
                {
                    "port": port,
                    "ok": False,
                    "raw_content_repr": repr(content),
                }
            ),
            file=sys.stderr,
        )
        raise
    if not parsed.get("context"):
        raise RuntimeError(f"Replica {port} returned no metadata context: {parsed}")
    await client.close()
    print(json.dumps({"port": port, "ok": True, "usage": body.get("usage")}))

async def main() -> None:
    await asyncio.gather(*(check(port) for port in ports))

asyncio.run(main())
PY
}

status "Starting GPT-OSS-20B DP4 memory runtime"
status "mode=${MODE}, model=${MODEL_PATH}, effort=${REASONING_EFFORT}, GPUs=${CUDA_DEVICES}"
status "max_model_len=${MAX_MODEL_LEN}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}, max_num_seqs=${MAX_NUM_SEQS}"
status "thermal_guard=on, pause_at_or_above=${GPU_TEMP_LIMIT_C}C, resume_at_or_below=${GPU_TEMP_RESUME_C}C, poll=${GPU_TEMP_POLL_SECONDS}s"
status "A-MEM concurrency=${AMEM_CONCURRENCY}, embedding_concurrency=${AMEM_EMBEDDING_CONCURRENCY}, response_format=${AMEM_RESPONSE_FORMAT}; Mem0 concurrency=${MEM0_CONCURRENCY}; prefix_cache=on; chunked_prefill=on"

for replica_index in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$replica_index]}"
    server_port=$((BACKEND_PORT_BASE + replica_index))
    replica_log="${LOG_DIR}/gpu${gpu_id}.server.log"
    replica_cache="${CACHE_ROOT}/gpu${gpu_id}"
    SERVER_PORTS+=("${server_port}")
    SERVER_LOGS+=("${replica_log}")
    mkdir -p "${replica_cache}/vllm" "${replica_cache}/torchinductor"
    touch "${replica_log}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    VLLM_CACHE_ROOT="${replica_cache}/vllm" \
    setsid "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --dtype bfloat16 \
        --host 127.0.0.1 \
        --port "${server_port}" \
        --tensor-parallel-size 1 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --async-scheduling \
        --reasoning-parser openai_gptoss \
        --disable-log-requests \
        --disable-uvicorn-access-log \
        --uvicorn-log-level warning \
        --trust-remote-code \
        >>"${replica_log}" 2>&1 &
    SERVER_PIDS+=("$!")
    status "Started replica ${replica_index}: GPU=${gpu_id}, port=${server_port}, pid=${SERVER_PIDS[$replica_index]}"
done

wait_for_backends

proxy_args=()
for server_port in "${SERVER_PORTS[@]}"; do
    proxy_args+=(--backend "http://127.0.0.1:${server_port}")
done
"${VLLM_PYTHON}" "${PROXY_SCRIPT}" \
    --host 127.0.0.1 \
    --port "${PROXY_PORT}" \
    "${proxy_args[@]}" \
    >>"${PROXY_LOG}" 2>&1 &
PROXY_PID=$!
wait_for_proxy

status "Warming structured decoding on every replica"
preflight_structured_outputs | tee -a "${RUN_LOG}"

nvidia-smi dmon \
    -i "${CUDA_DEVICES}" \
    -s pucvmt \
    -d 10 \
    >>"${GPU_LOG}" 2>&1 &
GPU_MONITOR_PID=$!

amem_args=(
    "${CLIENT_PYTHON}" "${REPO_ROOT}/methods/amem/build_memory_local.py"
    --input_path "${REPO_ROOT}/data/MPT_v2_0725.json"
    --output_dir "${AMEM_OUTPUT_DIR}"
    --base_url "${API_BASE}"
    --api_key EMPTY
    --model "${SERVED_MODEL_NAME}"
    --reasoning_effort "${REASONING_EFFORT}"
    --max_completion_tokens "${AMEM_MAX_COMPLETION_TOKENS}"
    --concurrency "${AMEM_CONCURRENCY}"
    --embedding_concurrency "${AMEM_EMBEDDING_CONCURRENCY}"
    --request_timeout_seconds "${REQUEST_TIMEOUT_SECONDS}"
    --retry_count "${RETRY_COUNT}"
    --response_format "${AMEM_RESPONSE_FORMAT}"
    --resume
)

mem0_args=(
    "${CLIENT_PYTHON}" "${REPO_ROOT}/methods/mem0_local/build_memory.py"
    --input_path "${REPO_ROOT}/data/MPT_v2_0725.json"
    --metrics_output "${MEM0_METRICS}"
    --vector_store_path "${MEM0_VECTOR_STORE}"
    --history_db_path "${MEM0_HISTORY_DB}"
    --base_url "${API_BASE}"
    --model "${SERVED_MODEL_NAME}"
    --reasoning_effort "${REASONING_EFFORT}"
    --disable_response_format
    --max_tokens "${MEM0_MAX_COMPLETION_TOKENS}"
    --collection_name "${MEM0_COLLECTION}"
    --snapshot_mode final
    --concurrency "${MEM0_CONCURRENCY}"
    --retry_count "${RETRY_COUNT}"
    --resume
)

if [[ "${MODE}" == "smoke" ]]; then
    amem_args+=(
        --max_examples "${SMOKE_MAX_EXAMPLES}"
        --max_sessions "${SMOKE_MAX_SESSIONS}"
    )
    mem0_args+=(
        --max_examples "${SMOKE_MAX_EXAMPLES}"
        --max_sessions "${SMOKE_MAX_SESSIONS}"
        --end_example "${SMOKE_MAX_EXAMPLES}"
    )
fi

status "Launching A-MEM and Mem0 clients concurrently"
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 \
"${amem_args[@]}" >>"${LOG_DIR}/amem.log" 2>&1 &
AMEM_PID=$!

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONUNBUFFERED=1 \
"${mem0_args[@]}" >>"${LOG_DIR}/mem0.log" 2>&1 &
MEM0_PID=$!

status "Clients started: A-MEM pid=${AMEM_PID}, Mem0 pid=${MEM0_PID}"
thermal_guard &
THERMAL_GUARD_PID=$!
status "Thermal guard started: pid=${THERMAL_GUARD_PID}"

set +e
wait "${AMEM_PID}"
amem_status=$?
AMEM_PID=""
wait "${MEM0_PID}"
mem0_status=$?
MEM0_PID=""
set -e
stop_pid "${THERMAL_GUARD_PID}" "thermal guard"
THERMAL_GUARD_PID=""

curl -fsS "http://127.0.0.1:${PROXY_PORT}/stats" \
    | "${CLIENT_PYTHON}" -m json.tool \
    | tee "${LOG_DIR}/proxy_stats.json" \
    | tee -a "${RUN_LOG}"

if (( amem_status != 0 )); then
    status "A-MEM failed with code=${amem_status}"
    tail -n 120 "${LOG_DIR}/amem.log" | tee -a "${RUN_LOG}"
fi
if (( mem0_status != 0 )); then
    status "Mem0 failed with code=${mem0_status}"
    tail -n 120 "${LOG_DIR}/mem0.log" | tee -a "${RUN_LOG}"
fi
if (( amem_status != 0 || mem0_status != 0 )); then
    exit 1
fi

status "Both memory builds completed successfully"
status "A-MEM output=${AMEM_OUTPUT_DIR}/memory.jsonl"
status "Mem0 output=${MEM0_METRICS}"
