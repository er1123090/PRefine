#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VLLM_PYTHON="${VLLM_PYTHON:-/data/minseo/.venvs/vllm/bin/python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-/data/minseo/.venvs/experiment8/bin/python}"
MODEL_REVISION="${MODEL_REVISION:-6cee5e81ee83917806bbde320786a8fb61efebee}"
MODEL_PATH="${MODEL_PATH:-/data/minseo/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/${MODEL_REVISION}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-oss-20b}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
DP_SIZE="${DP_SIZE:-4}"
PROXY_PORT="${PROXY_PORT:-8004}"
BACKEND_PORT_BASE="${BACKEND_PORT_BASE:-8300}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
CONCURRENCY="${CONCURRENCY:-256}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_RETRY_TOKENS="${MAX_RETRY_TOKENS:-2048}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-7200}"

GPU_TEMP_LIMIT_C="${GPU_TEMP_LIMIT_C:-85}"
GPU_TEMP_RESUME_C="${GPU_TEMP_RESUME_C:-70}"
GPU_TEMP_POLL_SECONDS="${GPU_TEMP_POLL_SECONDS:-5}"

PREPARED_ROOT="${PREPARED_ROOT:-${REPO_ROOT}/outputs/mpt_v2_0725_hint_no_easy_conflict}"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-${PREPARED_ROOT}}"
AMEM_PREPARED_DIR="${AMEM_PREPARED_DIR:-${PREPARED_ROOT}/amem/openai_gpt-5-mini_minimal_batch}"
LANGMEM_PREPARED_DIR="${LANGMEM_PREPARED_DIR:-${PREPARED_ROOT}/langmem/openai_gpt-5-mini_minimal_batch}"
RAG_PREPARED_DIR="${RAG_PREPARED_DIR:-${PREPARED_ROOT}/rag/openai_gpt-5-mini_minimal_batch}"
AMEM_OUTPUT_DIR="${AMEM_OUTPUT_DIR:-${LOCAL_OUTPUT_ROOT}/amem/local_gpt-oss-20b_low}"
LANGMEM_OUTPUT_DIR="${LANGMEM_OUTPUT_DIR:-${LOCAL_OUTPUT_ROOT}/langmem/local_gpt-oss-20b_low}"
RAG_OUTPUT_DIR="${RAG_OUTPUT_DIR:-${LOCAL_OUTPUT_ROOT}/rag/local_gpt-oss-20b_low}"

RUN_TAG="${RUN_TAG:-memory_artifact_gpt_oss20b_dp4_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/runtime/memory_artifact_gpt_oss20b/${RUN_TAG}}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/experiment8_memory_artifact_gpt_oss20b_cache}"
RUN_LOG="${LOG_DIR}/run.log"
PROXY_LOG="${LOG_DIR}/proxy.log"
GPU_LOG="${LOG_DIR}/gpu_dmon.log"
PID_FILE="${LOG_DIR}/runner.pid"
API_BASE="http://127.0.0.1:${PROXY_PORT}/v1"

mkdir -p "${LOG_DIR}" "${CACHE_ROOT}"
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
    printf 'Thermal guard values must be integer temperatures and a positive poll interval\n' >&2
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
ACTIVE_CLIENT_PID=""

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" \
        | tee -a "${RUN_LOG}"
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
    signal_pid_if_running STOP "${ACTIVE_CLIENT_PID}"
    signal_server_groups STOP
}

resume_workload() {
    signal_server_groups CONT
    signal_pid_if_running CONT "${ACTIVE_CLIENT_PID}"
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
            status "Thermal guard pausing: max_gpu_temp=${max_temperature}C >= ${GPU_TEMP_LIMIT_C}C"
            pause_workload
            paused=1
        elif (( paused == 1 && max_temperature <= GPU_TEMP_RESUME_C )); then
            resume_workload
            status "Thermal guard resumed: max_gpu_temp=${max_temperature}C <= ${GPU_TEMP_RESUME_C}C"
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
    stop_pid "${ACTIVE_CLIENT_PID}" "inference client"
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

preflight_manifests() {
    "${CLIENT_PYTHON}" - \
        "${AMEM_PREPARED_DIR}" \
        "${LANGMEM_PREPARED_DIR}" \
        "${RAG_PREPARED_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

for raw_directory in sys.argv[1:]:
    directory = Path(raw_directory)
    manifest = directory / "manifest.jsonl"
    summary_path = directory / "prepare_summary.json"
    if not manifest.is_file() or not summary_path.is_file():
        raise SystemExit(f"Missing prepared manifest or summary: {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    count = 0
    with manifest.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                count += 1
    if count != 4695:
        raise SystemExit(f"Expected 4695 manifest rows in {directory}, got {count}")
    if summary.get("manifest_sha256") != digest.hexdigest():
        raise SystemExit(f"Manifest hash mismatch: {directory}")
    print(f"manifest ready: {directory} rows={count} sha256={digest.hexdigest()}")
PY
}

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

preflight_reasoning() {
    "${CLIENT_PYTHON}" - "${API_BASE}" "${SERVED_MODEL_NAME}" "${REASONING_EFFORT}" <<'PY'
import sys
from openai import OpenAI

base_url, model, effort = sys.argv[1:]
client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=600.0, max_retries=0)
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Reply with exactly: READY"}],
    reasoning_effort=effort,
    max_tokens=256,
)
content = (response.choices[0].message.content or "").strip()
if not content:
    raise SystemExit("Reasoning preflight returned no final content")
print(f"reasoning preflight passed: effort={effort}, output={content!r}")
PY
}

run_method() {
    local method="$1"
    local prepared_dir="$2"
    local output_dir="$3"
    local method_log="${LOG_DIR}/${method}.log"
    status "Launching ${method} inference: prepared=${prepared_dir}, output=${output_dir}"
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    "${CLIENT_PYTHON}" "${SCRIPT_DIR}/run_memory_artifact_inference.py" local \
        --method "${method}" \
        --prepared-dir "${prepared_dir}" \
        --output-dir "${output_dir}" \
        --model "${SERVED_MODEL_NAME}" \
        --record-model-name "openai/gpt-oss-20b" \
        --api-base "${API_BASE}" \
        --api-key EMPTY \
        --reasoning-effort "${REASONING_EFFORT}" \
        --expected-count 4695 \
        --concurrency "${CONCURRENCY}" \
        --max-tokens "${MAX_TOKENS}" \
        --max-retry-tokens "${MAX_RETRY_TOKENS}" \
        --retry-rounds 2 \
        --request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}" \
        --client-max-retries 0 \
        --resume \
        >>"${method_log}" 2>&1 &
    ACTIVE_CLIENT_PID=$!
    set +e
    wait "${ACTIVE_CLIENT_PID}"
    local method_status=$?
    set -e
    ACTIVE_CLIENT_PID=""
    if (( method_status != 0 )); then
        status "${method} inference failed with code=${method_status}"
        tail -n 160 "${method_log}" | tee -a "${RUN_LOG}"
        return "${method_status}"
    fi
    status "${method} inference completed"
}

preflight_manifests | tee -a "${RUN_LOG}"
status "Starting GPT-OSS-20B DP4 artifact inference runtime"
status "model=${MODEL_PATH}, effort=${REASONING_EFFORT}, GPUs=${CUDA_DEVICES}, concurrency=${CONCURRENCY}"
status "thermal_guard=on, pause_at_or_above=${GPU_TEMP_LIMIT_C}C, resume_at_or_below=${GPU_TEMP_RESUME_C}C"

for replica_index in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$replica_index]}"
    server_port=$((BACKEND_PORT_BASE + replica_index))
    replica_log="${LOG_DIR}/gpu${gpu_id}.server.log"
    replica_cache="${CACHE_ROOT}/gpu${gpu_id}"
    SERVER_PORTS+=("${server_port}")
    SERVER_LOGS+=("${replica_log}")
    mkdir -p "${replica_cache}/vllm"
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
"${VLLM_PYTHON}" "${SCRIPT_DIR}/least_loaded_openai_proxy.py" \
    --host 127.0.0.1 \
    --port "${PROXY_PORT}" \
    "${proxy_args[@]}" \
    >>"${PROXY_LOG}" 2>&1 &
PROXY_PID=$!
wait_for_proxy
preflight_reasoning | tee -a "${RUN_LOG}"

nvidia-smi dmon -i "${CUDA_DEVICES}" -s pucvmt -d 10 \
    >>"${GPU_LOG}" 2>&1 &
GPU_MONITOR_PID=$!
thermal_guard &
THERMAL_GUARD_PID=$!
status "Thermal guard started: pid=${THERMAL_GUARD_PID}"

run_method amem "${AMEM_PREPARED_DIR}" "${AMEM_OUTPUT_DIR}"
run_method langmem "${LANGMEM_PREPARED_DIR}" "${LANGMEM_OUTPUT_DIR}"
run_method rag "${RAG_PREPARED_DIR}" "${RAG_OUTPUT_DIR}"

curl -fsS "http://127.0.0.1:${PROXY_PORT}/stats" \
    | "${CLIENT_PYTHON}" -m json.tool \
    | tee "${LOG_DIR}/proxy_stats.json" \
    | tee -a "${RUN_LOG}"
status "All GPT-OSS-20B artifact inference runs completed"
