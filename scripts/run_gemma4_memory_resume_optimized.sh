#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/tmp/minseo_gemma4_vllm_exact/bin/python}"
MODEL_NAME="${MODEL_NAME:-google/gemma-4-12B-it}"
MODEL_REVISION="${MODEL_REVISION:-12ace6d648d72bd41519e140f1185f34d38c7e3d}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-google/gemma-4-12B-it}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
DP_SIZE="${DP_SIZE:-4}"
PORT="${PORT:-8002}"
BACKEND_PORT_BASE="${BACKEND_PORT_BASE:-8200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
CONCURRENCY="${CONCURRENCY:-128}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-7200}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-5}"

HF_HOME="${HF_HOME:-/tmp/minseo_gemma4_hf}"
VLLM_CACHE_BASE="${VLLM_CACHE_BASE:-/tmp/minseo_gemma4_vllm_cache}"
MEMORY_SCRIPT="${MEMORY_SCRIPT:-/data/minseo/experiment4/ours_memory/cross_model_high_reasoning_step1.py}"
BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${SCRIPT_DIR}/bootstrap_memory_checkpoint.py}"
PROXY_SCRIPT="${PROXY_SCRIPT:-${SCRIPT_DIR}/least_loaded_openai_proxy.py}"
INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/data/MPT_v2_mix600.json}"
MEMORY_ROOT="${MEMORY_ROOT:-${REPO_ROOT}/outputs/ours_memory/cross_model_mix600_high_reasoning_20260721/gemma4_12b_it}"
SHARD_A="${SHARD_A:-${MEMORY_ROOT}/step1_shards/a/memory.jsonl}"
SHARD_B="${SHARD_B:-${MEMORY_ROOT}/step1_shards/b/memory.jsonl}"
MEMORY_OUTPUT="${MEMORY_OUTPUT:-${MEMORY_ROOT}/memory.jsonl}"

RUN_TAG="${RUN_TAG:-gemma4_memory_resume_optimized_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/runtime/gemma4_memory_resume_optimized}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_TAG}.run.log}"
PROXY_LOG="${PROXY_LOG:-${LOG_DIR}/${RUN_TAG}.proxy.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
API_BASE="http://127.0.0.1:${PORT}/v1"

mkdir -p "${LOG_DIR}" "${MEMORY_ROOT}" "${HF_HOME}" "${VLLM_CACHE_BASE}"
touch "${RUN_LOG}" "${PROXY_LOG}"
echo "$$" >"${PID_FILE}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'Gemma4 environment is missing: %s\n' "${PYTHON_BIN}" >&2
    exit 2
fi

IFS=',' read -r -a GPU_IDS <<<"${CUDA_DEVICES}"
if (( ${#GPU_IDS[@]} != DP_SIZE )); then
    printf 'Expected %d GPU IDs, got: %s\n' "${DP_SIZE}" "${CUDA_DEVICES}" >&2
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
            status "Stopping Gemma4 replica pid=${server_pid}"
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

checkpoint_count() {
    "${PYTHON_BIN}" - "${MEMORY_OUTPUT}" <<'PY'
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
            if example_id in ids:
                raise SystemExit(f"duplicate example_id: {example_id}")
            ids.add(example_id)
print(len(ids))
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
                status "Gemma4 replica ${replica_index} ready: GPU=${GPU_IDS[$replica_index]}, port=${server_port}"
                break
            fi
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                status "Gemma4 replica ${replica_index} exited during startup"
                tail -n 160 "${server_log}" | tee -a "${RUN_LOG}"
                return 1
            fi
            sleep 5
            waited=$((waited + 5))
        done
        if (( waited >= 3600 )); then
            status "Timed out waiting for Gemma4 replica ${replica_index}"
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
    return 1
}

status "Normalizing existing Gemma4 high-reasoning checkpoint"
"${PYTHON_BIN}" "${BOOTSTRAP_SCRIPT}" \
    --input "${INPUT_PATH}" \
    --output "${MEMORY_OUTPUT}" \
    --source "${SHARD_A}" \
    --source "${SHARD_B}" \
    --expected-count 600 | tee -a "${RUN_LOG}"

if [[ "$(checkpoint_count)" == "600" ]]; then
    status "Gemma4 high-reasoning memory already complete: 600/600"
    exit 0
fi

status "Starting Gemma4 optimized resume: completed=$(checkpoint_count), remaining=$((600 - $(checkpoint_count)))"
status "model=${MODEL_NAME}@${MODEL_REVISION}, GPUs=${CUDA_DEVICES}, DP=${DP_SIZE}, concurrency=${CONCURRENCY}"

for replica_index in "${!GPU_IDS[@]}"; do
    server_port=$((BACKEND_PORT_BASE + replica_index))
    replica_log="${LOG_DIR}/${RUN_TAG}.gpu${GPU_IDS[$replica_index]}.server.log"
    replica_cache="${VLLM_CACHE_BASE}/gpu${GPU_IDS[$replica_index]}"
    SERVER_PORTS+=("${server_port}")
    SERVER_LOGS+=("${replica_log}")
    mkdir -p "${replica_cache}"
    touch "${replica_log}"

    CUDA_VISIBLE_DEVICES="${GPU_IDS[$replica_index]}" \
    HF_HOME="${HF_HOME}" \
    VLLM_CACHE_ROOT="${replica_cache}" \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_NAME}" \
        --revision "${MODEL_REVISION}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --dtype bfloat16 \
        --host 127.0.0.1 \
        --port "${server_port}" \
        --tensor-parallel-size 1 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --reasoning-parser gemma4 \
        --model-impl vllm \
        --language-model-only \
        --enforce-eager \
        --trust-remote-code \
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

"${PYTHON_BIN}" - "${API_BASE}" "${SERVED_MODEL_NAME}" <<'PY' | tee -a "${RUN_LOG}"
import sys
from openai import OpenAI

api_base, model = sys.argv[1:]
client = OpenAI(api_key="EMPTY", base_url=api_base, timeout=7200, max_retries=0)
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Reply with exactly: READY"}],
    temperature=0.0,
    max_tokens=1024,
    reasoning_effort="high",
)
content = (response.choices[0].message.content or "").strip()
if not content:
    raise SystemExit("Gemma4 high-reasoning preflight returned no final content")
print(f"preflight passed: finish_reason={response.choices[0].finish_reason}")
PY

status "Resuming Gemma4 high-reasoning memory construction"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${MEMORY_SCRIPT}" run-shard \
    --input "${INPUT_PATH}" \
    --output "${MEMORY_OUTPUT}" \
    --model "${SERVED_MODEL_NAME}" \
    --api-base "${API_BASE}" \
    --api-key EMPTY \
    --num-shards 1 \
    --shard-index 0 \
    --concurrency "${CONCURRENCY}" \
    --timeout "${REQUEST_TIMEOUT_SECONDS}" \
    --client-retries "${CLIENT_MAX_RETRIES}" \
    --resume \
    2>&1 | tee -a "${RUN_LOG}"

"${PYTHON_BIN}" "${BOOTSTRAP_SCRIPT}" \
    --input "${INPUT_PATH}" \
    --output "${MEMORY_OUTPUT}" \
    --expected-count 600 | tee -a "${RUN_LOG}"

if [[ "$(checkpoint_count)" != "600" ]]; then
    status "Gemma4 memory completeness check failed: completed=$(checkpoint_count), expected=600"
    exit 1
fi
status "Gemma4 high-reasoning memory construction complete: 600/600"
