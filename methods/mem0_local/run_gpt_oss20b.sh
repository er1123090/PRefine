#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VLLM_PYTHON="${VLLM_PYTHON:-/data/minseo/.venvs/vllm/bin/python}"
MEM0_PYTHON="${MEM0_PYTHON:-/data/minseo/.venvs/experiment8/bin/python}"
MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-oss-20b}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
CONCURRENCY="${CONCURRENCY:-1}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/runtime/mem0_local}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/gpt_oss_20b_vllm.log}"
API_BASE="http://127.0.0.1:${PORT}/v1"

mkdir -p "${LOG_DIR}"
touch "${SERVER_LOG}"

server_pid=""
cleanup() {
    local exit_code=$?
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
"${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --reasoning-parser openai_gptoss \
    --trust-remote-code \
    >>"${SERVER_LOG}" 2>&1 &
server_pid=$!

for _ in $(seq 1 720); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        tail -n 120 "${SERVER_LOG}"
        exit 1
    fi
    sleep 5
done

if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    tail -n 120 "${SERVER_LOG}"
    exit 1
fi

"${MEM0_PYTHON}" "${SCRIPT_DIR}/build_memory.py" \
    --input_path "${REPO_ROOT}/data/MPT_v2_0725.json" \
    --metrics_output "${REPO_ROOT}/outputs/mem0_local/MPT_v2_0725_construction.jsonl" \
    --vector_store_path "${REPO_ROOT}/outputs/mem0_local/MPT_v2_0725_qdrant" \
    --history_db_path "${REPO_ROOT}/outputs/mem0_local/MPT_v2_0725_history.sqlite" \
    --base_url "${API_BASE}" \
    --model "${SERVED_MODEL_NAME}" \
    --reasoning_effort low \
    --snapshot_mode all \
    --concurrency "${CONCURRENCY}" \
    --resume \
    "$@"
