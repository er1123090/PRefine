#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/data/MPT_v2_0725.json}"
MEMORY_ROOT="${MEMORY_ROOT:-${REPO_ROOT}/outputs/mpt_v2_0725_hint_no_easy_conflict/memory}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/outputs/runtime/mpt0725_conflict_memory}"
BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${SCRIPT_DIR}/bootstrap_memory_checkpoint.py}"
VERIFY_SCRIPT="${VERIFY_SCRIPT:-${SCRIPT_DIR}/verify_mpt0725_memory.py}"
PROXY_SCRIPT="${PROXY_SCRIPT:-${SCRIPT_DIR}/least_loaded_openai_proxy.py}"
CROSS_MEMORY_SCRIPT="${CROSS_MEMORY_SCRIPT:-/data/minseo/experiment4/ours_memory/cross_model_high_reasoning_step1.py}"
GPT_MEMORY_SCRIPT="${GPT_MEMORY_SCRIPT:-/data/minseo/experiment4/ours_memory/Preference_Memory_step1_LATENTPREF_VLLM.py}"

RUN_TAG="${RUN_TAG:-mpt0725_conflict_memory_$(date '+%Y%m%d_%H%M%S')}"
RUN_DIR="${RUNTIME_ROOT}/${RUN_TAG}"
RUN_LOG="${RUN_DIR}/run.log"
PID_FILE="${RUN_DIR}/queue.pid"
PROXY_PORT="${PROXY_PORT:-8002}"
BACKEND_PORT_BASE="${BACKEND_PORT_BASE:-8100}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-7200}"
CLIENT_MAX_RETRIES="${CLIENT_MAX_RETRIES:-5}"
VLLM_CACHE_BASE="${VLLM_CACHE_BASE:-/tmp/minseo_mpt0725_conflict_vllm_cache}"

mkdir -p "${RUN_DIR}" "${VLLM_CACHE_BASE}"
touch "${RUN_LOG}"
echo "$$" >"${PID_FILE}"

declare -a SERVER_PIDS=()
declare -a SERVER_PORTS=()
PROXY_PID=""
ACTIVE_TARGET=""

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN_LOG}"
}

stop_runtime() {
    local exit_code=$?

    if [[ -n "${PROXY_PID}" ]] && kill -0 "${PROXY_PID}" 2>/dev/null; then
        status "Stopping ${ACTIVE_TARGET} proxy pid=${PROXY_PID}"
        kill -- "-${PROXY_PID}" 2>/dev/null || kill "${PROXY_PID}" 2>/dev/null || true
        wait "${PROXY_PID}" 2>/dev/null || true
    fi
    PROXY_PID=""

    local server_pid
    for server_pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "${server_pid}" 2>/dev/null; then
            status "Stopping ${ACTIVE_TARGET} vLLM process group pid=${server_pid}"
            kill -- "-${server_pid}" 2>/dev/null || kill "${server_pid}" 2>/dev/null || true
        fi
    done
    for server_pid in "${SERVER_PIDS[@]}"; do
        wait "${server_pid}" 2>/dev/null || true
    done
    SERVER_PIDS=()
    SERVER_PORTS=()

    return "${exit_code}"
}

cleanup() {
    local exit_code=$?
    stop_runtime || true
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
        local server_log="${RUN_DIR}/${ACTIVE_TARGET}.replica${replica_index}.server.log"
        while (( waited < 3600 )); do
            if curl -fsS "http://127.0.0.1:${server_port}/health" >/dev/null 2>&1; then
                status "${ACTIVE_TARGET} replica ${replica_index} ready on port ${server_port}"
                break
            fi
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                status "${ACTIVE_TARGET} replica ${replica_index} exited during startup"
                tail -n 160 "${server_log}" | tee -a "${RUN_LOG}"
                return 1
            fi
            sleep 5
            waited=$((waited + 5))
        done
        if (( waited >= 3600 )); then
            status "${ACTIVE_TARGET} replica ${replica_index} startup timed out"
            tail -n 160 "${server_log}" | tee -a "${RUN_LOG}"
            return 1
        fi
    done
}

wait_for_proxy() {
    local waited=0
    local proxy_log="${RUN_DIR}/${ACTIVE_TARGET}.proxy.log"
    while (( waited < 60 )); do
        if curl -fsS "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
            status "${ACTIVE_TARGET} load-balancing proxy ready"
            return 0
        fi
        if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
            status "${ACTIVE_TARGET} proxy exited during startup"
            tail -n 100 "${proxy_log}" | tee -a "${RUN_LOG}"
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    status "${ACTIVE_TARGET} proxy startup timed out"
    return 1
}

start_proxy() {
    local python_bin="$1"
    local proxy_log="${RUN_DIR}/${ACTIVE_TARGET}.proxy.log"
    local proxy_args=()
    local server_port
    for server_port in "${SERVER_PORTS[@]}"; do
        proxy_args+=(--backend "http://127.0.0.1:${server_port}")
    done

    setsid "${python_bin}" "${PROXY_SCRIPT}" \
        --host 127.0.0.1 \
        --port "${PROXY_PORT}" \
        "${proxy_args[@]}" \
        >>"${proxy_log}" 2>&1 &
    PROXY_PID=$!
    status "Started ${ACTIVE_TARGET} proxy pid=${PROXY_PID}"
    wait_for_proxy
}

seed_checkpoint() {
    local python_bin="$1"
    local target="$2"
    local target_root="${MEMORY_ROOT}/${target}"
    mkdir -p "${target_root}"
    "${python_bin}" "${BOOTSTRAP_SCRIPT}" \
        --input "${INPUT_PATH}" \
        --output "${target_root}/_memory.jsonl" \
        --source "${target_root}/memory.jsonl" \
        --expected-count 459 | tee -a "${RUN_LOG}"
}

verify_checkpoint() {
    local python_bin="$1"
    local target="$2"
    local target_root="${MEMORY_ROOT}/${target}"
    "${python_bin}" "${BOOTSTRAP_SCRIPT}" \
        --input "${INPUT_PATH}" \
        --output "${target_root}/_memory.jsonl" \
        --source "${target_root}/memory.jsonl" \
        --expected-count 459 | tee -a "${RUN_LOG}"
    "${python_bin}" "${VERIFY_SCRIPT}" \
        --input "${INPUT_PATH}" \
        --memory "${target_root}/_memory.jsonl" \
        --reused-source "${target_root}/memory.jsonl" \
        --normalize | tee -a "${RUN_LOG}"
}

preflight() {
    local python_bin="$1"
    local served_model="$2"
    local target="$3"
    "${python_bin}" - "${PROXY_PORT}" "${served_model}" "${target}" <<'PY' | tee -a "${RUN_LOG}"
import sys
from openai import OpenAI

port, model, target = sys.argv[1:]
client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://127.0.0.1:{port}/v1",
    timeout=7200,
    max_retries=0,
)
kwargs = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0.0,
    "max_tokens": 4096,
}
if target == "qwen3_8b":
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
else:
    kwargs["reasoning_effort"] = "high"
response = client.chat.completions.create(**kwargs)
content = (response.choices[0].message.content or "").strip()
if not content:
    raise SystemExit(f"{target} high-reasoning preflight returned no final content")
print(
    f"{target} high-reasoning preflight passed: "
    f"finish_reason={response.choices[0].finish_reason}"
)
PY
}

run_qwen() {
    ACTIVE_TARGET="qwen3_8b"
    local python_bin="/data/minseo/.venvs/vllm/bin/python"
    local model_path="/data/minseo/.cache/huggingface/Qwen3-32B"
    local served_model="Qwen/Qwen3-32B"
    local groups=("0,1" "2,3")
    local replica_index

    seed_checkpoint "${python_bin}" "${ACTIVE_TARGET}"
    status "Starting Qwen3-32B high-thinking memory: TP2 x DP2, conflict pending=160"

    for replica_index in "${!groups[@]}"; do
        local server_port=$((BACKEND_PORT_BASE + replica_index))
        local server_log="${RUN_DIR}/${ACTIVE_TARGET}.replica${replica_index}.server.log"
        local replica_cache="${VLLM_CACHE_BASE}/${ACTIVE_TARGET}/replica${replica_index}"
        mkdir -p "${replica_cache}"
        SERVER_PORTS+=("${server_port}")

        PATH="${python_bin%/*}:${PATH}" \
        CUDA_VISIBLE_DEVICES="${groups[$replica_index]}" \
        VLLM_CACHE_ROOT="${replica_cache}" \
        setsid "${python_bin}" -m vllm.entrypoints.openai.api_server \
            --model "${model_path}" \
            --served-model-name "${served_model}" \
            --dtype bfloat16 \
            --host 127.0.0.1 \
            --port "${server_port}" \
            --tensor-parallel-size 2 \
            --max-model-len 16384 \
            --gpu-memory-utilization 0.92 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 32 \
            --reasoning-parser qwen3 \
            --trust-remote-code \
            --enforce-eager \
            --async-scheduling \
            >>"${server_log}" 2>&1 &
        SERVER_PIDS+=("$!")
        status "Started Qwen replica ${replica_index}: GPUs=${groups[$replica_index]}, pid=${SERVER_PIDS[$replica_index]}"
    done

    wait_for_backends
    start_proxy "${python_bin}"
    preflight "${python_bin}" "${served_model}" "${ACTIVE_TARGET}"

    PYTHONUNBUFFERED=1 "${python_bin}" "${CROSS_MEMORY_SCRIPT}" run-shard \
        --input "${INPUT_PATH}" \
        --output "${MEMORY_ROOT}/${ACTIVE_TARGET}/_memory.jsonl" \
        --model "${served_model}" \
        --api-base "http://127.0.0.1:${PROXY_PORT}/v1" \
        --api-key EMPTY \
        --num-shards 1 \
        --shard-index 0 \
        --concurrency 64 \
        --timeout "${REQUEST_TIMEOUT_SECONDS}" \
        --client-retries "${CLIENT_MAX_RETRIES}" \
        --resume \
        2>&1 | tee -a "${RUN_LOG}"

    verify_checkpoint "${python_bin}" "${ACTIVE_TARGET}"
    stop_runtime
    status "Qwen memory complete"
}

run_gemma() {
    ACTIVE_TARGET="gemma4_12b_it"
    local python_bin="/tmp/minseo_gemma4_vllm_exact/bin/python"
    local model_path="/tmp/minseo_gemma4_hf/hub/models--google--gemma-4-12B-it/snapshots/12ace6d648d72bd41519e140f1185f34d38c7e3d"
    local served_model="google/gemma-4-12B-it"
    local groups=("0" "1" "2" "3")
    local replica_index

    seed_checkpoint "${python_bin}" "${ACTIVE_TARGET}"
    status "Starting Gemma4-12B high-reasoning memory: TP1 x DP4, conflict pending=160"

    for replica_index in "${!groups[@]}"; do
        local server_port=$((BACKEND_PORT_BASE + replica_index))
        local server_log="${RUN_DIR}/${ACTIVE_TARGET}.replica${replica_index}.server.log"
        local replica_cache="${VLLM_CACHE_BASE}/${ACTIVE_TARGET}/replica${replica_index}"
        mkdir -p "${replica_cache}"
        SERVER_PORTS+=("${server_port}")

        PATH="${python_bin%/*}:${PATH}" \
        CUDA_VISIBLE_DEVICES="${groups[$replica_index]}" \
        VLLM_CACHE_ROOT="${replica_cache}" \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        setsid "${python_bin}" -m vllm.entrypoints.openai.api_server \
            --model "${model_path}" \
            --served-model-name "${served_model}" \
            --dtype bfloat16 \
            --host 127.0.0.1 \
            --port "${server_port}" \
            --tensor-parallel-size 1 \
            --max-model-len 16384 \
            --gpu-memory-utilization 0.92 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 32 \
            --reasoning-parser gemma4 \
            --model-impl vllm \
            --language-model-only \
            --trust-remote-code \
            --enforce-eager \
            >>"${server_log}" 2>&1 &
        SERVER_PIDS+=("$!")
        status "Started Gemma replica ${replica_index}: GPU=${groups[$replica_index]}, pid=${SERVER_PIDS[$replica_index]}"
    done

    wait_for_backends
    start_proxy "${python_bin}"
    preflight "${python_bin}" "${served_model}" "${ACTIVE_TARGET}"

    PYTHONUNBUFFERED=1 "${python_bin}" "${CROSS_MEMORY_SCRIPT}" run-shard \
        --input "${INPUT_PATH}" \
        --output "${MEMORY_ROOT}/${ACTIVE_TARGET}/_memory.jsonl" \
        --model "${served_model}" \
        --api-base "http://127.0.0.1:${PROXY_PORT}/v1" \
        --api-key EMPTY \
        --num-shards 1 \
        --shard-index 0 \
        --concurrency 128 \
        --timeout "${REQUEST_TIMEOUT_SECONDS}" \
        --client-retries "${CLIENT_MAX_RETRIES}" \
        --resume \
        2>&1 | tee -a "${RUN_LOG}"

    verify_checkpoint "${python_bin}" "${ACTIVE_TARGET}"
    stop_runtime
    status "Gemma memory complete"
}

run_gpt() {
    ACTIVE_TARGET="gpt_oss_20b"
    local python_bin="/data/minseo/.venvs/vllm/bin/python"
    local model_path="/data/minseo/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    local served_model="gpt-oss-20b"
    local groups=("0" "1" "2" "3")
    local replica_index

    seed_checkpoint "${python_bin}" "${ACTIVE_TARGET}"
    status "Starting GPT-OSS-20B high-reasoning memory: TP1 x DP4, conflict pending=160"

    for replica_index in "${!groups[@]}"; do
        local server_port=$((BACKEND_PORT_BASE + replica_index))
        local server_log="${RUN_DIR}/${ACTIVE_TARGET}.replica${replica_index}.server.log"
        local replica_cache="${VLLM_CACHE_BASE}/${ACTIVE_TARGET}/replica${replica_index}"
        mkdir -p "${replica_cache}"
        SERVER_PORTS+=("${server_port}")

        CUDA_VISIBLE_DEVICES="${groups[$replica_index]}" \
        VLLM_CACHE_ROOT="${replica_cache}" \
        setsid "${python_bin}" -m vllm.entrypoints.openai.api_server \
            --model "${model_path}" \
            --served-model-name "${served_model}" \
            --dtype bfloat16 \
            --host 127.0.0.1 \
            --port "${server_port}" \
            --tensor-parallel-size 1 \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.92 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 64 \
            --reasoning-parser openai_gptoss \
            --trust-remote-code \
            --async-scheduling \
            >>"${server_log}" 2>&1 &
        SERVER_PIDS+=("$!")
        status "Started GPT replica ${replica_index}: GPU=${groups[$replica_index]}, pid=${SERVER_PIDS[$replica_index]}"
    done

    wait_for_backends
    start_proxy "${python_bin}"
    preflight "${python_bin}" "${served_model}" "${ACTIVE_TARGET}"

    PYTHONUNBUFFERED=1 "${python_bin}" "${GPT_MEMORY_SCRIPT}" \
        --input "${INPUT_PATH}" \
        --output "${MEMORY_ROOT}/${ACTIVE_TARGET}/_memory.jsonl" \
        --verifier_output "${MEMORY_ROOT}/${ACTIVE_TARGET}/_verifier_logs.jsonl" \
        --refinement_output "${MEMORY_ROOT}/${ACTIVE_TARGET}/_refinement_logs.jsonl" \
        --model "${served_model}" \
        --api_base "http://127.0.0.1:${PROXY_PORT}/v1" \
        --api_key EMPTY \
        --concurrency 256 \
        --reasoning_effort high \
        --generation_max_tokens 8192 \
        --verification_max_tokens 4096 \
        --request_timeout_seconds "${REQUEST_TIMEOUT_SECONDS}" \
        --client_max_retries 0 \
        --disable_response_format \
        --resume \
        2>&1 | tee -a "${RUN_LOG}"

    verify_checkpoint "${python_bin}" "${ACTIVE_TARGET}"
    stop_runtime
    status "GPT memory complete"
}

main() {
    local requested="${1:-all}"
    status "MPT_v2_0725 conflict memory queue started: requested=${requested}"
    status "Input=${INPUT_PATH}; reusable mix=299; new conflict_noise=80; new conflict_ordered=80"

    case "${requested}" in
        all)
            run_qwen
            run_gemma
            run_gpt
            ;;
        qwen|qwen3_8b)
            run_qwen
            ;;
        gemma|gemma4_12b_it)
            run_gemma
            ;;
        gpt|gpt_oss_20b)
            run_gpt
            ;;
        remaining|gemma-gpt)
            run_gemma
            run_gpt
            ;;
        *)
            printf 'Usage: %s [all|qwen|gemma|gpt|remaining]\n' "$0" >&2
            return 2
            ;;
    esac

    status "MPT_v2_0725 conflict memory queue complete: requested=${requested}"
}

main "$@"
