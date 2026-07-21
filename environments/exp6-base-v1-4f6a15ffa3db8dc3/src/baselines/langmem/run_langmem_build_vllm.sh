#!/bin/bash

set -euo pipefail

PYTHON_SCRIPT="/data/minseo/experiments6/langmem/step1_build_memory.py"
INPUT_DATA="${INPUT_DATA:-/data/minseo/experiments6/data/1229_dev_6.json}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/data/minseo/experiments6/langmem/memory_snapshots/semantic-custom}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a RAW_GPU_ID_LIST <<< "${GPU_IDS}"
GPU_ID_LIST=()
for gpu_id in "${RAW_GPU_ID_LIST[@]}"; do
    gpu_id="${gpu_id// /}"
    [ -n "${gpu_id}" ] || continue
    GPU_ID_LIST+=("${gpu_id}")
done

GPU_COUNT="${#GPU_ID_LIST[@]}"
TP_SIZE="${TP_SIZE:-${GPU_COUNT}}"
PORT="${PORT:-8002}"
VLLM_URL="http://localhost:${PORT}/v1"
CONCURRENCY="${CONCURRENCY:-50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-${OPENAI_API_KEY:-}}"
LOCAL_EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL:-intfloat/e5-small-v2}"
TOOL_CALL_PARSER_OVERRIDE="${TOOL_CALL_PARSER_OVERRIDE:-}"
#
# LangMem build presets:
# - auto: keep the existing auto backend routing
# - structured_output: force structured-output extraction
# - langmem_custom_semantic: LangMem backend + custom prompt + structured semantic memories
# - langmem_default_string: LangMem backend + default LangMem prompt + plain string memories
#
# Edit this value in the file for the default behavior, or override with env vars.
LANGMEM_BUILD_MODE="${LANGMEM_BUILD_MODE:-structured_output}"
case "${LANGMEM_BUILD_MODE}" in
    auto)
        DEFAULT_MEMORY_BACKEND="auto"
        DEFAULT_LANGMEM_MEMORY_SCHEMA="semantic"
        DEFAULT_LANGMEM_PROMPT_STYLE="custom"
        ;;
    structured_output)
        DEFAULT_MEMORY_BACKEND="structured_output"
        DEFAULT_LANGMEM_MEMORY_SCHEMA="semantic"
        DEFAULT_LANGMEM_PROMPT_STYLE="custom"
        ;;
    langmem_custom_semantic)
        DEFAULT_MEMORY_BACKEND="langmem"
        DEFAULT_LANGMEM_MEMORY_SCHEMA="semantic"
        DEFAULT_LANGMEM_PROMPT_STYLE="custom"
        ;;
    langmem_default_string)
        DEFAULT_MEMORY_BACKEND="langmem"
        DEFAULT_LANGMEM_MEMORY_SCHEMA="string"
        DEFAULT_LANGMEM_PROMPT_STYLE="default"
        ;;
    *)
        echo "[ERROR] Unsupported LANGMEM_BUILD_MODE='${LANGMEM_BUILD_MODE}'."
        echo "        Choose one of: auto, structured_output, langmem_custom_semantic, langmem_default_string"
        exit 1
        ;;
esac

MEMORY_BACKEND="${MEMORY_BACKEND:-${DEFAULT_MEMORY_BACKEND}}"
LANGMEM_MEMORY_SCHEMA="${LANGMEM_MEMORY_SCHEMA:-${DEFAULT_LANGMEM_MEMORY_SCHEMA}}"
LANGMEM_PROMPT_STYLE="${LANGMEM_PROMPT_STYLE:-${DEFAULT_LANGMEM_PROMPT_STYLE}}"
START_EXAMPLE="${START_EXAMPLE:-}"
END_EXAMPLE="${END_EXAMPLE:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

if [ "${GPU_COUNT}" -eq 0 ]; then
    echo "[ERROR] GPU_IDS is empty. Set GPU_IDS to at least one visible GPU id."
    exit 1
fi

if [ "${TP_SIZE}" -gt "${GPU_COUNT}" ]; then
    echo "[ERROR] TP_SIZE=${TP_SIZE} is larger than the number of visible GPUs (${GPU_COUNT}) from GPU_IDS='${GPU_IDS}'."
    echo "        Set TP_SIZE<=${GPU_COUNT}, or adjust GPU_IDS."
    exit 1
fi

MODELS=(
    #"meta-llama/Llama-3.1-8B-Instruct"
    #"Qwen/Qwen3-8B"
    #"google/gemma-3-12b-it"
    #"deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

if [ -n "${TARGET_MODELS:-}" ]; then
    IFS='|' read -r -a MODELS <<< "${TARGET_MODELS}"
fi

trap cleanup SIGINT SIGTERM ERR

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo ""
        echo "[WARN] Killing vLLM Server (PID: ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    exit 1
}

wait_for_server() {
    local server_log_path="$1"
    echo "Waiting for vLLM server at ${VLLM_URL}..."
    local max_retries=60
    local count=0

    while ! curl -fsS --max-time 5 "${VLLM_URL}/models" > /dev/null; do
        sleep 5
        echo -n "."
        count=$((count + 1))

        if ! ps -p "${SERVER_PID}" > /dev/null; then
            echo ""
            echo "[ERROR] vLLM Server process died unexpectedly."
            cat "${server_log_path}"
            exit 1
        fi

        if [ "${count}" -ge "${max_retries}" ]; then
            echo ""
            echo "[ERROR] Timeout waiting for vLLM server."
            kill "${SERVER_PID}" 2>/dev/null || true
            exit 1
        fi
    done
    echo ""
    echo ">> Server is READY!"
}

echo "========================================================"
echo "LangMem VLLM Memory Build Started at $(date)"
echo "GPU: ${GPU_IDS} | TP Size: ${TP_SIZE}"
echo "Embedding model: ${EMBEDDING_MODEL}"
if [ -n "${EMBEDDING_BASE_URL}" ]; then
    echo "Embedding base URL: ${EMBEDDING_BASE_URL}"
else
    echo "Embedding base URL: <default OpenAI endpoint>"
fi
echo "LangMem build mode: ${LANGMEM_BUILD_MODE}"
echo "Memory backend: ${MEMORY_BACKEND}"
echo "LangMem memory schema: ${LANGMEM_MEMORY_SCHEMA}"
echo "LangMem prompt style: ${LANGMEM_PROMPT_STYLE}"
echo "========================================================"

if [ -z "${EMBEDDING_BASE_URL}" ] && [ -z "${EMBEDDING_API_KEY}" ]; then
    if [[ "${EMBEDDING_MODEL}" == text-embedding-* ]]; then
        EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL}"
        echo "[INFO] No embedding credentials found. Falling back to local embedding model: ${EMBEDDING_MODEL}"
    else
        echo "[INFO] No embedding credentials found. Using local embedding model: ${EMBEDDING_MODEL}"
    fi
fi

mkdir -p "${BASE_OUTPUT_DIR}"

for model in "${MODELS[@]}"; do
    DATE_TAG="$(date +%Y%m%d_%H%M%S)"
    MODEL_SAFE_NAME="${model//\//_}"
    MODEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_SAFE_NAME}"
    mkdir -p "${MODEL_OUTPUT_DIR}"

    SNAPSHOT_PATH="${MODEL_OUTPUT_DIR}/langmem_1229_dev_6.jsonl"
    MANIFEST_PATH="${MODEL_OUTPUT_DIR}/langmem_1229_dev_6.manifest.json"
    RUN_LOG_PATH="${MODEL_OUTPUT_DIR}/langmem_1229_dev_6_${DATE_TAG}.run.log"
    SERVER_LOG_PATH="${MODEL_OUTPUT_DIR}/langmem_1229_dev_6_${DATE_TAG}.vllm_server.log"

    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: ${model}"
    echo "####################################################################"

    TOOL_CALL_PARSER="${TOOL_CALL_PARSER_OVERRIDE}"
    if [ -z "${TOOL_CALL_PARSER}" ]; then
        TOOL_CALL_PARSER="hermes"
        if [[ "${model}" == *"Llama-3"* ]]; then
            TOOL_CALL_PARSER="llama3_json"
        elif [[ "${model}" == *"Mistral"* ]]; then
            TOOL_CALL_PARSER="mistral"
        elif [[ "${model}" == *"Qwen3"* ]]; then
            TOOL_CALL_PARSER="qwen3_xml"
        fi
    fi
    echo "Tool parser: ${TOOL_CALL_PARSER}"

    CUDA_VISIBLE_DEVICES="${GPU_IDS}" nohup vllm serve "${model}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --enable-auto-tool-choice \
        --tool-call-parser "${TOOL_CALL_PARSER}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --trust-remote-code > "${SERVER_LOG_PATH}" 2>&1 &

    SERVER_PID=$!
    echo ">> vLLM Server PID: ${SERVER_PID}"

    wait_for_server "${SERVER_LOG_PATH}"

    echo "[STEP 2] Running LangMem Build..."
    echo "Saving snapshot to: ${SNAPSHOT_PATH}"

    CMD=(
        python "${PYTHON_SCRIPT}"
        --input_path "${INPUT_DATA}"
        --output_path "${SNAPSHOT_PATH}"
        --manifest_path "${MANIFEST_PATH}"
        --run_log_path "${RUN_LOG_PATH}"
        --memory_model "${model}"
        --embedding_model "${EMBEDDING_MODEL}"
        --base_url "${VLLM_URL}"
        --api_key "EMPTY"
        --concurrency "${CONCURRENCY}"
        --memory_backend "${MEMORY_BACKEND}"
        --langmem_memory_schema "${LANGMEM_MEMORY_SCHEMA}"
        --langmem_prompt_style "${LANGMEM_PROMPT_STYLE}"
    )

    # Always pass embedding_base_url explicitly.
    # When it is an empty string, step1_build_memory.py treats that as
    # "use the default OpenAI endpoint" instead of falling back to the vLLM chat URL.
    CMD+=(--embedding_base_url "${EMBEDDING_BASE_URL}")
    if [ -n "${EMBEDDING_API_KEY}" ]; then
        CMD+=(--embedding_api_key "${EMBEDDING_API_KEY}")
    fi
    if [ -n "${START_EXAMPLE}" ]; then
        CMD+=(--start_example "${START_EXAMPLE}")
    fi
    if [ -n "${END_EXAMPLE}" ]; then
        CMD+=(--end_example "${END_EXAMPLE}")
    fi
    if [ -n "${MAX_EXAMPLES}" ]; then
        CMD+=(--max_examples "${MAX_EXAMPLES}")
    fi

    "${CMD[@]}"

    echo "[STEP 3] Stopping vLLM Server..."
    kill "${SERVER_PID}"
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""

    echo ">> Server Stopped."
    echo "--------------------------------------------------------"
    sleep 10
done

echo "========================================================"
echo "All LangMem VLLM Memory Builds Finished at $(date)"
echo "========================================================"
