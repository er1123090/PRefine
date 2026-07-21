#!/bin/bash

set -euo pipefail

PYTHON_SCRIPT="/data/minseo/experiments5/extended_schema/ours_memory/inference_api_singleturn_fixed_pairs.py"

INPUT_PATH="/data/minseo/experiments5/data/1229_dev_6.json"
QUERY_PATH="/data/minseo/experiments5/config/query_new_single.json"
FIXED_PAIRS_PATH="/data/minseo/experiments5/data/fixed_singleturn_example_query_pairs_400.json"
PREF_LIST_PATH="/data/minseo/experiments5/config/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments5/config/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments5/config/schema_easy_extended.json"

RUN_ROOT="/data/minseo/experiments5/extended_schema/ours_memory/output"
MEMORY_ROOT="${MEMORY_ROOT:-/data/minseo/experiments5/outputs/our_memory/memories}"
DATE_TAG="$(date +%m%d)"
TEST_TAG="${TEST_TAG:-fixed_400_rerun}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8003}"
BASE_URL="http://localhost:$PORT/v1"
CONCURRENCY="${CONCURRENCY:-50}"
MAX_QUERIES="${MAX_QUERIES:-400}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

MODELS=(
    #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "google/gemma-3-12b-it"
    "google/codegemma-7b-it"
)

MEMORY_FOLDERS=(
    "gpt-4o-mini"
)

# STOP_AFTER_MEMORY_FOLDER="${STOP_AFTER_MEMORY_FOLDER:-gpt-4o-mini}"
# STOP_RUN_AFTER_TARGET_MEMORY="${STOP_RUN_AFTER_TARGET_MEMORY:-1}"

CONTEXT_TYPES=("memory_api")
PREF_TYPES=("hard")
PROMPT_NAME="implicit_zs"

trap cleanup SIGINT SIGTERM ERR

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo ""
        echo "[WARN] Killing vLLM server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit 1
}

wait_for_server() {
    echo "Waiting for vLLM server at $BASE_URL..."
    local max_retries=120
    local count=0

    while ! curl -fsS --max-time 5 "$BASE_URL/models" > /dev/null 2>&1; do
        sleep 5
        echo -n "."
        count=$((count + 1))

        if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
            echo ""
            echo "[ERROR] vLLM server process died unexpectedly."
            cat "$SERVER_LOG_PATH"
            exit 1
        fi

        if [ "$count" -ge "$max_retries" ]; then
            echo ""
            echo "[ERROR] Timeout waiting for vLLM server."
            kill "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
    done

    echo ""
    echo ">> Server is READY!"
}

resolve_parser_flag() {
    local model="$1"

    if [[ "$model" == *"Llama-3"* ]]; then
        printf '%s\n' "--tool-call-parser llama3_json"
    elif [[ "$model" == *"Mistral"* ]]; then
        printf '%s\n' "--tool-call-parser mistral"
    else
        printf '%s\n' "--tool-call-parser hermes"
    fi
}

echo "========================================================"
echo "Starting Fixed-400 vLLM Inference (Single-turn Hard) at $(date)"
echo "Fixed Pairs: $FIXED_PAIRS_PATH"
echo "GPU: $GPU_IDS | Port: $PORT | Concurrency: $CONCURRENCY"
echo "========================================================"

STOP_REQUESTED=0

for model in "${MODELS[@]}"; do
    MODEL_SAFE_NAME="${model//\//_}"
    PARSER_FLAG="$(resolve_parser_flag "$model")"

    BASE_OUTPUT_DIR="$RUN_ROOT/singleturn/vllm"
    BASE_LOG_DIR="$RUN_ROOT/logs/singleturn/vllm"
    mkdir -p "$BASE_OUTPUT_DIR" "$BASE_LOG_DIR"

    SERVER_LOG_PATH="$BASE_LOG_DIR/${DATE_TAG}_${TEST_TAG}_${MODEL_SAFE_NAME}.vllm_server.log"

    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: $model"
    echo "####################################################################"

    CUDA_VISIBLE_DEVICES="$GPU_IDS" nohup vllm serve "$model" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --enable-auto-tool-choice \
        $PARSER_FLAG \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --trust-remote-code > "$SERVER_LOG_PATH" 2>&1 &

    SERVER_PID=$!
    echo ">> vLLM Server PID: $SERVER_PID"
    wait_for_server

    for MEM_FOLDER in "${MEMORY_FOLDERS[@]}"; do
        MEMORY_PATH="$MEMORY_ROOT/${MEM_FOLDER}/memory_result.jsonl"
        MEMORY_OUTPUT_DIR="$BASE_OUTPUT_DIR/${MEM_FOLDER}"
        MEMORY_LOG_DIR="$BASE_LOG_DIR/${MEM_FOLDER}"
        mkdir -p "$MEMORY_OUTPUT_DIR" "$MEMORY_LOG_DIR"

        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do
                CURRENT_OUT_DIR="$MEMORY_OUTPUT_DIR/$context/$pref/${MODEL_SAFE_NAME}_high/$PROMPT_NAME"
                CURRENT_LOG_DIR="$MEMORY_LOG_DIR/$context/$pref/${MODEL_SAFE_NAME}_high/$PROMPT_NAME"
                mkdir -p "$CURRENT_OUT_DIR" "$CURRENT_LOG_DIR"

                OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_${TEST_TAG}.json"
                LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_${TEST_TAG}.log"

                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --memory_path "$MEMORY_PATH" \
                    --output_path "$OUTPUT_FILE" \
                    --log_path "$LOG_FILE" \
                    --query_path "$QUERY_PATH" \
                    --fixed_pairs_path "$FIXED_PAIRS_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --model_name "$model" \
                    --base_url "$BASE_URL" \
                    --api_key "EMPTY" \
                    --concurrency "$CONCURRENCY" \
                    --max_queries "$MAX_QUERIES"
            done
        done

        if [ "${STOP_RUN_AFTER_TARGET_MEMORY}" = "1" ] && [ "$MEM_FOLDER" = "$STOP_AFTER_MEMORY_FOLDER" ]; then
            echo ">> Stop requested after memory source: $MEM_FOLDER"
            STOP_REQUESTED=1
            break
        fi
    done

    echo "[STEP 2] Stopping vLLM Server..."
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    echo ">> Server Stopped."

    if [ "${STOP_REQUESTED}" = "1" ]; then
        echo ">> Exiting after requested memory boundary: ${STOP_AFTER_MEMORY_FOLDER}"
        break
    fi

    echo "--------------------------------------------------------"
    sleep 10
done

echo "========================================================"
echo "All Fixed-400 vLLM Single-turn Jobs Finished at $(date)"
echo "========================================================"
