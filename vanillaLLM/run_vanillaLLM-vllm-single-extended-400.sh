#!/bin/bash

set -euo pipefail

# ==============================================================================
# 1. Environment & Paths
# ==============================================================================
PYTHON_SCRIPT="/data/minseo/experiments4/vanillaLLM/vanillaLLM_inference-vllm-single.py"

INPUT_PATH="/data/minseo/experiments4/data/1229_dev_6.json"
QUERY_PATH="/data/minseo/experiments4/query_singleturn_extended.json"
PREF_LIST_PATH="/data/minseo/experiments4/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments4/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments4/schema_easy_extended.json"

BASE_OUTPUT_DIR="/data/minseo/experiments4/vanillaLLM/inference/extended/outputs/singleturn/vllm"
BASE_LOG_DIR="/data/minseo/experiments4/vanillaLLM/inference/extended/logs/singleturn/vllm"

DATE_TAG="$(date +%m%d)"
TEST_TAG="${TEST_TAG:-extended_400}"

GPU_ID="${GPU_ID:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8001}"
VLLM_URL="http://localhost:$PORT/v1"
CONCURRENCY="${CONCURRENCY:-50}"
MAX_QUERIES="${MAX_QUERIES:-400}"

# ==============================================================================
# 2. Experiment Variables
# ==============================================================================
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
    "google/gemma-3-12b-it"
    "google/codegemma-7b-it"
)

PROMPT_TYPES=("imp-zs")
CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("hard")

# ==============================================================================
# 3. Helper Functions
# ==============================================================================
trap cleanup SIGINT SIGTERM ERR

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo ""
        echo "[WARN] Killing vLLM Server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit 1
}

wait_for_server() {
    echo "Waiting for vLLM server at $VLLM_URL..."
    local max_retries=300
    local count=0

    while ! curl -s "$VLLM_URL/models" > /dev/null; do
        sleep 5
        echo -n "."
        count=$((count + 1))

        if ! ps -p "$SERVER_PID" > /dev/null; then
            echo ""
            echo "[ERROR] vLLM Server process died unexpectedly."
            cat vllm_server_single_extended.log
            exit 1
        fi

        if [ "$count" -ge "$max_retries" ]; then
            echo ""
            echo "[ERROR] Timeout waiting for vLLM server."
            kill "$SERVER_PID"
            exit 1
        fi
    done

    echo ""
    echo ">> Server is READY!"
}

# ==============================================================================
# 4. Main Execution
# ==============================================================================
echo "========================================================"
echo "VanillaLLM Extended vLLM Single-turn Started at $(date)"
echo "Query File : $QUERY_PATH"
echo "Max Queries: $MAX_QUERIES"
echo "GPU: $GPU_ID | Port: $PORT | Concurrency: $CONCURRENCY"
echo "========================================================"

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_LOG_DIR"

for model in "${MODELS[@]}"; do
    MODEL_SAFE_NAME="${model//\//_}"

    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: $model"
    echo "####################################################################"

    PARSER_FLAG="--tool-call-parser hermes"
    if [[ "$model" == *"Llama-3"* ]]; then
        PARSER_FLAG="--tool-call-parser llama3_json"
    elif [[ "$model" == *"Mistral"* ]]; then
        PARSER_FLAG="--tool-call-parser mistral"
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup vllm serve "$model" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --enable-auto-tool-choice \
        $PARSER_FLAG \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code > vllm_server_single_extended.log 2>&1 &

    SERVER_PID=$!
    echo ">> vLLM Server PID: $SERVER_PID"
    wait_for_server

    echo "[STEP 2] Running Python Client Scripts..."
    for context in "${CONTEXT_TYPES[@]}"; do
        for prompt_type in "${PROMPT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do
                echo " >> [Processing] Context: $context | Prompt: $prompt_type | Pref: $pref"

                CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/singleturn-query/$MODEL_SAFE_NAME/$prompt_type"
                CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/singleturn-query/$MODEL_SAFE_NAME/$prompt_type"
                mkdir -p "$CURRENT_OUT_DIR"
                mkdir -p "$CURRENT_LOG_DIR"

                OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_${TEST_TAG}.json"
                LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_${TEST_TAG}.log"

                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --output_path "$OUTPUT_FILE" \
                    --log_path "$LOG_FILE" \
                    --query_path "$QUERY_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --prompt_type "$prompt_type" \
                    --model_name "$model" \
                    --vllm_url "$VLLM_URL" \
                    --concurrency "$CONCURRENCY" \
                    --max_queries "$MAX_QUERIES"
            done
        done
    done

    echo "[STEP 3] Stopping vLLM Server..."
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    echo ">> Server Stopped."
    echo "--------------------------------------------------------"
    sleep 10
done

echo "========================================================"
echo "VanillaLLM Extended vLLM Single-turn Finished at $(date)"
echo "========================================================"
