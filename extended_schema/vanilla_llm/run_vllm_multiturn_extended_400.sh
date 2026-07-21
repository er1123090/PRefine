#!/bin/bash

set -euo pipefail

PYTHON_SCRIPT="/data/minseo/experiments5/extended_schema/vanilla_llm/inference_api_multiturn_fixed_pairs.py"

INPUT_PATH="/data/minseo/experiments5/data/1229_dev_6.json"
MULTITURN_QUERY_PATH="/data/minseo/experiments5/config/query_new_multi.json"
FIXED_PAIRS_PATH="/data/minseo/experiments5/data/fixed_multiturn_example_query_pairs_400.json"
PREF_LIST_PATH="/data/minseo/experiments5/config/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments5/config/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments5/config/schema_all_extended_complete.json"

RUN_ROOT="/data/minseo/experiments5/extended_schema/vanilla_llm/output"
BASE_OUTPUT_DIR="$RUN_ROOT/multiturn/vllm"
BASE_LOG_DIR="$RUN_ROOT/logs/multiturn/vllm"

DATE_TAG="$(date +%m%d)"
TEST_TAG="${TEST_TAG:-fixed_400_rerun}"

GPU_ID="${GPU_ID:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8002}"
BASE_URL="http://localhost:$PORT/v1"
CONCURRENCY="${CONCURRENCY:-50}"
MAX_QUERIES="${MAX_QUERIES:-400}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"

MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    #"google/gemma-3-12b-it"
    #"google/codegemma-7b-it"
)

PROMPT_TYPES=("imp-zs")
CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("hard")

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
    local max_retries=300
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

resolve_max_model_len() {
    local model="$1"

    if [ -n "${VLLM_MAX_MODEL_LEN:-}" ]; then
        printf '%s\n' "$VLLM_MAX_MODEL_LEN"
    elif [[ "$model" == "google/gemma-3-12b-it" ]] || [[ "$model" == "google/codegemma-7b-it" ]]; then
        printf '%s\n' "8192"
    else
        printf '%s\n' "16384"
    fi
}

echo "========================================================"
echo "VanillaLLM Fixed-400 vLLM Multi-turn Started at $(date)"
echo "Fixed Pairs: $FIXED_PAIRS_PATH"
echo "GPU: $GPU_ID | Port: $PORT | Concurrency: $CONCURRENCY"
echo "========================================================"

mkdir -p "$BASE_OUTPUT_DIR" "$BASE_LOG_DIR"

for model in "${MODELS[@]}"; do
    MODEL_SAFE_NAME="${model//\//_}"
    MAX_MODEL_LEN="$(resolve_max_model_len "$model")"
    PARSER_FLAG="$(resolve_parser_flag "$model")"
    SERVER_LOG_PATH="$BASE_LOG_DIR/${DATE_TAG}_${TEST_TAG}_${MODEL_SAFE_NAME}.vllm_server.log"

    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: $model"
    echo "####################################################################"
    echo ">> Max model len: $MAX_MODEL_LEN"

    CUDA_VISIBLE_DEVICES="$GPU_ID" nohup vllm serve "$model" \
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

    for prompt_type in "${PROMPT_TYPES[@]}"; do
        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do
                CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/multiturn-query/$MODEL_SAFE_NAME/$prompt_type"
                CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/multiturn-query/$MODEL_SAFE_NAME/$prompt_type"
                mkdir -p "$CURRENT_OUT_DIR" "$CURRENT_LOG_DIR"

                OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_${TEST_TAG}.json"
                LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_${TEST_TAG}.jsonl"

                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --multiturn_path "$MULTITURN_QUERY_PATH" \
                    --fixed_pairs_path "$FIXED_PAIRS_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --prompt_type "$prompt_type" \
                    --model_name "$model" \
                    --base_url "$BASE_URL" \
                    --api_key "EMPTY" \
                    --output_path "$OUTPUT_FILE" \
                    --log_path "$LOG_FILE" \
                    --concurrency "$CONCURRENCY" \
                    --max_queries "$MAX_QUERIES"
            done
        done
    done

    echo "[STEP 2] Stopping vLLM Server..."
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    echo ">> Server Stopped."
    echo "--------------------------------------------------------"
    sleep 10
done

echo "========================================================"
echo "VanillaLLM Fixed-400 vLLM Multi-turn Finished at $(date)"
echo "========================================================"
