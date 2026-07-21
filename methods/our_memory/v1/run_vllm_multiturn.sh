#!/bin/bash
# Run Our Memory multi-turn inference via vLLM (OpenAI-compatible endpoint).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/inference_vllm_multiturn.py"

INPUT_PATH="$ROOT_DIR/data/dev.json"
MULTITURN_PATH="$ROOT_DIR/config/query_multiturn-domain.json"
PREF_LIST_PATH="$ROOT_DIR/config/pref_list.json"
PREF_GROUP_PATH="$ROOT_DIR/config/pref_group.json"
TOOLS_SCHEMA_PATH="$ROOT_DIR/config/schema_easy.json"
MEMORY_ROOT="${MEMORY_ROOT:-$ROOT_DIR/outputs/our_memory/memories}"

BASE_OUTPUT_DIR="$ROOT_DIR/outputs/our_memory/vllm/multiturn"
BASE_LOG_DIR="$ROOT_DIR/logs/our_memory/vllm/multiturn"

DATE_TAG="$(date +%m%d)"
VLLM_URL="http://localhost:8000/v1"
CONCURRENCY=50

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "google/gemma-3-12b-it"
)

CONTEXT_TYPES=("memory_only")
PREF_TYPES=("easy" "medium" "hard")

echo "========================================================"
echo "Our Memory vLLM Multi-turn Inference Started at $(date)"
echo "vLLM URL: $VLLM_URL"
echo "========================================================"

for model in "${MODELS[@]}"; do
    MODEL_SAFE="${model//\//_}"
    MEMORY_PATH="$MEMORY_ROOT/${MODEL_SAFE}/memory_result.jsonl"

    if [ ! -f "$MEMORY_PATH" ]; then
        echo "[SKIP] Memory not found for $model: $MEMORY_PATH"
        echo "       Build it with step1_extract.py or set MEMORY_ROOT."
        continue
    fi

    for context in "${CONTEXT_TYPES[@]}"; do
        for pref in "${PREF_TYPES[@]}"; do
            echo "[RUNNING] Model: $model | Context: $context | Pref: $pref"

            OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/$MODEL_SAFE"
            LOG_DIR="$BASE_LOG_DIR/$context/$pref/$MODEL_SAFE"
            mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

            python $PYTHON_SCRIPT \
                --input_path $INPUT_PATH \
                --memory_path $MEMORY_PATH \
                --multiturn_path $MULTITURN_PATH \
                --pref_list_path $PREF_LIST_PATH \
                --pref_group_path $PREF_GROUP_PATH \
                --tools_schema_path $TOOLS_SCHEMA_PATH \
                --context_type $context \
                --pref_type $pref \
                --model_name $model \
                --vllm_url $VLLM_URL \
                --output_path $OUTPUT_DIR/${DATE_TAG}.json \
                --log_path $LOG_DIR/${DATE_TAG}.jsonl \
                --concurrency $CONCURRENCY \
            && echo "[SUCCESS] -> $OUTPUT_DIR/${DATE_TAG}.json" \
            || echo "[ERROR] Failed: $model / $context / $pref"
        done
    done
done

echo "All jobs finished at $(date)"
