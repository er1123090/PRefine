#!/bin/bash
# Run vanilla LLM single-turn inference via vLLM (OpenAI-compatible endpoint).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/inference_vllm_singleturn.py"

INPUT_PATH="$ROOT_DIR/data/dev.json"
QUERY_PATH="$ROOT_DIR/config/query_singleturn.json"
PREF_LIST_PATH="$ROOT_DIR/config/pref_list.json"
PREF_GROUP_PATH="$ROOT_DIR/config/pref_group.json"
TOOLS_SCHEMA_PATH="$ROOT_DIR/config/schema_easy.json"

BASE_OUTPUT_DIR="$ROOT_DIR/outputs/vanilla_llm/vllm/singleturn"
BASE_LOG_DIR="$ROOT_DIR/logs/vanilla_llm/vllm/singleturn"

DATE_TAG="$(date +%m%d)"
VLLM_URL="http://localhost:8000/v1"
CONCURRENCY=50

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("easy" "medium" "hard")

echo "========================================================"
echo "vLLM Batch Inference Started at $(date)"
echo "vLLM URL: $VLLM_URL"
echo "========================================================"

for model in "${MODELS[@]}"; do
    MODEL_SAFE="${model//\//_}"

    for context in "${CONTEXT_TYPES[@]}"; do
        for pref in "${PREF_TYPES[@]}"; do
            echo "[RUNNING] Model: $model | Pref: $pref"

            OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/$MODEL_SAFE"
            LOG_DIR="$BASE_LOG_DIR/$context/$pref/$MODEL_SAFE"
            mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

            python $PYTHON_SCRIPT \
                --input_path $INPUT_PATH \
                --query_path $QUERY_PATH \
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
            || echo "[ERROR] Failed: $model"
        done
    done
done

echo "All jobs finished at $(date)"
