#!/bin/bash
# Run RAG multi-turn inference via vLLM (OpenAI-compatible endpoint).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/inference_vllm_multiturn.py"

INPUT_PATH="$ROOT_DIR/data/dev.json"
MULTITURN_PATH="$ROOT_DIR/config/query_multiturn-domain.json"
PREF_LIST_PATH="$ROOT_DIR/config/pref_list.json"
PREF_GROUP_PATH="$ROOT_DIR/config/pref_group.json"
TOOLS_SCHEMA_PATH="$ROOT_DIR/config/schema_easy.json"
DB_PATH="$SCRIPT_DIR/chroma_db_rag"
COLLECTION_NAME="user_memories"

BASE_OUTPUT_DIR="$ROOT_DIR/outputs/rag/vllm/multiturn"
BASE_LOG_DIR="$ROOT_DIR/logs/rag/vllm/multiturn"

DATE_TAG="$(date +%m%d)"
VLLM_URL="http://localhost:8000/v1"
CONCURRENCY=50
RETRIEVAL_TOP_K=5

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
)

CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("easy" "medium" "hard")

echo "========================================================"
echo "RAG vLLM Multi-turn Inference Started at $(date)"
echo "vLLM URL: $VLLM_URL"
echo "========================================================"

for model in "${MODELS[@]}"; do
    MODEL_SAFE="${model//\//_}"

    for context in "${CONTEXT_TYPES[@]}"; do
        for pref in "${PREF_TYPES[@]}"; do
            echo "[RUNNING] Model: $model | Context: $context | Pref: $pref"

            OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/$MODEL_SAFE"
            LOG_DIR="$BASE_LOG_DIR/$context/$pref/$MODEL_SAFE"
            mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

            python $PYTHON_SCRIPT \
                --input_path $INPUT_PATH \
                --multiturn_path $MULTITURN_PATH \
                --pref_list_path $PREF_LIST_PATH \
                --pref_group_path $PREF_GROUP_PATH \
                --tools_schema_path $TOOLS_SCHEMA_PATH \
                --context_type $context \
                --pref_type $pref \
                --model_name $model \
                --vllm_url $VLLM_URL \
                --db_path $DB_PATH \
                --collection_name $COLLECTION_NAME \
                --rag_top_k $RETRIEVAL_TOP_K \
                --output_path $OUTPUT_DIR/${DATE_TAG}.json \
                --log_path $LOG_DIR/${DATE_TAG}.jsonl \
                --concurrency $CONCURRENCY \
            && echo "[SUCCESS] -> $OUTPUT_DIR/${DATE_TAG}.json" \
            || echo "[ERROR] Failed: $model / $context / $pref"
        done
    done
done

echo "All jobs finished at $(date)"
