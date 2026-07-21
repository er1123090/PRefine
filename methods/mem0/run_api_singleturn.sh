#!/bin/bash
# Run mem0 pipeline: ingestion + API singleturn inference.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_PATH="$ROOT_DIR/data/dev.json"
QUERY_PATH="$ROOT_DIR/config/query_singleturn.json"
PREF_LIST_PATH="$ROOT_DIR/config/pref_list.json"
PREF_GROUP_PATH="$ROOT_DIR/config/pref_group.json"
TOOLS_SCHEMA_PATH="$ROOT_DIR/config/schema_easy.json"

BASE_OUTPUT_DIR="$ROOT_DIR/outputs/mem0/api/singleturn"
BASE_LOG_DIR="$ROOT_DIR/logs/mem0/api/singleturn"
DATE_TAG="$(date +%m%d)"

MODEL="gpt-4o-mini-2024-07-18"
CONCURRENCY=20
PREF_TYPES=("easy" "medium" "hard")
CONTEXT_TYPES=("memory_only")

# Step 1: Ingest
echo "=== Step 1: Ingesting into mem0 ==="
python "$SCRIPT_DIR/step1_add.py" --input_path "$INPUT_PATH"

# Step 2: Inference
echo "=== Step 2: Running inference ==="
for pref in "${PREF_TYPES[@]}"; do
    for context in "${CONTEXT_TYPES[@]}"; do
        echo "[RUNNING] Pref: $pref | Context: $context"

        OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref"
        LOG_DIR="$BASE_LOG_DIR/$context/$pref"
        mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

        python "$SCRIPT_DIR/inference_api_singleturn.py" \
            --input_path "$INPUT_PATH" \
            --query_path "$QUERY_PATH" \
            --pref_list_path "$PREF_LIST_PATH" \
            --pref_group_path "$PREF_GROUP_PATH" \
            --tools_schema_path "$TOOLS_SCHEMA_PATH" \
            --pref_type "$pref" \
            --context_type "$context" \
            --model_name "$MODEL" \
            --output_path "$OUTPUT_DIR/${DATE_TAG}.json" \
            --log_path "$LOG_DIR/${DATE_TAG}.jsonl" \
            --concurrency $CONCURRENCY \
        && echo "[SUCCESS] -> $OUTPUT_DIR/${DATE_TAG}.json" \
        || echo "[ERROR] Failed: $pref / $context"
    done
done

echo "All jobs finished at $(date)"
