#!/bin/bash
# Run RAG pipeline: ChromaDB ingestion + API multiturn inference.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_PATH="$ROOT_DIR/data/dev.json"
MULTITURN_PATH="$ROOT_DIR/config/query_multiturn-domain.json"
PREF_LIST_PATH="$ROOT_DIR/config/pref_list.json"
PREF_GROUP_PATH="$ROOT_DIR/config/pref_group.json"
TOOLS_SCHEMA_PATH="$ROOT_DIR/config/schema_easy.json"
DB_PATH="$SCRIPT_DIR/chroma_db_rag"
COLLECTION_NAME="user_memories"

BASE_OUTPUT_DIR="$ROOT_DIR/outputs/rag/api/multiturn"
BASE_LOG_DIR="$ROOT_DIR/logs/rag/api/multiturn"
DATE_TAG="$(date +%m%d)"

MODEL="gpt-4o-mini-2024-07-18"
CONCURRENCY=20
RETRIEVAL_TOP_K=5
PREF_TYPES=("easy" "medium" "hard")
CONTEXT_TYPES=("diag-apilist")

echo "=== Step 2: Running RAG inference ==="
for pref in "${PREF_TYPES[@]}"; do
    for context in "${CONTEXT_TYPES[@]}"; do
        echo "[RUNNING] Pref: $pref | Context: $context"

        OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref"
        LOG_DIR="$BASE_LOG_DIR/$context/$pref"
        mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

        python "$SCRIPT_DIR/inference_api_multiturn.py" \
            --input_path "$INPUT_PATH" \
            --multiturn_path "$MULTITURN_PATH" \
            --pref_list_path "$PREF_LIST_PATH" \
            --pref_group_path "$PREF_GROUP_PATH" \
            --tools_schema_path "$TOOLS_SCHEMA_PATH" \
            --pref_type "$pref" \
            --context_type "$context" \
            --model_name "$MODEL" \
            --db_path "$DB_PATH" \
            --collection_name "$COLLECTION_NAME" \
            --rag_top_k $RETRIEVAL_TOP_K \
            --output_path "$OUTPUT_DIR/${DATE_TAG}.json" \
            --log_path "$LOG_DIR/${DATE_TAG}.jsonl" \
            --concurrency $CONCURRENCY \
        && echo "[SUCCESS] -> $OUTPUT_DIR/${DATE_TAG}.json" \
        || echo "[ERROR] Failed: $pref / $context"
    done
done

echo "All jobs finished at $(date)"
