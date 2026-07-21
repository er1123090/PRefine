#!/bin/bash
# Run Our Memory pipeline: step1 (preference extraction) + step2 API multiturn inference.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_PATH="$ROOT_DIR/data/dev.json"
PREF_LIST_PATH="$ROOT_DIR/config/pref_list.json"
PREF_GROUP_PATH="$ROOT_DIR/config/pref_group.json"
TOOLS_SCHEMA_PATH="$ROOT_DIR/config/schema_easy.json"
MULTITURN_PATH="$ROOT_DIR/config/query_multiturn-domain.json"

MEMORY_ROOT="${MEMORY_ROOT:-$ROOT_DIR/outputs/our_memory/memories}"
BASE_OUTPUT_DIR="$ROOT_DIR/outputs/our_memory/api/multiturn"
BASE_LOG_DIR="$ROOT_DIR/logs/our_memory/api/multiturn"
DATE_TAG="$(date +%m%d)"

INFER_MODEL="gpt-4o-mini-2024-07-18"
MEMORY_MODELS=("${MEMORY_MODELS[@]:-gpt-4o-mini}")
CONCURRENCY_STEP2=20
PREF_TYPES=("easy" "medium" "hard")
CONTEXT_TYPES=("memory_only")

echo "=== Step 2: Running inference ==="
for MEM_MODEL in "${MEMORY_MODELS[@]}"; do
    MEMORY_PATH="$MEMORY_ROOT/$MEM_MODEL/memory_result.jsonl"

    if [ ! -f "$MEMORY_PATH" ]; then
        echo "[SKIP] Memory not found: $MEMORY_PATH"
        echo "       Build it with step1_extract.py or set MEMORY_ROOT/MEMORY_MODELS."
        continue
    fi

    echo "[MEMORY MODEL] $MEM_MODEL"
    for pref in "${PREF_TYPES[@]}"; do
        for context in "${CONTEXT_TYPES[@]}"; do
            echo "[RUNNING] Memory: $MEM_MODEL | Pref: $pref | Context: $context"

            OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/$MEM_MODEL"
            LOG_DIR="$BASE_LOG_DIR/$context/$pref/$MEM_MODEL"
            mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

            python "$SCRIPT_DIR/inference_api_multiturn.py" \
                --input_path "$INPUT_PATH" \
                --memory_path "$MEMORY_PATH" \
                --multiturn_path "$MULTITURN_PATH" \
                --pref_list_path "$PREF_LIST_PATH" \
                --pref_group_path "$PREF_GROUP_PATH" \
                --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                --pref_type "$pref" \
                --context_type "$context" \
                --model_name "$INFER_MODEL" \
                --output_path "$OUTPUT_DIR/${DATE_TAG}.json" \
                --log_path "$LOG_DIR/${DATE_TAG}.jsonl" \
                --concurrency $CONCURRENCY_STEP2 \
            && echo "[SUCCESS] -> $OUTPUT_DIR/${DATE_TAG}.json" \
            || echo "[ERROR] Failed: $MEM_MODEL / $pref / $context"
        done
    done
done

echo "All jobs finished at $(date)"
