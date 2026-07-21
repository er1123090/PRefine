#!/bin/bash
# Self-Refine API singleturn inference.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_PATH="$ROOT_DIR/data/dev.json"
QUERY_PATH="$ROOT_DIR/config/query_singleturn.json"
PREF_LIST_PATH="$ROOT_DIR/config/pref_list.json"
PREF_GROUP_PATH="$ROOT_DIR/config/pref_group.json"

MODELS=(
    "gpt-4o-mini"
    # "gemini-2.0-flash"
)

PREF_TYPES=("easy" "medium" "hard")
PROVIDER="openai"
REFINE_ROUNDS=1
CONCURRENCY=10

echo "########################################################################"
echo "Self-Refine API Singleturn Inference"
echo "Start: $(date)"
echo "########################################################################"

for model in "${MODELS[@]}"; do
    echo ">> Model: $model"
    for pref in "${PREF_TYPES[@]}"; do
        echo "   >> pref_type=$pref"
        DATE_TAG="$(date +%m%d)"
        OUT_DIR="$ROOT_DIR/outputs/self_refine/api/singleturn/$model/$pref"
        LOG_DIR="$ROOT_DIR/logs/self_refine/api/singleturn/$model/$pref"
        mkdir -p "$OUT_DIR" "$LOG_DIR"

        case "$pref" in
            easy)   SCHEMA="$ROOT_DIR/config/schema_easy.json" ;;
            medium) SCHEMA="$ROOT_DIR/config/schema_medium.json" ;;
            hard)   SCHEMA="$ROOT_DIR/config/schema_hard.json" ;;
        esac

        python "$SCRIPT_DIR/inference_api_singleturn.py" \
            --input_path "$INPUT_PATH" \
            --output_path "$OUT_DIR/${DATE_TAG}_result.json" \
            --log_path "$LOG_DIR/${DATE_TAG}_run.jsonl" \
            --query_path "$QUERY_PATH" \
            --pref_list_path "$PREF_LIST_PATH" \
            --pref_group_path "$PREF_GROUP_PATH" \
            --tools_schema_path "$SCHEMA" \
            --pref_type "$pref" \
            --provider "$PROVIDER" \
            --model_name "$model" \
            --refine_rounds $REFINE_ROUNDS \
            --concurrency $CONCURRENCY \
        && echo "[SUCCESS] -> $OUT_DIR/${DATE_TAG}_result.json" \
        || echo "[ERROR] Failed: $model / $pref"
    done
    echo "--------------------------------------------------------"
done

echo "========================================================"
echo "Done at $(date)"
echo "========================================================"
