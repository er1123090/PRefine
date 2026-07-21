#!/bin/bash
# Run vanilla LLM multi-turn inference via API (OpenAI / Gemini / Anthropic).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/inference_api_multiturn.py"

INPUT_PATH="$ROOT_DIR/data/dev.json"
MULTITURN_PATH="$ROOT_DIR/config/query_multiturn-domain.json"
PREF_LIST_PATH="$ROOT_DIR/config/pref_list.json"
PREF_GROUP_PATH="$ROOT_DIR/config/pref_group.json"
TOOLS_SCHEMA_PATH="$ROOT_DIR/config/schema_easy.json"

BASE_OUTPUT_DIR="$ROOT_DIR/outputs/vanilla_llm/api/multiturn"
BASE_LOG_DIR="$ROOT_DIR/logs/vanilla_llm/api/multiturn"

DATE_TAG="$(date +%m%d)"
CONCURRENCY=20

MODELS=(
    "gpt-4o-mini-2024-07-18"
    # "gpt-4o-2024-11-20"
    # "gemini-2.0-flash"
    # "claude-3-5-sonnet-20241022"
)

CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("easy" "medium" "hard")

echo "========================================================"
echo "Batch Inference Started at $(date)"
echo "========================================================"

for model in "${MODELS[@]}"; do
    MODEL_SAFE="${model//\//_}"
    EFFORT_LEVELS=("default")

    if [[ "$model" == *"gpt-5"* ]] || [[ "$model" == *"o1"* ]] || \
       [[ "$model" == *"o3"* ]] || [[ "$model" == *"gemini"* ]] || \
       [[ "$model" == *"claude"* ]]; then
        EFFORT_LEVELS=("low")
    fi

    for context in "${CONTEXT_TYPES[@]}"; do
        for pref in "${PREF_TYPES[@]}"; do
            for effort in "${EFFORT_LEVELS[@]}"; do
                echo "[RUNNING] Model: $model | Pref: $pref | Effort: $effort"

                OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/$MODEL_SAFE-$effort"
                LOG_DIR="$BASE_LOG_DIR/$context/$pref/$MODEL_SAFE-$effort"
                mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

                CMD="python $PYTHON_SCRIPT \
                    --input_path $INPUT_PATH \
                    --multiturn_path $MULTITURN_PATH \
                    --pref_list_path $PREF_LIST_PATH \
                    --pref_group_path $PREF_GROUP_PATH \
                    --tools_schema_path $TOOLS_SCHEMA_PATH \
                    --context_type $context \
                    --pref_type $pref \
                    --model_name $model \
                    --output_path $OUTPUT_DIR/${DATE_TAG}.json \
                    --log_path $LOG_DIR/${DATE_TAG}.jsonl \
                    --concurrency $CONCURRENCY"

                if [ "$effort" != "default" ]; then
                    CMD="$CMD --reasoning_effort $effort"
                fi

                eval $CMD && echo "[SUCCESS] -> $OUTPUT_DIR/${DATE_TAG}.json" \
                          || echo "[ERROR] Failed: $model"
            done
        done
    done
done

echo "All jobs finished at $(date)"
