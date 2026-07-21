#!/bin/bash
# Ablation: Effect of context_type (memory_only / api_only / memory_api)
# and memory source (which step-1 model produced the memory file).
#
# Prerequisites: run step1_extract.py for each memory source model first,
# then set MEMORY_FOLDERS below to the corresponding output subdirectory names.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHON_SCRIPT="$SCRIPT_DIR/inference_api_singleturn.py"

INPUT_PATH="$ROOT_DIR/data/dev.json"
QUERY_PATH="$ROOT_DIR/config/query_singleturn.json"
PREF_LIST_PATH="$ROOT_DIR/config/pref_list.json"
PREF_GROUP_PATH="$ROOT_DIR/config/pref_group.json"
TOOLS_SCHEMA_PATH="$ROOT_DIR/config/schema_easy.json"

# Base directory where step1 JSONL memories are stored.
# Expected layout: $MEMORY_BASE/<folder_name>/memory_result.jsonl
MEMORY_BASE="$ROOT_DIR/outputs/our_memory"

BASE_OUTPUT_DIR="$ROOT_DIR/outputs/our_memory/ablation_context"
BASE_LOG_DIR="$ROOT_DIR/logs/our_memory/ablation_context"

# Inference models to evaluate with
MODELS=(
    "gpt-4o-mini"
    # "gemini-2.0-flash"
    # "claude-sonnet-4-6"
)

# Memory source folders (names under $MEMORY_BASE that contain memory_result.jsonl)
MEMORY_FOLDERS=(
    "gpt-4o-mini"
    # "gemini-2.0-flash"
    # "deepseek-r1"
)

# Ablation axis: context_type
CONTEXT_TYPES=("memory_only" "api_only" "memory_api")

PREF_TYPES=("easy" "medium" "hard")
CONCURRENCY=10

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_LOG_DIR"

echo "########################################################################"
echo "Context-Type Ablation (single-turn)"
echo "Start: $(date)"
echo "########################################################################"

for model in "${MODELS[@]}"; do
    echo ">> Inference Model: $model"

    for MEM_FOLDER in "${MEMORY_FOLDERS[@]}"; do
        MEMORY_PATH="$MEMORY_BASE/$MEM_FOLDER/memory_result.jsonl"

        if [[ ! -f "$MEMORY_PATH" ]]; then
            echo "   [SKIP] Memory file not found: $MEMORY_PATH"
            continue
        fi

        echo "   >> Memory Source: $MEM_FOLDER"

        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                DATE_TAG="$(date +%m%d)"
                OUT_DIR="$BASE_OUTPUT_DIR/$MEM_FOLDER/$context/$pref/$model"
                LOG_DIR="$BASE_LOG_DIR/$MEM_FOLDER/$context/$pref/$model"
                mkdir -p "$OUT_DIR" "$LOG_DIR"

                echo "      context=$context | pref=$pref"

                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --memory_path "$MEMORY_PATH" \
                    --output_path "$OUT_DIR/${DATE_TAG}_result.json" \
                    --log_path "$LOG_DIR/${DATE_TAG}_run.log" \
                    --query_path "$QUERY_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --model_name "$model" \
                    --concurrency "$CONCURRENCY"

            done
        done
    done

    echo "--------------------------------------------------------"
done

echo "========================================================"
echo "Done at $(date)"
echo "========================================================"
