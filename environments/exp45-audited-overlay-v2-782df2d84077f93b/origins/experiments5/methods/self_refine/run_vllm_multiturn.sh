#!/bin/bash
# Self-Refine multi-turn inference via vLLM (OpenAI-compatible endpoint).
# Both Phase 1 and Phase 2 use the same vLLM endpoint via --api_base.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_PATH="$ROOT_DIR/data/dev.json"
MULTITURN_PATH="$ROOT_DIR/config/query_multiturn-domain.json"
PREF_LIST_PATH="$ROOT_DIR/config/pref_list.json"
PREF_GROUP_PATH="$ROOT_DIR/config/pref_group.json"

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
)

PREF_TYPES=("easy" "medium" "hard")
PROVIDER="openai"
REFINE_ROUNDS=1
CONCURRENCY=10
VLLM_URL="http://localhost:8000/v1"

echo "########################################################################"
echo "Self-Refine vLLM Multi-turn Inference"
echo "Start: $(date)"
echo "vLLM URL: $VLLM_URL"
echo "########################################################################"

for model in "${MODELS[@]}"; do
    echo ">> Model: $model"

    for pref in "${PREF_TYPES[@]}"; do
        echo "   >> pref_type=$pref"

        DATE_TAG="$(date +%m%d)"
        MODEL_SAFE="${model//\//_}"
        OUT_DIR="$ROOT_DIR/outputs/self_refine/vllm/multiturn/$MODEL_SAFE/$pref"
        LOG_DIR="$ROOT_DIR/logs/self_refine/vllm/multiturn/$MODEL_SAFE/$pref"
        mkdir -p "$OUT_DIR" "$LOG_DIR"

        case "$pref" in
            easy)   SCHEMA="$ROOT_DIR/config/schema_easy.json" ;;
            medium) SCHEMA="$ROOT_DIR/config/schema_medium.json" ;;
            hard)   SCHEMA="$ROOT_DIR/config/schema_hard.json" ;;
        esac

        python "$SCRIPT_DIR/inference_api_multiturn.py" \
            --input_path "$INPUT_PATH" \
            --output_path "$OUT_DIR/${DATE_TAG}_result.json" \
            --log_path "$LOG_DIR/${DATE_TAG}_run.jsonl" \
            --multiturn_path "$MULTITURN_PATH" \
            --pref_list_path "$PREF_LIST_PATH" \
            --pref_group_path "$PREF_GROUP_PATH" \
            --tools_schema_path "$SCHEMA" \
            --pref_type "$pref" \
            --provider "$PROVIDER" \
            --model_name "$model" \
            --api_base "$VLLM_URL" \
            --api_key "dummy" \
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
