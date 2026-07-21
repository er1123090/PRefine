#!/bin/bash
# Ablation: Effect of verifier loop count (max_retries 0 to 5).
# Runs step1_extract.py for each max_retries value and saves to separate output dirs.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_SCRIPT="$SCRIPT_DIR/step1_extract.py"
INPUT_DATA="$ROOT_DIR/data/dev.json"
BASE_OUTPUT_DIR="$ROOT_DIR/outputs/our_memory/ablation_verifier"

# Model settings (edit as needed)
PROVIDER="openai"
MODEL_NAME="gpt-4o-mini"
CONCURRENCY=20

mkdir -p "$BASE_OUTPUT_DIR"

echo "========================================================"
echo "Verifier Loop Ablation (max_retries 0 to 5)"
echo "Model: $MODEL_NAME | Provider: $PROVIDER"
echo "Start: $(date)"
echo "========================================================"

for RETRY_COUNT in {0..5}; do

    echo "------------------------------------------------------------"
    echo "[RUNNING] max_retries=$RETRY_COUNT"
    echo "------------------------------------------------------------"

    CURRENT_DIR="$BASE_OUTPUT_DIR/retries_${RETRY_COUNT}"
    mkdir -p "$CURRENT_DIR"

    python "$PYTHON_SCRIPT" \
        --input "$INPUT_DATA" \
        --output "$CURRENT_DIR/memory_result.jsonl" \
        --verifier_output "$CURRENT_DIR/verifier_log.jsonl" \
        --refinement_output "$CURRENT_DIR/refinement_log.jsonl" \
        --provider "$PROVIDER" \
        --model "$MODEL_NAME" \
        --concurrency "$CONCURRENCY" \
        --max_retries "$RETRY_COUNT"

    echo "  -> Saved to: $CURRENT_DIR"
    sleep 3

done

echo "========================================================"
echo "Done at $(date)"
echo "========================================================"
