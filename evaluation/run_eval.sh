#!/bin/bash
# Evaluate all output files and write a CSV summary.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PREF_LIST_PATH="$ROOT_DIR/config/pref_list.json"
OUTPUT_BASE="$ROOT_DIR/outputs"
RESULTS_DIR="$ROOT_DIR/results"
DATE_TAG="$(date +%m%d)"

mkdir -p "$RESULTS_DIR"

echo "=== Evaluating single-turn outputs ==="
python "$SCRIPT_DIR/eval_multiturn.py" \
    --input_glob "$OUTPUT_BASE/**/*.json" \
    --pref_list_path "$PREF_LIST_PATH" \
    --csv_output "$RESULTS_DIR/summary_${DATE_TAG}.csv"

echo "CSV summary -> $RESULTS_DIR/summary_${DATE_TAG}.csv"
echo "Done at $(date)"
